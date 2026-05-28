"""
Export Haar wavelet sub-band images (LL/LH/HL/HH) for ReID inputs.

Usage examples:
    # Single image, network-friendly same-size SWT (default)
    python scripts/export_wavelet_images.py ^
        --input C:\\data\\sample.jpg ^
        --output_dir C:\\tmp\\wavelet_vis --save_residual

    # Directory (recursive), fixed ReID size 256x128
    python scripts/export_wavelet_images.py ^
        --input C:\\data\\market1501\\bounding_box_train ^
        --output_dir C:\\tmp\\wavelet_vis ^
        --height 256 --width 128 --recursive --save_residual

    # DWT mode (downsampled), then upsample for visualization
    python scripts/export_wavelet_images.py ^
        --input C:\\data\\sample.jpg ^
        --output_dir C:\\tmp\\wavelet_vis ^
        --mode dwt --upsample_dwt --save_residual
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_haar_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # LL / LH / HL / HH
    kernels = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[-0.5, -0.5], [0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[0.5, -0.5], [-0.5, 0.5]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernels.view(4, 1, 2, 2)


def haar_swt2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stationary-style Haar decomposition (no spatial downsampling).
    x: [1, C, H, W]
    returns LL/LH/HL/HH, each [1, C, H, W]
    """
    b, c, h, w = x.shape
    kernels = build_haar_kernels(x.device, x.dtype).repeat(c, 1, 1, 1)
    x_pad = F.pad(x, (0, 1, 0, 1), mode="replicate")
    coeff = F.conv2d(x_pad, kernels, stride=1, padding=0, groups=c)  # [1, 4C, H, W]
    coeff = coeff.view(b, c, 4, h, w)
    ll = coeff[:, :, 0]
    lh = coeff[:, :, 1]
    hl = coeff[:, :, 2]
    hh = coeff[:, :, 3]
    return ll, lh, hl, hh


def haar_dwt2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard Haar DWT with stride=2.
    x: [1, C, H, W]
    returns LL/LH/HL/HH, each [1, C, H/2, W/2] (after even-size padding if needed)
    """
    b, c, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    _, _, hp, wp = x.shape
    kernels = build_haar_kernels(x.device, x.dtype).repeat(c, 1, 1, 1)
    coeff = F.conv2d(x, kernels, stride=2, padding=0, groups=c)  # [1, 4C, Hp/2, Wp/2]
    coeff = coeff.view(b, c, 4, hp // 2, wp // 2)
    ll = coeff[:, :, 0]
    lh = coeff[:, :, 1]
    hl = coeff[:, :, 2]
    hh = coeff[:, :, 3]
    return ll, lh, hl, hh


def normalize_ll_for_vis(x: torch.Tensor) -> torch.Tensor:
    # LL is mostly non-negative; use min-max.
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


def normalize_hf_for_vis(x: torch.Tensor) -> torch.Tensor:
    # Signed high-frequency map -> map to [0,1] around 0.5.
    max_abs = x.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (0.5 + 0.5 * x / max_abs).clamp(0.0, 1.0)


def normalize_hf_with_gain_for_vis(x: torch.Tensor, gain: float) -> torch.Tensor:
    # Signed map with manual gain around 0.5 for easier visual inspection.
    max_abs = x.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (0.5 + 0.5 * gain * x / max_abs).clamp(0.0, 1.0)


def normalize_abs_for_vis(x: torch.Tensor) -> torch.Tensor:
    # --- FIXED VERSION ---
    # Absolute-valued map -> global min-max normalization into [0,1].
    # Use global maximum across all channels to preserve relative intensity.
    x_abs = x.abs()
    x_min = x_abs.amin()  # 全局最小值
    x_max = x_abs.amax()  # 全局最大值
    return (x_abs - x_min) / (x_max - x_min + 1e-6)
    # ---------------------


def save_tensor_as_image(x: torch.Tensor, path: Path) -> None:
    # x: [C,H,W], expected in [0,1]
    img = TF.to_pil_image(x.cpu().clamp(0.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if recursive:
        all_files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        all_files = [p for p in input_path.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(all_files)


def process_one_image(
    img_path: Path,
    input_root: Path,
    output_root: Path,
    height: int,
    width: int,
    mode: str,
    upsample_dwt: bool,
    save_tensor: bool,
    save_residual: bool,
    residual_gain: float,
) -> None:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    x = TF.to_tensor(img).unsqueeze(0)  # [1,3,H,W], [0,1]

    if mode == "swt":
        ll, lh, hl, hh = haar_swt2d(x)
    else:
        ll, lh, hl, hh = haar_dwt2d(x)
        if upsample_dwt:
            ll = F.interpolate(ll, size=(height, width), mode="bilinear", align_corners=False)
            lh = F.interpolate(lh, size=(height, width), mode="bilinear", align_corners=False)
            hl = F.interpolate(hl, size=(height, width), mode="bilinear", align_corners=False)
            hh = F.interpolate(hh, size=(height, width), mode="bilinear", align_corners=False)

    if input_root.is_dir():
        rel = img_path.relative_to(input_root)
        sample_dir = output_root / rel.parent / rel.stem
    else:
        sample_dir = output_root / img_path.stem

    save_tensor_as_image(x.squeeze(0), sample_dir / "input_resized.png")
    save_tensor_as_image(normalize_ll_for_vis(ll.squeeze(0)), sample_dir / "LL.png")
    save_tensor_as_image(normalize_hf_for_vis(lh.squeeze(0)), sample_dir / "LH.png")
    save_tensor_as_image(normalize_hf_for_vis(hl.squeeze(0)), sample_dir / "HL.png")
    save_tensor_as_image(normalize_hf_for_vis(hh.squeeze(0)), sample_dir / "HH.png")

    if save_residual:
        # Residual shows details removed by low-frequency branch.
        ll_for_residual = ll
        if ll_for_residual.shape[-2:] != x.shape[-2:]:
            ll_for_residual = F.interpolate(ll_for_residual, size=x.shape[-2:], mode="bilinear", align_corners=False)

        residual = x.squeeze(0) - (ll_for_residual.squeeze(0) / 2.0)
        save_tensor_as_image(normalize_hf_for_vis(residual), sample_dir / "residual_signed.png")
        save_tensor_as_image(normalize_abs_for_vis(residual), sample_dir / "residual_abs.png")
        save_tensor_as_image(normalize_hf_with_gain_for_vis(residual, residual_gain), sample_dir / "residual_signed_boosted.png")
        save_tensor_as_image(normalize_abs_for_vis(x.squeeze(0) - normalize_ll_for_vis(ll_for_residual.squeeze(0))), sample_dir / "input_vs_ll_vis_absdiff.png")

    if save_tensor:
        torch.save(
            {
                "ll": ll.squeeze(0).cpu(),
                "lh": lh.squeeze(0).cpu(),
                "hl": hl.squeeze(0).cpu(),
                "hh": hh.squeeze(0).cpu(),
            },
            sample_dir / "bands.pt",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Haar wavelet sub-band images for ReID inputs.")
    parser.add_argument("--input", required=True, type=str, help="Path to an image file or image directory.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs.")
    parser.add_argument("--height", type=int, default=256, help="Resize height (network input height).")
    parser.add_argument("--width", type=int, default=128, help="Resize width (network input width).")
    parser.add_argument("--mode", type=str, default="swt", choices=["swt", "dwt"], help="Wavelet mode.")
    parser.add_argument("--upsample_dwt", action="store_true", help="Upsample DWT sub-bands to input size for visualization.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directory for images.")
    parser.add_argument("--save_tensor", action="store_true", help="Also save LL/LH/HL/HH tensors as bands.pt.")
    parser.add_argument("--save_residual", action="store_true", help="Also save residual maps (input-LL) for visualization.")
    parser.add_argument("--residual_gain", type=float, default=8.0, help="Gain for residual_signed_boosted.png visualization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")

    print(f"[Wavelet Export] mode={args.mode}, size=({args.height},{args.width}), images={len(images)}")
    for idx, img_path in enumerate(images, start=1):
        process_one_image(
            img_path=img_path,
            input_root=input_path,
            output_root=output_root,
            height=args.height,
            width=args.width,
            mode=args.mode,
            upsample_dwt=args.upsample_dwt,
            save_tensor=args.save_tensor,
            save_residual=args.save_residual,
            residual_gain=args.residual_gain,
        )
        if idx % 50 == 0 or idx == len(images):
            print(f"[Wavelet Export] processed {idx}/{len(images)}")

    print(f"[Wavelet Export] done. outputs -> {output_root}")


if __name__ == "__main__":
    main()