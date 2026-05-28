"""
Compute a PHA-aligned similarity metric through MambaVision/Transformer blocks.

This script feeds one selected input component:
  RGB / LL / LH / HL / HH
and captures per-layer outputs for each target module.

Compared with the previous block-delta version, this version follows the PHA
idea more closely:
  1. select top-K high-frequency image patches from the INPUT image/component
  2. keep that high-frequency set Omega fixed for a given layer grid size
  3. compute one layer-level similarity s on the layer OUTPUT only

Outputs are CSV summaries only. For the closest PHA-style comparison, prefer
using `--component RGB` and `--module_type block`.

`--image` can be a single image path or an image directory. In directory mode,
the script computes per-image raw results and dataset-level summaries.

Example:
python scripts/visualize_mamba_block_components.py ^
  --config_file configs/DukeMTMC/mambavision_tiny_transreid.yml ^
  --weight logs/Duke/exp/transformer_160.pth ^
  --image C:\\data\\DukeMTMC-reID\\query\\0001_c1_f0044158.jpg ^
  --output_dir C:\\tmp\\mamba_component_vis

# No checkpoint (randomly initialized model), single block behavior only
python scripts/visualize_mamba_block_components.py ^
  --config_file configs/DukeMTMC/mambavision_tiny_transreid.yml ^
  --image C:\\data\\DukeMTMC-reID\\query\\0001_c1_f0044158.jpg ^
  --output_dir C:\\tmp\\mamba_component_vis_nowt ^
  --no_weight --num_classes 702
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg  # noqa: E402
from model.make_model import make_model  # noqa: E402


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_haar_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernels = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],    # LL
            [[-0.5, -0.5], [0.5, 0.5]],  # LH
            [[-0.5, 0.5], [-0.5, 0.5]],  # HL
            [[0.5, -0.5], [-0.5, 0.5]],  # HH
        ],
        dtype=dtype,
        device=device,
    )
    return kernels.view(4, 1, 2, 2)


def haar_swt2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [1,C,H,W], value in [0,1]
    b, c, h, w = x.shape
    kernels = build_haar_kernels(x.device, x.dtype).repeat(c, 1, 1, 1)
    x_pad = F.pad(x, (0, 1, 0, 1), mode="replicate")
    coeff = F.conv2d(x_pad, kernels, stride=1, padding=0, groups=c)  # [1,4C,H,W]
    coeff = coeff.view(b, c, 4, h, w)
    ll = coeff[:, :, 0]
    lh = coeff[:, :, 1]
    hl = coeff[:, :, 2]
    hh = coeff[:, :, 3]
    return ll, lh, hl, hh


def normalize_component_for_net(x: torch.Tensor, kind: str) -> torch.Tensor:
    # x: [1,3,H,W]
    if kind == "LL":
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)
    if kind in ("LH", "HL", "HH"):
        max_abs = x.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (0.5 + 0.5 * x / max_abs).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


def apply_imagenet_norm(x: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def resize_and_to_tensor(image_path: Path, height: int, width: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    tr = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),  # [0,1]
        ]
    )
    return tr(img).unsqueeze(0)  # [1,3,H,W]


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if recursive:
        images = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        images = [p for p in input_path.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(images)


def safe_load_model_weights(model: torch.nn.Module, weight_path: Path) -> None:
    ckpt = torch.load(str(weight_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format.")

    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    model_state = model.state_dict()
    compatible = {k: v for k, v in cleaned.items() if k in model_state and model_state[k].shape == v.shape}
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    print(f"[Load] loaded compatible params: {len(compatible)}/{len(model_state)}")


def infer_hw_from_tokens(n: int) -> Tuple[int, int]:
    if n == 128:
        return 16, 8
    sq = int(np.sqrt(n))
    if sq * sq == n:
        return sq, sq
    return 1, n


def robust_normalize_map(arr: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(arr, q_low), np.percentile(arr, q_high)
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def feature_to_spatial(feat: torch.Tensor, h: Optional[int] = None, w: Optional[int] = None) -> torch.Tensor:
    # feat: [1,N,C] or [1,C,H,W] -> [1,C,H,W]
    if feat.dim() == 3:
        n = feat.shape[1]
        c = feat.shape[2]
        if h is None or w is None or h * w != n:
            h, w = infer_hw_from_tokens(n)
        return feat.transpose(1, 2).reshape(feat.shape[0], c, h, w)
    if feat.dim() == 4:
        return feat
    raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")


def feature_to_map(feat: torch.Tensor, h: Optional[int] = None, w: Optional[int] = None) -> np.ndarray:
    spatial = feature_to_spatial(feat, h, w)
    m = torch.sqrt((spatial[0] * spatial[0]).mean(dim=0) + 1e-12)
    arr = m.detach().float().cpu().numpy()
    return robust_normalize_map(arr)


def haar_swt2d_feature(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [1,C,H,W]
    b, c, h, w = x.shape
    kernels = build_haar_kernels(x.device, x.dtype).repeat(c, 1, 1, 1)
    x_pad = F.pad(x, (0, 1, 0, 1), mode="replicate")
    coeff = F.conv2d(x_pad, kernels, stride=1, padding=0, groups=c)
    coeff = coeff.view(b, c, 4, h, w)
    return coeff[:, :, 0], coeff[:, :, 1], coeff[:, :, 2], coeff[:, :, 3]


def normalize_feature_channels(x: torch.Tensor) -> torch.Tensor:
    # x: [1,C,H,W]
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def feature_to_hf_strength_map(feat: torch.Tensor, h: Optional[int] = None, w: Optional[int] = None) -> np.ndarray:
    spatial = normalize_feature_channels(feature_to_spatial(feat, h, w).float())
    _, lh, hl, hh = haar_swt2d_feature(spatial)
    hf = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1).squeeze(0)
    return hf.detach().cpu().numpy().astype(np.float32)

def image_to_patch_hf_strength_map(img_01: torch.Tensor, target_hw: Tuple[int, int]) -> np.ndarray:
    # img_01: [1,3,H,W] in [0,1]
    _, lh, hl, hh = haar_swt2d(img_01.float())
    hf = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1, keepdim=True)  # [1,1,H,W]
    hf = F.interpolate(hf, size=target_hw, mode="area")
    return hf.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def resolve_topk_count(num_tokens: int, pha_topk: int, pha_topk_ratio: float) -> int:
    if pha_topk > 0:
        return min(pha_topk, max(num_tokens - 1, 1))
    if pha_topk_ratio > 0:
        return min(max(int(round(num_tokens * pha_topk_ratio)), 1), max(num_tokens - 1, 1))
    return min(max(num_tokens // 8, 1), max(num_tokens - 1, 1))


def make_topk_mask(strength_map: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    flat = strength_map.reshape(-1)
    topk = min(max(int(topk), 1), max(flat.size - 1, 1))
    idx = np.argpartition(-flat, topk - 1)[:topk]
    mask = np.zeros_like(flat, dtype=np.float32)
    mask[idx] = 1.0
    return mask.reshape(strength_map.shape), idx.astype(np.int64)


def compute_pha_similarity(
    feat: torch.Tensor,
    selected_idx: np.ndarray,
    h: Optional[int] = None,
    w: Optional[int] = None,
) -> float:
    spatial = feature_to_spatial(feat, h, w).float()
    tokens = spatial.flatten(2).transpose(1, 2).squeeze(0)  # [N, C]
    num_tokens = tokens.shape[0]
    if num_tokens <= 1:
        return float("nan")

    selected = torch.as_tensor(selected_idx, device=tokens.device, dtype=torch.long)
    selected = torch.unique(selected)
    if selected.numel() == 0 or selected.numel() >= num_tokens:
        return float("nan")

    others_mask = torch.ones(num_tokens, device=tokens.device, dtype=torch.bool)
    others_mask[selected] = False
    others = torch.nonzero(others_mask, as_tuple=False).squeeze(1)
    if others.numel() == 0:
        return float("nan")

    tokens = F.normalize(tokens, dim=1, eps=1e-6)
    sim = tokens[selected] @ tokens[others].transpose(0, 1)
    return float(sim.abs().mean().item())


def build_component_inputs(
    img_tensor_01: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    ll, lh, hl, hh = haar_swt2d(img_tensor_01)
    comps_01 = {
        "RGB": img_tensor_01,
        "LL": normalize_component_for_net(ll, "LL"),
        "LH": normalize_component_for_net(lh, "LH"),
        "HL": normalize_component_for_net(hl, "HL"),
        "HH": normalize_component_for_net(hh, "HH"),
    }
    comps_net = {k: apply_imagenet_norm(v, mean, std) for k, v in comps_01.items()}
    return comps_01, comps_net


def parse_num_classes_from_ckpt(weight_path: Path, fallback: int = 751) -> int:
    try:
        ckpt = torch.load(str(weight_path), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        if isinstance(state, dict):
            for key in ("classifier.weight", "module.classifier.weight"):
                if key in state and state[key].dim() == 2:
                    return int(state[key].shape[0])
    except Exception:
        pass
    return fallback


def register_target_hooks(
    model: torch.nn.Module,
    captures: List[dict],
    name_regex: str,
    target_layer: str,
    max_layers: int,
    module_type: str,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    pattern = re.compile(name_regex) if name_regex else None

    def make_hook(layer_name: str):
        def hook(module, args, kwargs, output):
            x_in = args[0] if len(args) > 0 else None
            h = kwargs.get("H", None)
            w = kwargs.get("W", None)
            if x_in is None:
                return
            out = output[0] if isinstance(output, tuple) else output
            captures.append(
                {
                    "name": layer_name,
                    "x_in": x_in.detach().cpu(),
                    "x_out": out.detach().cpu(),
                    "H": int(h) if h is not None else None,
                    "W": int(w) if w is not None else None,
                }
            )
        return hook

    module_type_to_class = {
        "mixer": "MambaVisionMixer",
        "attention": "Attention",
        "block": "Block",
    }
    target_class = module_type_to_class[module_type]

    candidates: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != target_class:
            continue
        candidates.append((name, module))

    if target_layer:
        candidates = [(n, m) for n, m in candidates if n == target_layer]
    elif pattern is not None:
        candidates = [(n, m) for n, m in candidates if pattern.search(n) is not None]

    candidates = sorted(candidates, key=lambda x: x[0])
    if max_layers > 0:
        candidates = candidates[:max_layers]
    if candidates:
        print(f"[Hook] matched {module_type} layers:")
        for n, _ in candidates:
            print(f"  - {n}")
    else:
        print(f"[Hook] no {module_type} layers matched.")

    for name, module in candidates:
        try:
            h = module.register_forward_hook(make_hook(name), with_kwargs=True)
        except TypeError:
            # Older torch fallback: no kwargs, infer HW later.
            def old_hook(mod, args, output, layer_name=name):
                x_in = args[0] if len(args) > 0 else None
                if x_in is None:
                    return
                out = output[0] if isinstance(output, tuple) else output
                captures.append(
                    {
                        "name": layer_name,
                        "x_in": x_in.detach().cpu(),
                        "x_out": out.detach().cpu(),
                        "H": None,
                        "W": None,
                    }
                )
            h = module.register_forward_hook(old_hook)
        handles.append(h)
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a PHA-aligned token similarity for target layers.")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--weight", default="", type=str, help="Checkpoint path. Optional.")
    parser.add_argument("--no_weight", action="store_true", help="Do not load checkpoint weights.")
    parser.add_argument("--num_classes", default=751, type=int, help="Used when --no_weight is set.")
    parser.add_argument("--image", required=True, type=str, help="Path to an image file or image directory.")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan image directory.")
    parser.add_argument(
        "--component",
        default="RGB",
        choices=["RGB", "LL", "LH", "HL", "HH", "ALL"],
        help="Which input component to visualize. Default: RGB (single example).",
    )
    parser.add_argument(
        "--module_type",
        default="block",
        choices=["mixer", "attention", "block"],
        help="Target module type to hook. For the closest PHA-style comparison, use block.",
    )
    parser.add_argument("--name_regex", default="", type=str,
                        help="Regex for candidate module names when --target_layer is not set. Default depends on --module_type.")
    parser.add_argument("--target_layer", default="", type=str,
                        help="Exact layer name (e.g. base.levels.2.blocks.0.mixer).")
    parser.add_argument("--max_layers", default=1, type=int,
                        help="How many matched mixer layers to hook (default: 1).")
    parser.add_argument(
        "--pha_topk",
        default=0,
        type=int,
        help="Top-K high-frequency tokens used by the PHA-style similarity metric. 0 means auto.",
    )
    parser.add_argument(
        "--pha_topk_ratio",
        default=0.0,
        type=float,
        help="If > 0 and --pha_topk is 0, use this ratio of tokens as the high-frequency set.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Extra cfg options.")
    args = parser.parse_args()

    if not args.name_regex:
        if args.module_type == "mixer":
            args.name_regex = r"base\.levels\.[23]\..*mixer"
        elif args.module_type == "attention":
            args.name_regex = r"base\.levels\.[23]\.blocks\.\d+\.mixer"
        else:
            args.name_regex = r"base\.levels\.[23]\.blocks\.\d+$"

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    weight_path = Path(args.weight).expanduser().resolve() if args.weight else None
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(image_path, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {image_path}")

    no_weight_effective = bool(args.no_weight or weight_path is None)
    if no_weight_effective:
        num_classes = int(args.num_classes)
    else:
        num_classes = parse_num_classes_from_ckpt(weight_path)
    model = make_model(cfg, num_class=num_classes, camera_num=0, view_num=0)
    if not no_weight_effective:
        safe_load_model_weights(model, weight_path)
    else:
        print("[Load] checkpoint disabled (no --weight or --no_weight set). Using random-init model.")
    model = model.to(args.device)
    model.eval()

    h, w = int(cfg.INPUT.SIZE_TEST[0]), int(cfg.INPUT.SIZE_TEST[1])
    mean = list(cfg.INPUT.PIXEL_MEAN)
    std = list(cfg.INPUT.PIXEL_STD)

    all_component_rows: Dict[str, List[Dict[str, object]]] = {}

    with torch.no_grad():
        for img_idx, img_path in enumerate(image_paths, start=1):
            img_01 = resize_and_to_tensor(img_path, h, w).to(args.device)
            component_inputs_01, component_inputs_net = build_component_inputs(img_01, mean, std)

            if img_idx == 1:
                rgb_np = np.uint8(np.clip(img_01.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255.0)
                cv2.imwrite(str(output_dir / "input_resized_rgb.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            if args.component == "ALL":
                selected_components = [(k, component_inputs_01[k], component_inputs_net[k]) for k in component_inputs_net.keys()]
            else:
                selected_components = [(args.component, component_inputs_01[args.component], component_inputs_net[args.component])]

            rel_path = str(img_path.relative_to(image_path)) if image_path.is_dir() else img_path.name

            for comp_name, comp_01, inp in selected_components:
                captures: List[dict] = []
                hooks = register_target_hooks(
                    model=model,
                    captures=captures,
                    name_regex=args.name_regex,
                    target_layer=args.target_layer,
                    max_layers=args.max_layers,
                    module_type=args.module_type,
                )
                _ = model.base(inp, cam_label=None, view_label=None)
                for hk in hooks:
                    hk.remove()

                comp_dir = output_dir / comp_name
                comp_dir.mkdir(parents=True, exist_ok=True)
                if not captures:
                    print(f"[Warn] no captures for component {comp_name}. Check --name_regex.")
                    continue

                component_rows = all_component_rows.setdefault(comp_name, [])
                for idx, cap in enumerate(captures):
                    h_cap, w_cap = cap["H"], cap["W"]
                    if h_cap is None or w_cap is None:
                        spatial_out = feature_to_spatial(cap["x_out"])
                        h_cap, w_cap = int(spatial_out.shape[-2]), int(spatial_out.shape[-1])

                    omega_strength = image_to_patch_hf_strength_map(comp_01, target_hw=(h_cap, w_cap))
                    num_tokens = int(omega_strength.size)
                    pha_k = resolve_topk_count(num_tokens, args.pha_topk, args.pha_topk_ratio)
                    _, pha_idx = make_topk_mask(omega_strength, pha_k)
                    pha_sim = compute_pha_similarity(cap["x_out"], pha_idx, h_cap, w_cap)

                    component_rows.append(
                        {
                            "image_path": rel_path,
                            "component": comp_name,
                            "layer_idx": idx,
                            "layer_name": cap["name"],
                            "height": h_cap if h_cap is not None else "",
                            "width": w_cap if w_cap is not None else "",
                            "pha_topk": pha_k,
                            "pha_sim": pha_sim,
                        }
                    )

            if img_idx % 50 == 0 or img_idx == len(image_paths):
                print(f"[Progress] processed {img_idx}/{len(image_paths)} images")

    for comp_name, metric_rows in all_component_rows.items():
        comp_dir = output_dir / comp_name
        comp_dir.mkdir(parents=True, exist_ok=True)

        raw_csv_path = comp_dir / "pha_similarity_raw.csv"
        with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_path",
                    "component",
                    "layer_idx",
                    "layer_name",
                    "height",
                    "width",
                    "pha_topk",
                    "pha_sim",
                ],
            )
            writer.writeheader()
            writer.writerows(metric_rows)

        summary_rows: List[Dict[str, object]] = []
        grouped: Dict[Tuple[str, int, str, str, str, int], List[float]] = {}
        for row in metric_rows:
            key = (
                row["component"],
                int(row["layer_idx"]),
                str(row["layer_name"]),
                str(row["height"]),
                str(row["width"]),
                int(row["pha_topk"]),
            )
            grouped.setdefault(key, []).append(float(row["pha_sim"]))

        for key, values in sorted(grouped.items(), key=lambda kv: kv[0][1]):
            component, layer_idx, layer_name, height_s, width_s, pha_topk = key
            arr = np.asarray(values, dtype=np.float32)
            summary_rows.append(
                {
                    "component": component,
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "height": height_s,
                    "width": width_s,
                    "pha_topk": pha_topk,
                    "num_images": int(arr.size),
                    "pha_sim_mean": float(arr.mean()),
                    "pha_sim_std": float(arr.std()),
                    "pha_sim_min": float(arr.min()),
                    "pha_sim_max": float(arr.max()),
                }
            )

        summary_csv_path = comp_dir / "pha_similarity_summary.csv"
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "component",
                    "layer_idx",
                    "layer_name",
                    "height",
                    "width",
                    "pha_topk",
                    "num_images",
                    "pha_sim_mean",
                    "pha_sim_std",
                    "pha_sim_min",
                    "pha_sim_max",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print(
            f"[Done] {comp_name}: raw={len(metric_rows)} rows -> {raw_csv_path}, "
            f"summary={len(summary_rows)} rows -> {summary_csv_path}"
        )

    print(f"[All Done] outputs: {output_dir}")


if __name__ == "__main__":
    main()
