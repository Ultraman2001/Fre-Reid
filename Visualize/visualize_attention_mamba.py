import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 动态添加上级目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import cfg
from model.make_model import make_model


# Channel aggregation mode for attention map extraction.
# Options: "rms", "mean", "rms_mean"
AGG_MODE = "mean"
RMS_WEIGHT = 0.0


def normalize_scoremap(score_map, lower_pct=15.0, upper_pct=98.5, gamma=1.6):
    """Normalize a 2D score map with aggressive contrast enhancement."""
    score_map = score_map.astype(np.float32)
    lo = np.percentile(score_map, lower_pct)
    hi = np.percentile(score_map, upper_pct)
    if hi - lo < 1e-12:
        return np.zeros_like(score_map, dtype=np.float32)

    score_map = np.clip(score_map, lo, hi)
    score_map = (score_map - lo) / (hi - lo)
    score_map = np.power(score_map, gamma)
    return score_map


def suppress_low_res_attention_sink(am):
    """Suppress top-left sink artifact in low-resolution attention maps."""
    h, w = am.shape
    sink_ch = max(2, int(h * 0.15))
    sink_cw = max(2, int(w * 0.16))

    safe_region = am.copy()
    safe_region[:sink_ch, :sink_cw] = np.min(am)

    safe_min = np.min(safe_region)
    safe_mean = np.mean(safe_region)
    fill_value = safe_min * 0.6 + safe_mean * 0.4

    am[:sink_ch, :sink_cw] = fill_value
    return am


def suppress_right_edge_artifact(am, edge_ratio=0.12):
    """Replace the right border activation with background-level intensity."""
    h, w = am.shape
    edge_w = max(1, int(w * edge_ratio))
    if edge_w >= w:
        return am

    safe_region = am[:, : w - edge_w]
    safe_min = np.min(safe_region)
    safe_mean = np.mean(safe_region)
    fill_value = safe_min * 0.6 + safe_mean * 0.4

    am[:, w - edge_w :] = fill_value
    return am


def apply_colormap_on_image(org_img_np, activation_map, alpha=0.48, bg_darken=0.5, peak_clip=0.75):
    """Resize, smooth and overlay the heatmap with high contrast."""
    height, width, _ = org_img_np.shape

    am = cv2.resize(activation_map, (width, height), interpolation=cv2.INTER_CUBIC)

    blur_kernel = int(min(height, width) * 0.09)
    blur_kernel = min(max(blur_kernel, 7), 15)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    am = cv2.GaussianBlur(am, (blur_kernel, blur_kernel), sigmaX=0)

    am = np.clip(am, 0, 1)

    # Directly cap top responses to avoid overly saturated red regions.
    am = np.minimum(am, peak_clip)

    am_u8 = np.uint8(255 * am)

    am_colored = cv2.applyColorMap(am_u8, cv2.COLORMAP_TURBO)
    am_colored = cv2.cvtColor(am_colored, cv2.COLOR_BGR2RGB)

    base = org_img_np.astype(np.float32) * bg_darken
    overlapped = base * (1.0 - alpha) + am_colored.astype(np.float32) * alpha
    overlapped = np.clip(overlapped, 0, 255).astype(np.uint8)

    return overlapped


def get_attention_hook(module, input, output, attention_maps):
    """
    Hook function to capture outputs of specific layers during forward pass.
    将捕获到的特征地图存储到外部列表中。
    """
    # Detach to avoid memory leaks
    if isinstance(output, tuple):
        out = output[0].detach()
    else:
        out = output.detach()
    attention_maps.append((module.__class__.__name__, out))


def register_attention_hooks(model, attention_maps):
    """Register hooks on stage-4 MambaVision mixer/attention modules."""
    hooks = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "MambaVisionMixer" in class_name or "Attention" in class_name:
            # 仅截取 Stage 4
            if "levels.3" in name or "layers.3" in name: 
                # 使用带状态的挂钩闭包传递 attention_maps 容器
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: get_attention_hook(m, i, o, attention_maps)
                )
                hooks.append(hook)
    return hooks


def process_image(img_path, target_size=(256, 128)):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    img_pil = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    input_tensor = transform(img_pil).unsqueeze(0)
    
    orig_img_resized = img_pil.resize((target_size[1], target_size[0]))
    orig_img_np = np.array(orig_img_resized)
    
    return input_tensor, orig_img_np


def aggregate_channel_map(x, mode="rms_mean", rms_weight=0.6):
    """Aggregate channel responses into a 2D map with selectable strategy."""
    if x.ndim == 3:
        # [B, N, C] -> [B, N]
        rms = torch.sqrt((x ** 2).mean(dim=2) + 1e-12)
        mean_abs = x.abs().mean(dim=2)
    elif x.ndim == 4:
        # [B, C, H, W] -> [B, H, W]
        rms = torch.sqrt((x ** 2).mean(dim=1) + 1e-12)
        mean_abs = x.abs().mean(dim=1)
    else:
        raise ValueError(f"Unsupported tensor shape for aggregation: {tuple(x.shape)}")

    if mode == "rms":
        out = rms
    elif mode == "mean":
        out = mean_abs
    elif mode == "rms_mean":
        out = rms_weight * rms + (1.0 - rms_weight) * mean_abs
    else:
        raise ValueError(f"Unknown AGG_MODE: {mode}")

    return out


def visualize_attention_layers(model, img_paths, save_path, target_size=(256, 128)):
    """
    使用 Hook 从 MambaVision 网络中捕获纯净 Attention 演化特征并作图。
    """
    model.eval()
    visualize_dir = os.path.dirname(os.path.abspath(__file__))
    
    for img_idx, img_path in enumerate(img_paths):
        print(f"\nProcessing image: {img_path}")
        
        # 1. 挂载 Hook，创建一个空的容器，随推理动态注入值
        attention_maps = []
        hooks = register_attention_hooks(model, attention_maps)
        
        if len(hooks) == 0:
            print("Warning: 未能在模型中找到匹配 `MambaVisionMixer` 相关的层，无法执行 Hook！")
            return
            
        print(f"成功挂载 {len(hooks)} 个钩子 (Hooks) 准备监听 Mamba 内部演进...")
        
        # 2. 前置处理与模型推理
        input_tensor, orig_img_np = process_image(img_path, target_size)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        with torch.no_grad():
            _ = model.base(input_tensor) # 我们只关心 backbone，结果自然留在 attention_maps

        # 清理钩子，避免影响之后的操作
        for h in hooks:
            h.remove()
            
        if not attention_maps:
            print("Error: Hook 未搜集到任何数据！请检查网络模块名。")
            continue
            
        # 我们提取最有代表性的【Final Backbone Attention Map】
        # 根据需求：取 stage 4 中最后 2 个 attention block
        display_maps = attention_maps[-2:] if len(attention_maps) > 1 else attention_maps
        num_layers = max(1, len(display_maps))
        
        h_orig, w_orig = target_size
        combined_attn_map = np.zeros((h_orig, w_orig))

        # 输出路径统一为 Visualize/<image_name>/
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(visualize_dir, img_name)
        os.makedirs(out_dir, exist_ok=True)

        # 同目录保存一份原图，便于与热力图直接对照
        original_out_path = os.path.join(out_dir, "original.png")
        cv2.imwrite(original_out_path, cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR))
        
        for i, (layer_name, attn_output) in enumerate(display_maps):
            if len(attn_output.shape) == 3:
                seq_length = attn_output.shape[1]
                if seq_length == 128:
                    feat_h, feat_w = 16, 8
                else:
                    feat_w = int(np.sqrt(seq_length / 2))
                    feat_h = feat_w * 2

                current_attn_map = aggregate_channel_map(
                    attn_output, mode=AGG_MODE, rms_weight=RMS_WEIGHT
                )
                current_attn_map = current_attn_map[0].view(feat_h, feat_w).cpu().numpy()
                
            elif len(attn_output.shape) == 4:
                current_attn_map = aggregate_channel_map(
                    attn_output, mode=AGG_MODE, rms_weight=RMS_WEIGHT
                )
                current_attn_map = current_attn_map[0].cpu().numpy()
            else:
                continue

            current_attn_map = suppress_low_res_attention_sink(current_attn_map)
            current_attn_map = suppress_right_edge_artifact(current_attn_map)
            current_attn_map = cv2.resize(current_attn_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            
            norm_map = normalize_scoremap(current_attn_map)
            combined_attn_map += norm_map
            
            # 也可以单独存下每一层的图 (可选)
            # layer_overlay = apply_colormap_on_image(orig_img_np, norm_map)
            # cv2.imwrite(os.path.join(out_dir, f"{img_name}_{layer_name}_L{i}.png"), cv2.cvtColor(layer_overlay, cv2.COLOR_RGB2BGR))

        # 重点：计算大一统的纯 Backbone 特征注意力！
        combined_attn_map /= num_layers
        combined_overlay = apply_colormap_on_image(orig_img_np, combined_attn_map)

        combined_out_path = os.path.join(out_dir, "mamba_backbone_combined.png")
        cv2.imwrite(combined_out_path, cv2.cvtColor(combined_overlay, cv2.COLOR_RGB2BGR))
        
        print(f" => 原图已保存至: {original_out_path}")
        print(f" => MambaVision 纯主干最终注意力图 (Combined from layers) 已保存至: {combined_out_path}\n")

    print(f"\n=> 已完成，输出位于: {visualize_dir}/<图片名>/")


def main():
    parser = argparse.ArgumentParser(description="MambaVision Internal Attention Hook Visualizer")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    # 额外兼容一下直接传 yml 作为第一个位置参数的情况
    parser.add_argument("config_file_pos", nargs="?", default="", help="path to config file (positional)")
    parser.add_argument("--img_paths", default="", help="comma-separated paths to image files", type=str)
    parser.add_argument("--weights", default="weights_placeholder.pth", help="path to trained model weights", type=str)
    parser.add_argument("--output", default="mamba_hook_results.png", help="(Unused, saves per image)", type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    # 使用 parse_known_args 完美地把属于 parser 的参数和属于 YACS 的未知参数分开
    args, unknown_args = parser.parse_known_args()

    # 兼容 positional 参数
    config_file = args.config_file if args.config_file else args.config_file_pos

    if config_file != "":
        cfg.merge_from_file(config_file)
        
    # 余下的未知参数全是给 YACS 的 opts
    if unknown_args and len(unknown_args) > 0:
        cfg.merge_from_list(unknown_args)
    
    cfg.TEST.WEIGHT = args.weights

    # Force visualization input resolution for consistent cross-run comparison.
    cfg.INPUT.SIZE_TEST = [256, 128]
    
    # 【强制关闭 SFM】
    # 以表明我们在做纯 MambaVision Network 的 Ablation 可视化研究。
    cfg.MODEL.MAMBAVISION.USE_SFM = False

    print("Building model (Pure Backbone Mode)...")
    # 自动适配 Market-1501 和 DukeMTMC 数据集
    dataset_name = cfg.DATASETS.NAMES
    if "Duke" in dataset_name or "duke" in dataset_name:
        num_class = 702
        camera_num = 8
        print("=> Detected DukeMTMC dataset, using num_class=702, camera_num=8")
    else:
        num_class = 751
        camera_num = 6
        print("=> Detected Market-1501 dataset, using num_class=751, camera_num=6")
        
    model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        model.load_param(args.weights)
    else:
        print(f"Warning: Weights file '{args.weights}' not found. Heatmaps will be random!")

    if torch.cuda.is_available():
        model = model.cuda()
        
    img_list = args.img_paths.split(',') if args.img_paths else []
    
    if not img_list or not os.path.exists(img_list[0]):
        print("未提供有效图片路径，生成虚拟数据测试...")
        dummy_path = "dummy_test.jpg"
        dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        Image.fromarray(dummy_img).save(dummy_path)
        img_list = [dummy_path]

    visualize_attention_layers(model, img_list, args.output, target_size=tuple(cfg.INPUT.SIZE_TEST))

if __name__ == "__main__":
    main()
