import os
import sys
# 动态添加上级目录到 sys.path，解决在 Visualize 目录下运行时找不到 config 和 model 的问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

# 导入必要模块
from config import cfg
from model.make_model import make_model


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
    """
    处理序列视觉模型（如 ViT/Mamba）特有的 Attention Sink（注意力黑洞）伪影。
    模型常常把被丢弃的背景/无效特征推到起始的 Token（也就是左上角 0,0 附近 Patch），
    导致在最终的低分辨率连续激活图中，左上角的值异常之大，掩盖了人体真实响应点。
    
    我们在低分辨率 (如 24x8) 特征图阶段，将左上角的响应峰值
    强行限制为全图背景强度的均值，彻底将其从视觉极值中消除。
    """
    h, w = am.shape
    
    # 定义 Attention Sink 潜在的波及 Patch 范围 (左上角)
    # 考虑到低分辨率和高斯模糊的扩散，范围稍微拉宽到至少 2x2 patch
    sink_ch = max(2, int(h * 0.15))
    sink_cw = max(2, int(w * 0.16))
    
    # 提取除了左上角黑洞区域以外的“安全验证区域”
    safe_region = am.copy()
    safe_region[:sink_ch, :sink_cw] = np.min(am)
    
    # 用周围背景的基础强度（均值与最小值的折中）填平黑洞
    safe_min = np.min(safe_region)
    safe_mean = np.mean(safe_region)
    fill_value = safe_min * 0.6 + safe_mean * 0.4
    
    # 彻底填平左上角的伪影
    am[:sink_ch, :sink_cw] = fill_value
    
    return am


def get_attention_hook(module, input, output, attention_maps):
    """
    Hook function to capture outputs of specific layers during forward pass.
    将捕获到的特征地图存储到外部列表中。
    """
    if isinstance(output, tuple):
        out = output[0].detach()
    else:
        out = output.detach()
    attention_maps.append((module.__class__.__name__, out))

def register_attention_hooks(model, attention_maps):
    """
    劫持 MambaVision 最深层 Block 的核心特征 (同 visualize_attention_mamba)
    """
    hooks = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "MambaVisionMixer" in class_name or "Attention" in class_name:
            if "levels.3" in name or "layers.3" in name: 
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: get_attention_hook(m, i, o, attention_maps)
                )
                hooks.append(hook)
    return hooks

def compute_activation_map_from_hooks(attention_maps, target_size=(256, 128)):
    """
    将 Hook 劫持到的最后的特征计算为一个 Combined 注意力图
    """
    display_maps = attention_maps[-1:] if len(attention_maps) > 0 else attention_maps
    num_layers = max(1, len(display_maps))
    h_orig, w_orig = target_size
    combined_attn_map = np.zeros((h_orig, w_orig))

    for layer_name, attn_output in display_maps:
        if len(attn_output.shape) == 3:
            seq_length = attn_output.shape[1]
            if seq_length == 128:
                feat_h, feat_w = 16, 8
            else:
                feat_w = int(np.sqrt(seq_length / 2))
                feat_h = feat_w * 2
            current_attn_map = torch.sqrt((attn_output ** 2).mean(dim=2, keepdim=True) + 1e-12)
            current_attn_map = current_attn_map.view(feat_h, feat_w).cpu().numpy()
        elif len(attn_output.shape) == 4:
            current_attn_map = torch.sqrt((attn_output ** 2).mean(dim=1, keepdim=True) + 1e-12)
            current_attn_map = current_attn_map.squeeze().cpu().numpy()
        else:
            continue

        current_attn_map = suppress_low_res_attention_sink(current_attn_map)
        current_attn_map = cv2.resize(current_attn_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        norm_map = normalize_scoremap(current_attn_map)
        combined_attn_map += norm_map

    combined_attn_map /= num_layers
    return combined_attn_map

def compute_activation_map_simple(feature_map, target_size=(256, 128)):
    """
    用于提取 HCEN 的 2D 空间特征图。
    Args:
        feature_map: Tensor (B, C, H, W)
    """
    x = torch.sqrt((feature_map ** 2).mean(dim=1, keepdim=True) + 1e-12)
    x = x.detach().cpu().numpy()
    am = x[0, 0]
    am = suppress_low_res_attention_sink(am)
    
    h_orig, w_orig = target_size
    am = cv2.resize(am, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    am = normalize_scoremap(am)
    return am

def apply_colormap_on_image(org_img_np, activation_map, alpha=0.48, bg_darken=0.5, peak_clip=0.82):
    """Resize, smooth and overlay the heatmap with high contrast."""
    height, width, _ = org_img_np.shape

    # 1. 将激活特征图 resize 到原图分辨率
    am = cv2.resize(activation_map, (width, height), interpolation=cv2.INTER_CUBIC)

    # 低分辨率特征图上采样后容易块状，动态平滑更稳健
    blur_kernel = int(min(height, width) * 0.09)
    blur_kernel = min(max(blur_kernel, 9), 21)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    am = cv2.GaussianBlur(am, (blur_kernel, blur_kernel), sigmaX=0)

    # 2. 确保合法色彩区间
    am = np.clip(am, 0, 1)

    # 2.1 直接压缩最高响应，避免大面积深红饱和
    # 不再重新拉伸，保证峰值颜色被真实压低
    am = np.minimum(am, peak_clip)

    am_u8 = np.uint8(255 * am)

    # 3. 使用 TURBO 色谱上色，过渡更平滑，视觉上更接近之前效果
    # 注意: cv2.applyColorMap 返回的是 BGR 格式
    am_colored = cv2.applyColorMap(am_u8, cv2.COLORMAP_TURBO)
    am_colored = cv2.cvtColor(am_colored, cv2.COLOR_BGR2RGB)

    # 4. 背景压暗后再进行全局透明度混合，由于没有了左上角黑洞的强权，人体会成为唯一亮点
    base = org_img_np.astype(np.float32) * bg_darken
    overlapped = base * (1.0 - alpha) + am_colored.astype(np.float32) * alpha
    overlapped = np.clip(overlapped, 0, 255).astype(np.uint8)

    return overlapped

def save_per_image_heatmaps(img_path, save_path, backbone_over, hcen_over):
    """Save Backbone/HCEN PNG files under Visualize/<image_name>/."""
    visualize_dir = os.path.dirname(os.path.abspath(__file__))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join(visualize_dir, img_name)
    os.makedirs(output_dir, exist_ok=True)

    backbone_file = os.path.join(output_dir, "backbone.png")
    hcen_file = os.path.join(output_dir, "hcen.png")

    Image.fromarray(backbone_over).save(backbone_file)
    Image.fromarray(hcen_over).save(hcen_file)

    print(f"  Saved: {backbone_file}")
    print(f"  Saved: {hcen_file}")

def process_image(img_path, target_size=(256, 128)):
    """
    加载图像，并应用预处理转换为模型输入张量
    """
    # 提取 ImageNet 均值和方差 (项目中默认使用)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # 使用 PIL 打开并转为 RGB
    img_pil = Image.open(img_path).convert('RGB')
    
    # 模型推断所需的 transform
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    input_tensor = transform(img_pil).unsqueeze(0) # 扩充 Batch 维度
    
    # 为了可视化，获得 resize 到目标大小的原始 NumPy 图像
    orig_img_resized = img_pil.resize((target_size[1], target_size[0])) # PIL size 是 (W, H)
    orig_img_np = np.array(orig_img_resized)
    
    return input_tensor, orig_img_np

def visualize_heatmaps(model, img_paths, save_path, target_size=(256, 128)):
    """
    对多张图片提取 Backbone / HCEN 两类特征图并分别保存 PNG
    """
    model.eval()
        
    with torch.no_grad():
        for col_idx, img_path in enumerate(img_paths):
            print(f"Processing image: {img_path}")
            
            # 1. 挂载 Hook 用于完美劫持 Backbone 特征
            attention_maps = []
            hooks = register_attention_hooks(model, attention_maps)
            
            # 2. 前处理
            input_tensor, orig_img_np = process_image(img_path, target_size)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # 3. 模型推理 (开启 SFM 拿到 HCEN 字典)
            output = model.base(input_tensor)
            
            # 清理钩子
            for h in hooks:
                h.remove()
            
            if not isinstance(output, dict) or 'backbone_map' not in output:
                raise ValueError("Model didn't return a dict with 'backbone_map'. "
                                 "Make sure USE_SFM=True and the model architecture aligns.")
                
            fused_maps = output['fused_maps']
            
            # HCEN 融合特征选取最后一层的输出 (SFM3)
            # 因为 HCEN 是直接的 (B, C, H, W) 2D 张量，所以保留原处理方法
            hcen_map = fused_maps[-1] if len(fused_maps) > 0 else output['backbone_map']
            
            # 4. 计算大一统的纯 Backbone 特征注意力 (来自 Hook)
            if len(attention_maps) == 0:
                print("Warning: Hook failed! Using simple RMS on final map instead.")
                backbone_am = compute_activation_map_simple(output['backbone_map'], target_size)
            else:
                backbone_am = compute_activation_map_from_hooks(attention_maps, target_size)
                
            # 计算 HCEN 特征图注意力
            hcen_am = compute_activation_map_simple(hcen_map, target_size)
            
            # 5. 颜色叠加与生成
            backbone_over = apply_colormap_on_image(orig_img_np, backbone_am)
            hcen_over = apply_colormap_on_image(orig_img_np, hcen_am)

            # 5.1 每张图单独保存 2 张热力图
            save_per_image_heatmaps(
                img_path=img_path,
                save_path=save_path,
                backbone_over=backbone_over,
                hcen_over=hcen_over,
            )

    visualize_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n=> 已完成，Backbone/HCEN PNG 位于: {visualize_dir}/<图片名>/")

def main():
    parser = argparse.ArgumentParser(description="MambaVision+HCEN Heatmap Generator")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--img_paths", default="", help="comma-separated paths to image files", type=str)
    parser.add_argument("--weights", default="weights_placeholder.pth", help="path to trained model weights", type=str)
    parser.add_argument("--output", default="heatmap_results.png", help="path to save output grid image", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    cfg.TEST.WEIGHT = args.weights
    
    # 强制开启 SFM 以获得层级融合字典输出
    cfg.MODEL.MAMBAVISION.USE_SFM = True

    print("Building model...")
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
        print("Please replace '--weights' with your exact trained model .pth path on the server.")

    if torch.cuda.is_available():
        model = model.cuda()
        
    img_list = args.img_paths.split(',') if args.img_paths else []
    
    # 测试模式提供占位图
    if not img_list or not os.path.exists(img_list[0]):
        print("\n未提供有效图片路径，将创建随机张量测试功能完整...")
        dummy_path = "dummy_test.jpg"
        dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        Image.fromarray(dummy_img).save(dummy_path)
        img_list = [dummy_path]

    visualize_heatmaps(model, img_list, args.output, target_size=cfg.INPUT.SIZE_TEST)

if __name__ == "__main__":
    main()
