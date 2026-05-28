"""
测试模型的 Params / FLOPs / FPS

使用方法:
    # 基本用法 (使用配置文件)
    python scripts/test_model_metrics.py --config_file configs/Market/mambavision_tiny_transreid.yml

    # 指定输入尺寸 (覆盖配置文件中的尺寸)
    python scripts/test_model_metrics.py --config_file configs/Market/mambavision_tiny_transreid.yml --img_size 256 128

    # 指定 GPU
    python scripts/test_model_metrics.py --config_file configs/Market/mambavision_tiny_transreid.yml MODEL.DEVICE_ID "('0')"

    # FPS 测试参数
    python scripts/test_model_metrics.py --config_file configs/Market/mambavision_tiny_transreid.yml --warmup 50 --repeat 300 --batch_size 1

依赖:
    pip install thop   # 用于计算 FLOPs (推荐)
    # 或
    pip install fvcore  # 备选方案
"""

import sys
import os
import argparse
import time

import torch
import torch.nn as nn
import numpy as np

# 将项目根目录加入 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import cfg
from model import make_model


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_flops_thop(model, input_tensor):
    """使用 thop 计算 FLOPs"""
    try:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, params, flops_str, params_str
    except ImportError:
        print("[Warning] thop 未安装, 请运行: pip install thop")
        return None, None, None, None


def count_flops_fvcore(model, input_tensor):
    """使用 fvcore 计算 FLOPs"""
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flop_analyzer = FlopCountAnalysis(model, input_tensor)
        flop_analyzer.unsupported_ops_warnings(False)
        flop_analyzer.uncalled_modules_warnings(False)
        flops = flop_analyzer.total()
        return flops
    except ImportError:
        print("[Warning] fvcore 未安装, 请运行: pip install fvcore")
        return None


def measure_fps(model, input_size, device, batch_size=1, warmup=50, repeat=300):
    """
    测量模型的 FPS (Frames Per Second)

    Args:
        model: 模型
        input_size: 输入尺寸 (C, H, W)
        device: 设备
        batch_size: 批量大小
        warmup: 预热轮数
        repeat: 重复测量轮数

    Returns:
        fps: 每秒处理帧数
        latency_ms: 单帧延迟 (ms)
    """
    model.eval()
    dummy_input = torch.randn(batch_size, *input_size, device=device)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # 同步 GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 测量
    print(f"  Measuring ({repeat} iterations, batch_size={batch_size})...")
    timings = []
    with torch.no_grad():
        for _ in range(repeat):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

    timings = np.array(timings)
    avg_time = timings.mean()
    std_time = timings.std()
    latency_ms = avg_time * 1000  # 转换为毫秒
    fps = batch_size / avg_time

    return fps, latency_ms, std_time * 1000


def main():
    parser = argparse.ArgumentParser(description="测试模型 Params / FLOPs / FPS")
    parser.add_argument("--config_file", default="", type=str, help="配置文件路径")
    parser.add_argument("--img_size", nargs=2, type=int, default=None,
                        help="输入图像尺寸 H W (覆盖配置文件), 例如: --img_size 256 128")
    parser.add_argument("--batch_size", type=int, default=1, help="FPS 测试的 batch size")
    parser.add_argument("--warmup", type=int, default=50, help="FPS 预热轮数")
    parser.add_argument("--repeat", type=int, default=300, help="FPS 测量轮数")
    parser.add_argument("--no_fps", action="store_true", help="跳过 FPS 测试")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 测试")
    parser.add_argument("opts", help="其他配置选项", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ==================== 加载配置 ====================
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # 关闭不必要的功能
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = 'self'  # 不加载预训练权重
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.freeze()

    # 确定输入尺寸
    if args.img_size:
        img_h, img_w = args.img_size
    else:
        img_h, img_w = cfg.INPUT.SIZE_TRAIN

    # 确定设备
    device_id = cfg.MODEL.DEVICE_ID
    if isinstance(device_id, str):
        device_id = device_id.strip("()'\" ")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  模型性能指标测试 (Params / FLOPs / FPS)")
    print("=" * 70)
    print(f"  Config:          {args.config_file}")
    print(f"  Backbone:        {cfg.MODEL.TRANSFORMER_TYPE}")
    print(f"  Input Size:      {img_h} x {img_w}")
    print(f"  Device:          {device}")
    print(f"  FP16:            {args.fp16}")
    if hasattr(cfg.MODEL, 'MAMBAVISION'):
        print(f"  USE_SFM:         {cfg.MODEL.MAMBAVISION.USE_SFM}")
        print(f"  SFM_DEPTHS:      {list(cfg.MODEL.MAMBAVISION.SFM_DEPTHS)}")
        print(f"  GLOBAL_STAGES:   {list(cfg.MODEL.MAMBAVISION.GLOBAL_STAGES)}")
        print(f"  SASF_STAGES:     {list(cfg.MODEL.MAMBAVISION.SASF_STAGES)}")
    print("=" * 70)

    # ==================== 构建模型 ====================
    print("\n[1/3] 构建模型...")
    # 使用虚拟的 num_classes, camera_num, view_num
    num_classes = 751  # Market-1501 的类别数
    camera_num = 6
    view_num = 0
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model = model.to(device)
    model.eval()

    if args.fp16 and device.type == 'cuda':
        model = model.half()

    # ==================== 统计参数量 ====================
    print("\n[2/3] 统计参数量...")
    total_params, trainable_params = count_parameters(model)
    print(f"  Total Params:      {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Trainable Params:  {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # ==================== 计算 FLOPs ====================
    print("\n[3/3] 计算 FLOPs...")
    dummy_input = torch.randn(1, 3, img_h, img_w, device=device)
    if args.fp16 and device.type == 'cuda':
        dummy_input = dummy_input.half()

    flops_value = None

    # 尝试 thop
    flops, params, flops_str, params_str = count_flops_thop(model, dummy_input)
    if flops is not None:
        flops_value = flops
        print(f"  [thop] FLOPs:    {flops_str} ({flops / 1e9:.2f}G)")
        print(f"  [thop] Params:   {params_str}")

    # 尝试 fvcore
    fvcore_flops = count_flops_fvcore(model, dummy_input)
    if fvcore_flops is not None:
        flops_value = flops_value or fvcore_flops
        print(f"  [fvcore] FLOPs:  {fvcore_flops / 1e9:.2f}G")

    if flops_value is None:
        print("  [Error] 无法计算 FLOPs，请安装 thop 或 fvcore:")
        print("          pip install thop")
        print("          pip install fvcore")

    # ==================== 测量 FPS ====================
    if not args.no_fps:
        print(f"\n[4/4] 测量 FPS (batch_size={args.batch_size})...")
        fps, latency_ms, std_ms = measure_fps(
            model,
            input_size=(3, img_h, img_w),
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(f"  FPS:             {fps:.1f}")
        print(f"  Latency:         {latency_ms:.2f} ± {std_ms:.2f} ms")
        print(f"  Throughput:      {fps:.1f} images/sec (batch_size={args.batch_size})")

        # 也测试大 batch 的吞吐量
        if args.batch_size == 1 and device.type == 'cuda':
            for bs in [16, 32, 64]:
                try:
                    fps_bs, lat_bs, std_bs = measure_fps(
                        model,
                        input_size=(3, img_h, img_w),
                        device=device,
                        batch_size=bs,
                        warmup=10,
                        repeat=50,
                    )
                    print(f"  Throughput (bs={bs:>2d}): {fps_bs:.1f} images/sec, "
                          f"latency={lat_bs:.2f} ± {std_bs:.2f} ms")
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"  Throughput (bs={bs:>2d}): OOM, 跳过")
                        torch.cuda.empty_cache()
                        break
                    raise

    # ==================== 汇总 ====================
    print("\n" + "=" * 70)
    print("  汇总结果")
    print("=" * 70)
    print(f"  Model:           {cfg.MODEL.TRANSFORMER_TYPE}")
    print(f"  Input:           {img_h} x {img_w}")
    print(f"  Params:          {total_params / 1e6:.2f}M")
    if flops_value is not None:
        print(f"  FLOPs:           {flops_value / 1e9:.2f}G")
    if not args.no_fps:
        print(f"  FPS (bs=1):      {fps:.1f}")
        print(f"  Latency (bs=1):  {latency_ms:.2f}ms")
    print("=" * 70)


if __name__ == '__main__':
    main()
