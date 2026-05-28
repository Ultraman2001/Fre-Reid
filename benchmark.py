"""
MambaVision-ReID 网络性能基准测试
计算: 参数量 (Params) / FLOPs / 吞吐量 (Throughput)

SSM (selective_scan) FLOPs 计算参考:
  https://github.com/NVlabs/MambaVision/issues/21
  公式: FLOPs_ssm = 9 * L * D * N  (D = d_inner//2, N = d_state)
  额外: FLOPs_D   = L * D           (D parameter 贡献)

使用方法:
  python benchmark.py                          # 默认 Tiny + SFM [1,2,3]
  python benchmark.py --variant small          # Small 变体
  python benchmark.py --no-sfm                 # 不使用 SFM
  python benchmark.py --sfm-depths 1 1 1       # 自定义 SFM depths
  python benchmark.py --sasf-stages 2 3        # 自定义 SASF stages
  python benchmark.py --throughput-only        # 只测吞吐量
"""

import torch
import torch.nn as nn
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from model.backbones.mambavision.mamba_vision_reid import (
        MambaVisionBackbone, MambaVisionMixer,
        mambavision_tiny_reid, mambavision_small_reid, mambavision_base_reid,
    )
except ImportError:
    # Fallback for flat-file layouts.
    from mamba_vision_reid import (
        MambaVisionBackbone, MambaVisionMixer,
        mambavision_tiny_reid, mambavision_small_reid, mambavision_base_reid,
    )


# ========================== 1. 参数量统计 ==========================

def count_parameters(model):
    """统计总参数量和可训练参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_backbone_checkpoint(model, ckpt_path):
    """加载训练得到的checkpoint到backbone，兼容 base./module.base. 前缀。"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model' in ckpt):
        state_dict = ckpt.get('state_dict', ckpt.get('model'))
    else:
        state_dict = ckpt

    # 兼容常见前缀：module.base. / base. / module.
    remapped = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith('module.base.'):
            nk = nk[len('module.base.'):]
        elif nk.startswith('base.'):
            nk = nk[len('base.'):]
        elif nk.startswith('module.'):
            nk = nk[len('module.'):]
        remapped[nk] = v

    model_dict = model.state_dict()
    matched = {k: v for k, v in remapped.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)
    return len(matched), len(model_dict)


def print_param_breakdown(model):
    """按组件分解参数量"""
    components = {}
    for name, param in model.named_parameters():
        # 提取顶层组件名
        top = name.split('.')[0]
        if top not in components:
            components[top] = 0
        components[top] += param.numel()

    print(f"\n  --- 各组件参数量 ---")
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        print(f"  {comp:25s}: {count:>12,} ({count/1e6:.3f}M)")

    # SFM 汇总
    sfm_total = sum(v for k, v in components.items() if k.startswith('sfm_'))
    backbone_total = sum(v for k, v in components.items() if not k.startswith('sfm_'))
    print(f"  {'--- Backbone 小计':25s}: {backbone_total:>12,} ({backbone_total/1e6:.3f}M)")
    print(f"  {'--- SFM 小计':25s}: {sfm_total:>12,} ({sfm_total/1e6:.3f}M)")


# ========================== 2. FLOPs 计算 ==========================

def get_selective_scan_flops(d_inner_half, d_state, seq_len):
    """
    单个 MambaVisionMixer 的 selective_scan FLOPs.

    参考: https://github.com/NVlabs/MambaVision/issues/21
    公式:
      - SSM 核心: 9 * L * D * N
      - D 参数:   L * D
    其中 D = d_inner//2, N = d_state, L = sequence length
    """
    ssm_flops = 9 * seq_len * d_inner_half * d_state
    d_flops = seq_len * d_inner_half
    return ssm_flops + d_flops


def count_standard_flops(model, img_size=(256, 128)):
    """
    使用 fvcore 计算标准层 (Conv, Linear, BN, Attention 等) 的 FLOPs.
    selective_scan_fn 是自定义 CUDA op, fvcore 无法识别, 需要手动补算.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("  [WARN] fvcore 未安装, 尝试使用 thop...")
        try:
            from thop import profile
            x = torch.randn(1, 3, img_size[0], img_size[1])
            # 需要一个 wrapper 来处理 dict 输出
            class Wrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    out = self.m(x)
                    if isinstance(out, dict):
                        return out['backbone_map']
                    return out
            flops, _ = profile(Wrapper(model), inputs=(x,), verbose=False)
            return flops
        except ImportError:
            print("  [WARN] thop 也未安装, 标准层 FLOPs 设为 0")
            return 0

    x = torch.randn(1, 3, img_size[0], img_size[1])

    # fvcore 需要模型返回 Tensor, 包装一下处理 dict 输出
    class FlopWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            if isinstance(out, dict):
                # 汇总所有输出 tensor 的 dummy sum 触发所有计算路径
                result = out['backbone_map']
                for fm in out.get('fused_maps', []):
                    result = result + fm.mean() * 0  # 不影响数值, 但触发计算图
                return result
            return out

    wrapper = FlopWrapper(model)

    def _run_fvcore(_wrapper, _x):
        analysis = FlopCountAnalysis(_wrapper, _x)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return analysis.total()

    try:
        return _run_fvcore(wrapper, x)
    except RuntimeError as e:
        # selective_scan CUDA op fails during CPU tracing; retry on CUDA if available.
        if "Expected u.is_cuda() to be true" in str(e) and torch.cuda.is_available():
            wrapper = wrapper.cuda().eval()
            x = x.cuda()
            return _run_fvcore(wrapper, x)
        raise


def count_all_ssm_flops(model, img_size=(256, 128)):
    """
    遍历模型中所有 MambaVisionMixer, 手动计算 selective_scan FLOPs.
    这些 FLOPs 被 fvcore/thop 等工具遗漏.
    """
    dim = model.dim
    H, W = img_size
    total_ssm_flops = 0
    details = []

    # =============== Backbone stages ===============
    feat_h, feat_w = H // 4, W // 4  # PatchEmbed 4x 下采样

    for stage_idx, level in enumerate(model.levels):
        if level.conv:
            # Conv stages: 无 Mamba blocks
            if level.downsample is not None:
                feat_h = feat_h // 2
                feat_w = feat_w // 2
            continue

        # 确定当前 stage 维度
        if stage_idx == 3:
            stage_dim = model.proj_dim  # 512
        else:
            stage_dim = dim * (2 ** stage_idx)  # Stage2=320, Stage3 before proj

        # 遍历该 stage 中的 blocks
        mamba_count = 0
        stage_ssm_flops = 0
        for blk in level.blocks:
            if not isinstance(blk.mixer, MambaVisionMixer):
                continue

            mixer = blk.mixer
            d_inner_half = mixer.d_inner // 2
            d_state = mixer.d_state

            if level.use_global:
                # 全局模式: seq_len = H * W
                seq_len = feat_h * feat_w
                n_windows = 1
            else:
                # 窗口模式
                ws = level.window_size
                pad_r = (ws - feat_w % ws) % ws
                pad_b = (ws - feat_h % ws) % ws
                Hp = feat_h + pad_b
                Wp = feat_w + pad_r
                n_windows = (Hp // ws) * (Wp // ws)
                seq_len = ws * ws

            blk_ssm = get_selective_scan_flops(d_inner_half, d_state, seq_len) * n_windows
            stage_ssm_flops += blk_ssm
            mamba_count += 1

        if mamba_count > 0:
            total_ssm_flops += stage_ssm_flops
            details.append(
                f"  Stage {stage_idx}: {mamba_count} Mamba blocks, "
                f"dim={stage_dim}, d_state=8, "
                f"seq_len={seq_len}, windows={n_windows}, "
                f"SSM FLOPs={stage_ssm_flops/1e6:.3f}M"
            )

        if level.downsample is not None:
            feat_h = feat_h // 2
            feat_w = feat_w // 2

    # =============== SFM modules ===============
    if model.use_sfm:
        target_h, target_w = 16, 8
        # Current code path (Block/MambaVisionMixer) scans full sequence in SFM.
        # Only WaveSplit-style blocks use LL quarter sequence.
        seq_len_sfm_full = target_h * target_w  # 128 for 16x8
        seq_len_sfm_ll = (target_h // 2) * (target_w // 2)  # 32 for 16x8

        sfm_names = ['sfm_s12', 'sfm_s23', 'sfm_s34']
        for sfm_idx, sfm_name in enumerate(sfm_names):
            if sfm_idx >= len(model.sfm_depths) or model.sfm_depths[sfm_idx] <= 0:
                continue
            if not hasattr(model, sfm_name):
                continue

            sfm_module = getattr(model, sfm_name)
            sfm_ssm_flops = 0
            mamba_count = 0

            for blk in sfm_module.fusion_mamba:
                mixer = None
                seq_len_this_blk = seq_len_sfm_full
                if hasattr(blk, 'mamba') and isinstance(blk.mamba, MambaVisionMixer):
                    mixer = blk.mamba
                    # WaveSplit-style block: mixer processes LL subband tokens.
                    seq_len_this_blk = seq_len_sfm_ll
                elif hasattr(blk, 'mixer') and isinstance(blk.mixer, MambaVisionMixer):
                    mixer = blk.mixer

                if mixer is None:
                    continue

                d_inner_half = mixer.d_inner // 2
                d_state = mixer.d_state
                # SFM: 全局扫描 (无窗口), 1 个序列
                blk_ssm = get_selective_scan_flops(d_inner_half, d_state, seq_len_this_blk)
                sfm_ssm_flops += blk_ssm
                mamba_count += 1

            if mamba_count > 0:
                total_ssm_flops += sfm_ssm_flops
                details.append(
                    f"  {sfm_name}: {mamba_count} Mamba blocks, "
                    f"dim={sfm_module.out_dim*2}, d_state=8, "
                    f"seq_len={seq_len_sfm_full}, "
                    f"SSM FLOPs={sfm_ssm_flops/1e6:.3f}M"
                )

    return total_ssm_flops, details


# ========================== 3. 吞吐量测试 ==========================

def measure_throughput(model, img_size=(256, 128), batch_size=64,
                       warmup=50, iterations=200, device='cuda'):
    """测量吞吐量 (images/second) 和延迟 (ms)"""
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = batch_size * iterations / elapsed  # img/s
    latency_per_img = elapsed / (iterations * batch_size) * 1000  # ms/img
    latency_per_batch = elapsed / iterations * 1000  # ms/batch
    return throughput, latency_per_img, latency_per_batch


# ========================== Main ==========================

def main():
    parser = argparse.ArgumentParser(description='MambaVision-ReID Benchmark')
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='模型变体 (default: tiny)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 128],
                        help='输入图片尺寸 H W (default: 256 128)')
    parser.add_argument('--no-sfm', action='store_true',
                        help='禁用 SFM')
    parser.add_argument('--sfm-depths', type=int, nargs='+', default=[1, 2, 3],
                        help='SFM 各层深度 (default: 1 2 3)')
    parser.add_argument('--sasf-stages', type=int, nargs='*', default=[2, 3],
                        help='启用 SASF 的 stage 索引 (default: 2 3)')
    parser.add_argument('--global-stages', type=int, nargs='*', default=[],
                        help='使用全局注意力的 stage 索引 (default: none)')
    parser.add_argument('--throughput-only', action='store_true',
                        help='只测吞吐量')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 32, 64, 128],
                        help='吞吐量测试的 batch sizes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (default: cuda)')
    parser.add_argument('--weight', type=str, default='',
                        help='训练好的模型权重路径 (.pth)')
    args = parser.parse_args()

    img_size = tuple(args.img_size)
    use_sfm = not args.no_sfm

    # ========== 构建模型 ==========
    build_fn = {
        'tiny': mambavision_tiny_reid,
        'small': mambavision_small_reid,
        'base': mambavision_base_reid,
    }[args.variant]

    model = build_fn(
        img_size=img_size,
        use_sfm=use_sfm,
        sfm_depths=args.sfm_depths,
        sfm_drop_path=0.0,
        global_stages=args.global_stages,
        sasf_stages=args.sasf_stages,
    )

    if args.weight:
        if os.path.isfile(args.weight):
            loaded, total = load_backbone_checkpoint(model, args.weight)
            print(f"[INFO] Loaded checkpoint: {args.weight}")
            print(f"[INFO] Matched params: {loaded}/{total}")
        else:
            print(f"[WARN] Checkpoint not found, skip loading: {args.weight}")
    model.eval()

    print("\n" + "=" * 70)
    print(f"  MambaVision-ReID Benchmark  |  variant={args.variant}")
    print(f"  img_size={img_size}, SFM={'ON' if use_sfm else 'OFF'}, "
          f"sfm_depths={args.sfm_depths}")
    print(f"  sasf_stages={args.sasf_stages}, global_stages={args.global_stages}")
    print("=" * 70)

    if args.throughput_only:
        # 仅测吞吐量
        device = args.device if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print("\n⚠️  CUDA 不可用, 吞吐量测试跳过")
            return
        model = model.to(device)
        print(f"\n📊 吞吐量 (Throughput) @ {device}")
        for bs in args.batch_sizes:
            try:
                tp, lat_img, lat_batch = measure_throughput(
                    model, img_size, batch_size=bs,
                    warmup=30, iterations=100, device=device
                )
                print(f"  batch_size={bs:>3d}: {tp:>8.1f} img/s | "
                      f"{lat_img:.2f} ms/img | {lat_batch:.1f} ms/batch")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  batch_size={bs:>3d}: OOM (显存不足)")
                    torch.cuda.empty_cache()
                else:
                    raise
        return

    # ========== 1. 参数量 ==========
    total_params, trainable_params = count_parameters(model)
    print(f"\n{'─'*40}")
    print(f"📊 参数量 (Parameters)")
    print(f"{'─'*40}")
    print(f"  总参数量:       {total_params:>12,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数量:   {trainable_params:>12,} ({trainable_params/1e6:.2f}M)")
    print_param_breakdown(model)

    # ========== 2. FLOPs ==========
    print(f"\n{'─'*40}")
    print(f"📊 FLOPs (input: 1×3×{img_size[0]}×{img_size[1]})")
    print(f"{'─'*40}")

    # 2a. 标准层 FLOPs (fvcore / thop)
    standard_flops = count_standard_flops(model, img_size)

    # 2b. SSM FLOPs (手动计算, 工具无法识别 selective_scan_fn)
    ssm_flops, ssm_details = count_all_ssm_flops(model, img_size)

    total_flops = standard_flops + ssm_flops

    print(f"\n  标准层 FLOPs (fvcore):   {standard_flops/1e9:.3f} GFLOPs")
    print(f"  SSM FLOPs (手动补算):    {ssm_flops/1e9:.3f} GFLOPs")
    print(f"  ──────────────────────────────────")
    print(f"  总 FLOPs:                {total_flops/1e9:.3f} GFLOPs")

    if ssm_details:
        print(f"\n  --- SSM FLOPs 明细 ---")
        for line in ssm_details:
            print(line)

    # ========== 3. 吞吐量 ==========
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n{'─'*40}")
    print(f"📊 吞吐量 (Throughput) @ {device}")
    print(f"{'─'*40}")

    if device != 'cuda':
        print("  ⚠️  CUDA 不可用, 吞吐量测试需在 GPU 服务器上运行")
    else:
        model = model.to(device)
        for bs in args.batch_sizes:
            try:
                tp, lat_img, lat_batch = measure_throughput(
                    model, img_size, batch_size=bs,
                    warmup=30, iterations=100, device=device
                )
                print(f"  batch_size={bs:>3d}: {tp:>8.1f} img/s | "
                      f"{lat_img:.2f} ms/img | {lat_batch:.1f} ms/batch")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  batch_size={bs:>3d}: OOM (显存不足)")
                    torch.cuda.empty_cache()
                else:
                    raise

    # ========== 汇总 ==========
    print(f"\n{'='*70}")
    print(f"  ✅ 汇总")
    print(f"{'='*70}")
    print(f"  模型变体:    MambaVision-{args.variant.capitalize()}-ReID")
    print(f"  输入尺寸:    {img_size}")
    print(f"  SFM 配置:    {'OFF' if not use_sfm else str(args.sfm_depths)}")
    print(f"  参数量:      {total_params/1e6:.2f}M")
    print(f"  FLOPs:       {total_flops/1e9:.3f} GFLOPs")
    if device == 'cuda':
        # 使用默认 batch_size=64 报告 Throughput(image/s)
        try:
            tp64, _, _ = measure_throughput(
                model, img_size, batch_size=64,
                warmup=30, iterations=100, device=device
            )
            print(f"  Throughput(image/s) (bs=64):  {tp64:.1f}")
        except:
            pass
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
