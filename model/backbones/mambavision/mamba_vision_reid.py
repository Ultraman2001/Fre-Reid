"""
MambaVision backbone for ReID tasks
Adapted from NVIDIA MambaVision for TransReID framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath
from timm.models.vision_transformer import Mlp
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

try:
    from Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None

class StateFusion(nn.Module):
    """
    Ultra-simplified SASF: Pure multi-scale depthwise conv (DWConv).
    Uses GroupNorm for stability in small-batch ReID.
    Returns a spatial residual feature (to be injected by sasf_scale outside).
    """
    def __init__(self, dim):
        super().__init__()
        # Branch 1: 3x3 local (Capture fine textures)
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # Branch 2: 3x3 with dilation=2 (5x5 effective context)
        self.dw5 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
        
        # Stability: GroupNorm is more robust than BatchNorm for small batches
        num_groups = min(32, dim) if dim % 32 == 0 else (dim // 8 if dim % 8 == 0 else 1)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.act = nn.SiLU(inplace=True)
        
        # Initialization
        nn.init.kaiming_normal_(self.dw3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw5.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, h):
        """
        h: (B, C, H, W)
        """
        # Multi-scale 2D structural perception
        spatial = self.dw3(h) + self.dw5(h)
        # Apply normalization and SiLU activation
        spatial = self.act(self.norm(spatial))
        
        return spatial


class NeABottleneck(nn.Module):
    """
    SASF-Style Multi-Scale Non-linear Enhancement (NeA) module.
    
    借鉴 Spatial-Mamba StateFusion 的设计:
    - 深度可分离卷积 (groups=dim)
    - 多尺度空洞卷积 (dilation=1, 2)
    - 可学习的 alpha 加权融合
    - Replicate padding 保持边界连续性
    
    感受野: 3×3 (dilation=1) + 5×5 (dilation=2)
    """
    def __init__(self, inplanes, planes):
        super().__init__()
        # 1x1 降维
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 多尺度深度可分离卷积 (SASF 风格)
        # dilation=1: RF=3×3, dilation=2: RF=5×5
        self.kernel_d1 = nn.Parameter(torch.ones(planes, 1, 3, 3) * 0.1)  # 深度可分离
        self.kernel_d2 = nn.Parameter(torch.ones(planes, 1, 3, 3) * 0.1)  # 深度可分离
        
        # 可学习的融合权重 [α₀, α₁]
        self.alpha = nn.Parameter(torch.ones(2))
        
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 升维
        self.conv3 = nn.Conv2d(planes, inplanes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.act = nn.GELU()
    
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        
        # 多尺度深度可分离卷积 (SASF 风格)
        # Replicate padding 保持边界连续性
        h1 = F.conv2d(
            F.pad(out, (1, 1, 1, 1), mode='replicate'),
            self.kernel_d1,
            padding=0, dilation=1, groups=out.size(1)
        )
        h2 = F.conv2d(
            F.pad(out, (2, 2, 2, 2), mode='replicate'),
            self.kernel_d2,
            padding=0, dilation=2, groups=out.size(1)
        )
        
        # 加权融合
        out = self.alpha[0] * h1 + self.alpha[1] * h2
        out = self.act(self.bn2(out))
        
        out = self.bn3(self.conv3(out))
        return self.act(out + identity)


def _resize_to(x, size):
    """
    抗混叠 resize：下采样用 area，上采样用 bilinear
    """
    H, W = x.shape[-2:]
    tH, tW = size
    if H > tH or W > tW:
        # 下采样：使用 area 模式抗混叠（或 adaptive_avg_pool2d）
        return F.interpolate(x, size=size, mode='area')
    elif H < tH or W < tW:
        # 上采样：使用 bilinear 插值
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return x


class LinearAlign(nn.Module):
    """
    线性对齐模块 (用于 SFM_34 深层融合)
    
    结构: Resize + Conv1x1 + BN (无激活、无DW)
    
    设计理由:
    - 深层特征已包含高度抽象的行人语义
    - 过多处理会破坏特征的"流形结构"
    - 只做维度投影，最大程度保留原始语义
    """
    def __init__(self, in_dim, out_dim, target_size=(16, 8)):
        super().__init__()
        self.target_size = target_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
    
    def forward(self, x):
        x = _resize_to(x, self.target_size)
        return self.proj(x)

class DenoiseAlign(nn.Module):
    """
    去噪对齐模块 (用于 SFM_12/SFM_23 浅层融合)
    
    结构: Resize + Conv1x1 + BN + 残差 DWConv3x3
    
    设计理由:
    - 浅层特征噪声多，下采样跨度大
    - 先抗混叠 (area resize)，再做轻去噪 (residual DWConv)
    - DWConv 用残差式避免"强改分布"
    
    Args:
        in_dim: 输入通道数
        out_dim: 输出通道数
        target_size: 目标空间尺寸
        use_act: 是否在 DW 分支使用激活 (SFM_12=True, SFM_23=False)
    """
    def __init__(self, in_dim, out_dim, target_size=(16, 8), use_act=True):
        super().__init__()
        self.target_size = target_size
        
        # 通道投影: Conv1x1 + BN
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        
        # 残差去噪分支: DWConv3x3 + BN (+ 可选激活)
        if use_act:
            self.dw = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.dw = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim, bias=False),
                nn.BatchNorm2d(out_dim),
            )
    
    def forward(self, x):
        x = _resize_to(x, self.target_size)
        x = self.proj(x)
        x = x + self.dw(x)  # 残差式：轻去噪，不强改分布
        return x


class GateNet(nn.Module):
    """
    统一通道门控模块 (Unified Channel Gating Module)
    
    基于 SFM_UNIFIED_GATENET_PLAN.md 规范实现：
    - 输入: 对齐后的两路特征 x_l, x_h (B, C, H, W)
    - 输出: 通道门控 g ∈ [0, 1]^(B, C, 1, 1)
    
    结构: Concat -> GAP -> 2-layer Bottleneck MLP -> Sigmoid
    
    Args:
        dim: 通道数 (对齐后的统一通道数)
        reduction: 瓶颈降维比例 (default: 8)
        temperature: Sigmoid 温度参数 (default: 2.0, 高温度使输出更平滑)
    """
    def __init__(self, dim, reduction=8, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        hidden_dim = max(dim * 2 // reduction, 16)  # 保证最小隐层维度
        
        # 两层瓶颈 MLP (等价于 1x1 Conv)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim, bias=False),
            nn.ReLU(inplace=True),  # 原文建议 SiLU/GELU，这里用 ReLU 保持简洁
            nn.Linear(hidden_dim, dim, bias=True),
        )
        
        # 初始化: 让初始门控接近 0.5 (平衡状态)
        nn.init.zeros_(self.gate_mlp[-1].bias)
    
    def forward(self, x_low, x_high):
        """
        Args:
            x_low: 对齐后的低层特征 (B, C, H, W)
            x_high: 对齐后的高层特征 (B, C, H, W)
        Returns:
            gate: 通道门控 (B, C, 1, 1)
        """
        B, C, H, W = x_low.shape
        
        # 1. 拼接两路特征
        concat = torch.cat([x_low, x_high], dim=1)  # (B, 2C, H, W)
        
        # 2. 全局平均池化 (GAP)
        z = concat.mean(dim=(2, 3))  # (B, 2C)
        
        # 3. 两层瓶颈 MLP (FP32 保护，防止 FP16 溢出)
        orig_dtype = z.dtype
        with torch.cuda.amp.autocast(enabled=False):
            u = self.gate_mlp(z.float())  # (B, C)
        
        # 4. 温度缩放 Sigmoid (高温度使输出更平滑，减少极端值)
        gate = torch.sigmoid(u / self.temperature)  # (B, C)
        
        # 5. Soft clamp: [0.05, 0.95] 防止完全的 0 或 1 导致梯度消失
        gate = 0.05 + 0.9 * gate
        
        # 6. Reshape 为通道门控并恢复原始 dtype
        gate = gate.view(B, C, 1, 1).to(orig_dtype)  # (B, C, 1, 1)
        
        return gate


class SimpleFusionMambaBlock(nn.Module):
    """
    Simple Fusion Mamba Block (SFM) with Unified GateNet Fusion.
    
    基于 SFM_UNIFIED_GATENET_PLAN.md 规范实现 (附录 A: Gated Concat 版本)：
    - 统一门控: GateNet (GAP + 2-layer bottleneck) 
    - Gated Concat: 保留 2C 交互空间，门控只调制 high 分支的贡献
    - 差分注入公式: x_in = Concat(x_l, g ⊙ x_h)
    
    Architecture:
        1. Align low/high features (保留分层对齐策略)
        2. Pre-Normalize aligned features for gate computation
        3. Generate channel gate via GateNet
        4. Gated Concat: Concat(low, gate * high) -> 保留 2C 交互
        5. Process through SS2D (MambaVisionMixer) in 2C space
        6. Split and residual injection
        7. Enhance with NeA bottleneck
    
    Args:
        low_dim: Channel dimension of low-level features
        high_dim: Channel dimension of high-level features
        out_dim: Output channel dimension (default: 512)
        target_size: Target spatial size (H, W) for alignment
        depth: Number of Mamba blocks in the fusion module (default: 1)
        align_type: Alignment strategy ('denoise_strong', 'denoise_weak', 'linear')
        gate_reduction: GateNet bottleneck reduction ratio (default: 8)
    """
    def __init__(self, low_dim, high_dim, out_dim=512, 
                 target_size=(16, 8), depth=1, drop_path=0.1, 
                 align_type='linear', gate_reduction=8):
        super().__init__()
        self.target_size = target_size
        self.out_dim = out_dim
        self.depth = depth
        self.align_type = align_type
        
        # ========== 分层对齐模块 ==========
        if align_type == 'denoise_strong':
            # SFM_12: 最浅层，需要强去噪 (DWConv + SiLU)
            self.align_low = DenoiseAlign(low_dim, out_dim, target_size, use_act=True)
            self.align_high = DenoiseAlign(high_dim, out_dim, target_size, use_act=True)
        elif align_type == 'denoise_weak':
            # SFM_23: 中层，弱去噪 (DWConv only, 无激活)
            self.align_low = DenoiseAlign(low_dim, out_dim, target_size, use_act=False)
            self.align_high = DenoiseAlign(high_dim, out_dim, target_size, use_act=False)
        else:  # 'linear'
            # SFM_34: 深层，只做线性投影，保留语义
            self.align_low = LinearAlign(low_dim, out_dim, target_size)
            self.align_high = LinearAlign(high_dim, out_dim, target_size)
        
        # ========== Pre-Normalization (仅用于门控计算，消除量纲差异) ==========
        self.norm_low = nn.LayerNorm(out_dim)
        self.norm_high = nn.LayerNorm(out_dim)
        
        # ========== 统一通道门控 (Unified GateNet) ==========
        self.gate_net = GateNet(out_dim, reduction=gate_reduction)
        
        # ========== SS2D fusion blocks (2C 交互空间) ==========
        self.fusion_mamba = nn.ModuleList([
            Block(
                dim=out_dim * 2,  # 保持 2C 空间以保留 cross-stream interaction
                counter=i,
                transformer_blocks=[],
                num_heads=8,
                mlp_ratio=2.,  # 将膨胀率从 4 降至 2，大幅削减参数和显存负担
                qkv_bias=True,
                qk_scale=False,
                drop=0.,
                attn_drop=0.,
                drop_path=drop_path,
                layer_scale=1e-5,
                use_sasf=False,
            ) for i in range(depth)
        ])
        
        # ========== Split projections (从 2C 回到 C) ==========
        self.split_proj_low = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        self.split_proj_high = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        
        # ========== NeA non-linear enhancement ==========
        self.NeA = NeABottleneck(out_dim, out_dim // 4)

        print(f"[SimpleFusionMambaBlock] low={low_dim}, high={high_dim}, "
              f"out={out_dim}, align={align_type}, depth={depth}, gate_r={gate_reduction}")
    
    def forward(self, feat_low, feat_high):
        """
        Args:
            feat_low: Low-level features (B, low_dim, H1, W1)
            feat_high: High-level features (B, high_dim, H2, W2)
        
        Returns:
            fused_map: Fused feature map (B, out_dim, H, W)
        """
        # 1. 对齐两路特征到相同空间尺寸与通道
        low_aligned = self.align_low(feat_low)    # (B, out_dim, tH, tW)
        high_aligned = self.align_high(feat_high)  # (B, out_dim, tH, tW)
        
        
        # 2. Pre-Normalization (仅用于门控计算)
        B, C, H, W = low_aligned.shape
        low_normed = self.norm_low(low_aligned.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        high_normed = self.norm_high(high_aligned.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 3. GateNet: 生成通道门控
        gate = self.gate_net(low_normed, high_normed)  # (B, C, 1, 1)
        
        # 4. Gated Concat (附录 A 方案，保留 2C 交互空间)
        # x_in = Concat(x_l, g ⊙ x_h)
        gated_high = gate * high_aligned  # (B, C, H, W)
        concat_feat = torch.cat([low_aligned, gated_high], dim=1)  # (B, 2C, H, W)
        
        # 5. SS2D 深度交互 (在 2C 空间进行 cross-stream interaction)
        # 垂直扫描: 先 W 后 H，沿行人身体方向展平
        concat_seq = rearrange(concat_feat, 'b c h w -> b (w h) c')  # vertical scan
        for blk in self.fusion_mamba:
            concat_seq = blk(concat_seq, H=W, W=H)  # 注意: H/W 参数交换
        fused_feat = rearrange(concat_seq, 'b (w h) c -> b c h w', w=W, h=H)  # restore
        
        # 6. Split and residual injection (FP32 保护)
        orig_dtype = fused_feat.dtype
        fused_feat_fp32 = fused_feat.float()
        
        with torch.cuda.amp.autocast(enabled=False):
            trans_low = self.split_proj_low(fused_feat_fp32)
            trans_high = self.split_proj_high(fused_feat_fp32)
            
        trans_low = trans_low.to(orig_dtype)
        trans_high = trans_high.to(orig_dtype)
        
        out_low = low_aligned + trans_low
        out_high = high_aligned + trans_high
        
        # 7. Merge and NeA enhancement
        fused_map = out_low + out_high
        fused_map = self.NeA(fused_map)
        
        return fused_map  # (B, out_dim, H, W)




def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


class Downsample(nn.Module):
    """Down-sampling block"""
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding block"""
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        use_sasf=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.use_sasf = use_sasf
        if use_sasf:
            # SASF operates AFTER concat, so input is full d_inner (SSM + Conv1d branches)
            self.state_fusion = StateFusion(self.d_inner)
            # Initialize with 0 scale to ensure "identity" behavior at start
            self.sasf_scale = nn.Parameter(torch.zeros(1))
            
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            padding=d_conv//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            padding=d_conv//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states, H=None, W=None):
        """
        hidden_states: (B, L, D)
        H, W: spatial dimensions for non-square feature maps
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))
        
        # Slice to original seqlen (conv padding may add 1)
        # Slice to original seqlen and ensure contiguous memory
        x = x[..., :seqlen].contiguous()
        z = z[..., :seqlen].contiguous()
        
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen).contiguous()
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        
        # SSM 扫描
        # SSM 扫描 (强制 FP32 计算，防止长序列数值溢出)
        y = selective_scan_fn(x.float(), dt.float(), A.float(), B.float(), C.float(), self.D.float(), z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True, return_last_state=None).to(x.dtype)
        
        # Concatenation FIRST: merge SSM branch (y) and Conv1d branch (z)
        # y: has global sequential context, but lacks 2D vertical structure
        # z: has local horizontal texture, but lacks 2D vertical structure
        y = torch.cat([y, z], dim=1)  # (B, D_inner, L)
        
        # Apply SASF AFTER concat - unified 2D spatial calibration for both branches
        B_batch, D_inner, L_seq = y.shape
        if self.use_sasf:
            # Determine H, W for reshape
            if H is not None and W is not None and H * W == L_seq:
                H_feat, W_feat = H, W
            else:
                # Fallback: try perfect square
                H_feat = int(math.sqrt(L_seq))
                if H_feat * H_feat == L_seq:
                    W_feat = H_feat
                else:
                    H_feat, W_feat = None, None
            
            if H_feat is not None and W_feat is not None:
                y = rearrange(y, "b d (h w) -> b d h w", h=H_feat, w=W_feat)
                # SASF now sees BOTH SSM and Conv features, can calibrate vertical structure for both
                y_sasf = self.state_fusion(y)
                y = y + self.sasf_scale * y_sasf
                y = rearrange(y, "b d h w -> b d (h w)").contiguous()

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, counter, transformer_blocks,
                 mlp_ratio=4., qkv_bias=False, qk_scale=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, Mlp_block=Mlp, layer_scale=None, use_sasf=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1, use_sasf=use_sasf)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x, H=None, W=None):
        # Pass H, W to mixer if it's MambaVisionMixer (which has use_sasf)
        if hasattr(self.mixer, 'use_sasf'):
            x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x), H=H, W=W))
        else:
            x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """MambaVision layer"""
    def __init__(self, dim, depth, num_heads, window_size, conv=False,
                 downsample=True, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., layer_scale=None,
                 layer_scale_conv=None, transformer_blocks=[], use_global=False, use_sasf=False):
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        self.use_global = use_global
        
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                        for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim, counter=i,
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads, mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               drop=drop, attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale,
                                               use_sasf=use_sasf)
                                        for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape

        if self.transformer_block:
            if self.use_global:
                # 全局注意力：不做窗口划分，直接展平为序列
                x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, H*W, C)
                for blk in self.blocks:
                    x = blk(x, H=H, W=W)  # Pass real H, W for non-square feature maps
                x = x.transpose(1, 2).view(B, C, H, W)  # (B, H*W, C) -> (B, C, H, W)
            else:
                # 窗口注意力：原有逻辑
                pad_r = (self.window_size - W % self.window_size) % self.window_size
                pad_b = (self.window_size - H % self.window_size) % self.window_size
                if pad_r > 0 or pad_b > 0:
                    x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                    _, _, Hp, Wp = x.shape
                else:
                    Hp, Wp = H, W
                x = window_partition(x, self.window_size)

                for blk in self.blocks:
                    # In window mode, each window is window_size × window_size (square)
                    x = blk(x, H=self.window_size, W=self.window_size)
                    
                x = window_reverse(x, self.window_size, Hp, Wp)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :, :H, :W].contiguous()
        else:
            # 卷积层：直接处理
            for blk in self.blocks:
                x = blk(x)

        if self.downsample is None:
            return x
        return self.downsample(x)


class FeatureDWTBranch(nn.Module):
    """Stage-1 feature-level Haar DWT branch with WTConv-style subband mixing."""

    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.subband_dwconv = nn.Conv2d(
            in_dim * 4,
            in_dim * 4,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_dim * 4,
            bias=False,
        )
        self.subband_norm = nn.BatchNorm2d(in_dim * 4)
        self.subband_act = nn.GELU(approximate='tanh')
        self.subband_scale = nn.Parameter(torch.ones(1, in_dim * 4, 1, 1) * 0.1)

        self.ll_adapter = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(approximate='tanh'),
        )
        self.hf_adapter = nn.Sequential(
            nn.Conv2d(in_dim * 3, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(approximate='tanh'),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(approximate='tanh'),
        )

    @staticmethod
    def _haar_dwt2d(x):
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            pad_mode = 'reflect' if h > 1 and w > 1 else 'replicate'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (x00 - x01 + x10 - x11) * 0.25
        hl = (x00 + x01 - x10 - x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        return ll, lh, hl, hh

    def forward(self, x):
        subbands = torch.cat(self._haar_dwt2d(x), dim=1)
        subbands = subbands + self.subband_scale * self.subband_act(
            self.subband_norm(self.subband_dwconv(subbands))
        )
        ll, lh, hl, hh = torch.chunk(subbands, 4, dim=1)
        ll_feat = self.ll_adapter(ll)
        hf_feat = self.hf_adapter(torch.cat([lh, hl, hh], dim=1))
        return self.fuse(torch.cat([ll_feat, hf_feat], dim=1))


class MambaVisionBackbone(nn.Module):
    """MambaVision backbone for ReID with optional Fine-Grained Branch"""
    def __init__(self, img_size=(256, 128), dim=80, in_dim=32, depths=[1, 3, 8, 4],
                 window_size=[8, 8, 16, 8], mlp_ratio=4, num_heads=[2, 4, 8, 16],
                 drop_path_rate=0.2, in_chans=3, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., layer_scale=None, layer_scale_conv=None,
                 global_stages=[], sasf_stages=[],
                 camera=0, view=0, sie_xishu=1.5,
                 use_sfm=False, sfm_num_layers=1, sfm_depths=[1, 1], sfm_drop_path=0.0,
                 use_fd=False):
        """
        Args:
            global_stages: list of stage indices to use global attention instead of window attention.
            sasf_stages: list of stage indices to enable SASF in MambaMixer.
            camera: number of cameras for SIE.
            view: number of views for SIE.
            sie_xishu: scaling factor for SIE embedding.
            use_sfm: whether to enable SimpleFusionMamba module.
            sfm_num_layers: number of SFM layers (1 or 2).
            sfm_depths: list of depths for each SFM module [sfm_1_depth, sfm_2_depth].
            sfm_drop_path: drop path rate for SFM modules.
            use_fd: whether to enable Stage-1 feature-level DWT dual branch.
        """
        super().__init__()
        
        self.dim = dim  # 保存 dim 以供 make_model.py 访问
        self.img_size = img_size
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        
        # SIE (Side Information Embedding)
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, dim, 1, 1))
            trunc_normal_(self.sie_embed, std=.02)
            print(f'[MambaVision SIE] camera={camera}, view={view}')
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, dim, 1, 1))
            trunc_normal_(self.sie_embed, std=.02)
            print(f'[MambaVision SIE] camera={camera}')
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, dim, 1, 1))
            trunc_normal_(self.sie_embed, std=.02)
            print(f'[MambaVision SIE] view={view}')
        else:
            self.sie_embed = None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()

        current_dim = dim
        num_features = current_dim
        
        # Target projection dimension (always 512, regardless of fine_branch)
        self.proj_dim = 512
        
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            use_global = (i in global_stages) and (not conv)
            use_sasf = (i in sasf_stages) and (not conv)
            
            # Stage 3 always uses 512-dim (after main_proj at original downsample position)
            stage_dim = current_dim
            if i == 3:
                stage_dim = self.proj_dim  # 512 for Stage 3
            
            level = MambaVisionLayer(
                dim=stage_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 2),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                use_global=use_global,
                use_sasf=use_sasf,
            )
            self.levels.append(level)
            if level.downsample is not None:
                current_dim = level.downsample.reduction[0].out_channels
            else:
                current_dim = stage_dim  # Update for Stage 3 (512)
            num_features = current_dim
        
        if global_stages:
            print(f'[MambaVision] Global attention enabled for stages: {global_stages}')
        
        # Dimension Projection (384 → 512) - ALWAYS ACTIVE
        # Applied between Stage 2 and Stage 3 (at original downsample position)
        # Using 1x1 Conv to preserve spatial structure and semantics
        # No ReLU: matches native Downsample behavior (linear projection only)
        final_dim = dim * 4  # 384
        self.main_proj = nn.Sequential(
            nn.Conv2d(final_dim, self.proj_dim, 1, bias=False),
            nn.BatchNorm2d(self.proj_dim),
        )
        print(f'[MambaVision] Dimension projection: {final_dim} -> {self.proj_dim} (between Stage 2-3)')
        
        # num_features is always 512 now (after Stage 3)
        self.num_features = self.proj_dim
        
        # Hierarchical SimpleFusionMamba (SFM) modules
        self.use_sfm = use_sfm
        self.use_fd = use_fd
        if self.use_fd and self.use_sfm:
            raise ValueError('Feature DWT branch and SFM should not be enabled at the same time.')
        if self.use_fd:
            stage1_dim = dim * 2
            self.fd_branch = FeatureDWTBranch(stage1_dim, self.proj_dim)
            print(f'[MambaVision] Feature-level DWT branch enabled: Stage1 {stage1_dim} -> {self.proj_dim}')

        self.sfm_depths = list(sfm_depths)
        while len(self.sfm_depths) < 3:
            self.sfm_depths.append(0)
        self.sfm_drop_path = sfm_drop_path
        
        if self.use_sfm:
            s1_dim = dim * 2  # 160 or 192
            s2_dim = dim * 4  # 320 or 384
            s3_dim = dim * 4  # 320 or 384
            s4_dim = self.proj_dim # 512
            
            # ========== 统一 GateNet 融合 (Gated Concat 方案) ==========
            # 门控机制统一使用 GateNet，对齐策略分层 (浅层去噪、深层线性)
            # 1. SFM_S12 (Stage 1 + Stage 2) - 强去噪
            if self.sfm_depths[0] > 0:
                self.sfm_s12 = SimpleFusionMambaBlock(
                    low_dim=s1_dim,
                    high_dim=s2_dim,
                    out_dim=s2_dim,
                    target_size=(16, 8),
                    depth=self.sfm_depths[0],
                    drop_path=self.sfm_drop_path,
                    align_type='denoise_strong',
                )
            
            # 2. SFM_S23 (F12 or Stage 2 + Stage 3) - 弱去噪
            if self.sfm_depths[1] > 0:
                self.sfm_s23 = SimpleFusionMambaBlock(
                    low_dim=s2_dim,
                    high_dim=s3_dim,
                    out_dim=s3_dim,
                    target_size=(16, 8),
                    depth=self.sfm_depths[1],
                    drop_path=self.sfm_drop_path,
                    align_type='denoise_weak',
                )
            
            # 3. SFM_S34 (F23 or Stage 3 + Stage 4) - 线性对齐
            if self.sfm_depths[2] > 0:
                self.sfm_s34 = SimpleFusionMambaBlock(
                    low_dim=s3_dim,
                    high_dim=s4_dim,
                    out_dim=s4_dim,
                    target_size=(16, 8),
                    depth=self.sfm_depths[2],
                    drop_path=self.sfm_drop_path,
                    align_type='linear',
                )
            print(f'[MambaVision] Unified GateNet SFM enabled, depths={self.sfm_depths}')

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, cam_label=None, view_label=None, return_features=False):
        x = self.patch_embed(x)
        
        # Apply SIE embedding (broadcast from 1x1 to HxW)
        if self.sie_embed is not None and cam_label is not None:
            # Clamp indices to valid range for safety
            max_idx = self.sie_embed.shape[0] - 1
            if self.cam_num > 1 and self.view_num > 1 and view_label is not None:
                idx = cam_label * self.view_num + view_label
                idx = idx.clamp(0, max_idx)
                sie = self.sie_xishu * self.sie_embed[idx]
            elif self.cam_num > 1:
                idx = cam_label.clamp(0, max_idx)
                sie = self.sie_xishu * self.sie_embed[idx]
            elif self.view_num > 1 and view_label is not None:
                idx = view_label.clamp(0, max_idx)
                sie = self.sie_xishu * self.sie_embed[idx]
            else:
                sie = 0
            x = x + sie
        
        if return_features:
            features = []
            for i, level in enumerate(self.levels):
                x = level(x)
                features.append(x)
            return features
        
        # Stage 1, 2
        x = self.levels[0](x)  # Stage 1 -> (B, 192, 32, 16)
        feat_s1 = x
        freq_map = self.fd_branch(feat_s1) if self.use_fd else None
        x = self.levels[1](x)  # Stage 2 -> (B, 384, 16, 8)
        feat_s2 = x
        
        # Stage 3
        x = self.levels[2](x)  # Stage 3 -> (B, 384, 16, 8)
        feat_s3 = x
        
        # Dimension Projection (384 -> 512) and Stage 4
        x = self.main_proj(x) 
        x = self.levels[3](x)  # Stage 4 -> (B, 512, 16, 8)
        feat_s4 = x
        if self.use_fd:
            return {
                'backbone_map': feat_s4,
                'freq_map': freq_map,
            }
        
        # SFM hierarchical processing
        if self.use_sfm:
            fused_maps = []
            
            # --- Tier 1: S1 + S2 ---
            curr_fused = None
            if self.sfm_depths[0] > 0:
                curr_fused = self.sfm_s12(feat_s1, feat_s2)
                fused_maps.append(curr_fused)
            
            # --- Tier 2: (F12 or S2) + S3 ---
            if self.sfm_depths[1] > 0:
                low_feat = curr_fused if curr_fused is not None else feat_s2
                curr_fused = self.sfm_s23(low_feat, feat_s3)
                fused_maps.append(curr_fused)
            
            # --- Tier 3: (F23 or S3) + S4 ---
            if self.sfm_depths[2] > 0:
                low_feat = curr_fused if curr_fused is not None else feat_s3
                curr_fused = self.sfm_s34(low_feat, feat_s4)
                fused_maps.append(curr_fused)
            
            return {
                'backbone_map': feat_s4,
                'fused_maps': fused_maps,  # List of maps for deep supervision
            }
        else:
            # Return SPATIAL MAP (B, C, H, W)
            return feat_s4

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        elif 'model' in param_dict:
            param_dict = param_dict['model']
        
        # Remove module. prefix if exists
        if list(param_dict.keys())[0].startswith('module.'):
            param_dict = {k[7:]: v for k, v in param_dict.items()}
        
        # Filter and load compatible weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in param_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f'Loading pretrained MambaVision model from {model_path}')
        print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} parameters')


def mambavision_tiny_reid(img_size=(256, 128), pretrained_path='', **kwargs):
    """MambaVision-Tiny for ReID"""
    # Extract parameters from kwargs
    drop_path_rate = kwargs.pop('drop_path_rate', 0.2)
    global_stages = kwargs.pop('global_stages', [])
    sasf_stages = kwargs.pop('sasf_stages', [])
    use_sfm = kwargs.pop('use_sfm', False)
    sfm_num_layers = kwargs.pop('sfm_num_layers', 1)
    sfm_depths = kwargs.pop('sfm_depths', [1, 1])
    sfm_drop_path = kwargs.pop('sfm_drop_path', 0.0)
    use_fd = kwargs.pop('use_fd', False)
    
    model = MambaVisionBackbone(
        img_size=img_size,
        dim=80,
        in_dim=32,
        depths=[1, 3, 8, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        global_stages=global_stages,
        sasf_stages=sasf_stages,
        use_sfm=use_sfm,
        sfm_num_layers=sfm_num_layers,
        sfm_depths=sfm_depths,
        sfm_drop_path=sfm_drop_path,
        use_fd=use_fd,
        **kwargs
    )
    if pretrained_path:
        model.load_param(pretrained_path)
    return model


def mambavision_small_reid(img_size=(256, 128), pretrained_path='', **kwargs):
    """MambaVision-Small for ReID"""
    # Extract parameters from kwargs
    drop_path_rate = kwargs.pop('drop_path_rate', 0.2)
    global_stages = kwargs.pop('global_stages', [])
    sasf_stages = kwargs.pop('sasf_stages', [])
    use_sfm = kwargs.pop('use_sfm', False)
    sfm_num_layers = kwargs.pop('sfm_num_layers', 1)
    sfm_depths = kwargs.pop('sfm_depths', [1, 1])
    sfm_drop_path = kwargs.pop('sfm_drop_path', 0.0)
    use_fd = kwargs.pop('use_fd', False)
    
    model = MambaVisionBackbone(
        img_size=img_size,
        dim=96,
        in_dim=64,
        depths=[3, 3, 7, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 16, 8],
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        global_stages=global_stages,
        sasf_stages=sasf_stages,
        use_sfm=use_sfm,
        sfm_num_layers=sfm_num_layers,
        sfm_depths=sfm_depths,
        sfm_drop_path=sfm_drop_path,
        use_fd=use_fd,
        **kwargs
    )

    if pretrained_path:
        model.load_param(pretrained_path)
    return model


def mambavision_base_reid(img_size=(256, 128), pretrained_path='', **kwargs):
    """MambaVision-Base for ReID"""
    # Extract parameters from kwargs
    drop_path_rate = kwargs.pop('drop_path_rate', 0.3)
    global_stages = kwargs.pop('global_stages', [])
    sasf_stages = kwargs.pop('sasf_stages', [])
    use_sfm = kwargs.pop('use_sfm', False)
    sfm_num_layers = kwargs.pop('sfm_num_layers', 1)
    sfm_depths = kwargs.pop('sfm_depths', [1, 1])
    sfm_drop_path = kwargs.pop('sfm_drop_path', 0.0)
    use_fd = kwargs.pop('use_fd', False)
    
    model = MambaVisionBackbone(
        img_size=img_size,
        dim=128,
        in_dim=64,
        depths=[3, 3, 10, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 16, 16],
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
        global_stages=global_stages,
        layer_scale=1e-5,
        sasf_stages=sasf_stages,
        use_sfm=use_sfm,
        sfm_num_layers=sfm_num_layers,
        sfm_depths=sfm_depths,
        sfm_drop_path=sfm_drop_path,
        use_fd=use_fd,
        **kwargs
    )
    if pretrained_path:
        model.load_param(pretrained_path)
    return model

