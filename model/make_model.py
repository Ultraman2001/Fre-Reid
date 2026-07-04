import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, Bottleneck
import copy
import os
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .backbones.mambavision.mamba_vision_reid import (
    MambaVisionMixer,
    mambavision_tiny_reid,
    mambavision_small_reid,
    mambavision_base_reid,
)
from .backbones.osnet import osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25, osnet_ibn_x1_0
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1e-6)
        # 强制在 FP32 下计算，防止 FP16 下 x.pow(3) 溢出
        x_dtype = x.dtype
        x = x.float().clamp(min=self.eps).pow(p.view(1, 1, 1, 1))
        x = x.mean(dim=(-1, -2), keepdim=True)
        return x.pow(1.0 / p.view(1, 1, 1, 1)).to(x_dtype)


class AvgPool(nn.Module):
    """Average Pooling wrapper with FP32 protection."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(x.float(), 1).to(x.dtype)


class MaxPool(nn.Module):
    """Max Pooling wrapper with FP32 protection."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_max_pool2d(x.float(), 1).to(x.dtype)


class AvgMaxPool(nn.Module):
    """Average + Max Pooling with FP32 protection and mean scaling."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        avg = nn.functional.adaptive_avg_pool2d(x_f, 1)
        max_p = nn.functional.adaptive_max_pool2d(x_f, 1)
        # 使用均值 (Avg + Max) / 2 保持与原特征相同的量级
        return ((avg + max_p) / 2.0).to(x.dtype)


def create_pooling(pooling_type: str):
    """
    工厂函数：根据配置创建池化层
    
    Args:
        pooling_type: 'gem', 'avg', 'max', 'avg_max'
    
    Returns:
        Pooling module
    """
    pooling_type = pooling_type.lower()
    if pooling_type == 'gem':
        return GeM()
    elif pooling_type == 'avg':
        return AvgPool()
    elif pooling_type == 'max':
        return MaxPool()
    elif pooling_type in ('avg_max', 'avgmax'):
        return AvgMaxPool()
    else:
        print(f"[Warning] Unknown pooling type '{pooling_type}', using GeM")
        return GeM()


class StripeTokenInsertion(nn.Module):
    """Insert pooled stripe tokens into the flattened spatial-token sequence."""

    def __init__(
        self,
        dim,
        num_stripes=4,
        mixer_type='mambavision',
        token_pooling_type='gem',
        mode='even',
        kernel_size=3,
        mlp_ratio=2.0,
        init_scale=1e-3,
    ):
        super().__init__()
        if mode not in ('head', 'tail', 'even'):
            raise ValueError("TOKEN_INSERTION.MODE must be one of: head, tail, even")
        if kernel_size % 2 == 0:
            raise ValueError("TOKEN_INSERTION.KERNEL_SIZE must be odd")

        self.num_stripes = num_stripes
        self.mixer_type = mixer_type
        self.mode = mode
        self.token_pooling = create_pooling(token_pooling_type)
        self.norm = nn.LayerNorm(dim)
        if mixer_type == 'mambavision':
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=kernel_size,
                expand=1,
                use_sasf=False,
            )
        elif mixer_type == 'conv':
            hidden_dim = int(dim * mlp_ratio)
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.act = nn.GELU()
            self.dwconv = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
            )
            self.fc2 = nn.Linear(hidden_dim, dim)
        else:
            raise ValueError("TOKEN_INSERTION.MIXER_TYPE must be one of: mambavision, conv")
        self.gamma = nn.Parameter(torch.ones(1) * init_scale)

    def _stripe_tokens(self, feature_map):
        stripes = torch.chunk(feature_map, self.num_stripes, dim=2)
        return torch.stack(
            [self.token_pooling(stripe).flatten(1) for stripe in stripes],
            dim=1,
        )

    def _insert_even(self, seq, stripe_tokens, height, width):
        stripe_h = height // self.num_stripes
        stripe_len = stripe_h * width
        pieces = []
        for idx in range(self.num_stripes):
            start = idx * stripe_len
            end = (idx + 1) * stripe_len
            pieces.append(stripe_tokens[:, idx:idx + 1])
            pieces.append(seq[:, start:end])
        return torch.cat(pieces, dim=1), stripe_len

    def _remove_even(self, tokens, stripe_len):
        pieces = []
        cursor = 0
        for _ in range(self.num_stripes):
            cursor += 1
            pieces.append(tokens[:, cursor:cursor + stripe_len])
            cursor += stripe_len
        return torch.cat(pieces, dim=1)

    def forward(self, feature_map):
        b, c, h, w = feature_map.shape
        if h % self.num_stripes != 0:
            raise ValueError(
                "Feature-map height must be divisible by LOCAL_STRIPE.NUM_STRIPES"
            )

        seq = feature_map.flatten(2).transpose(1, 2).contiguous()
        stripe_tokens = self._stripe_tokens(feature_map)

        if self.mode == 'head':
            tokens = torch.cat([stripe_tokens, seq], dim=1)
            stripe_len = None
        elif self.mode == 'tail':
            tokens = torch.cat([seq, stripe_tokens], dim=1)
            stripe_len = None
        else:
            tokens, stripe_len = self._insert_even(seq, stripe_tokens, h, w)

        y = self.norm(tokens)
        if self.mixer_type == 'mambavision':
            y = self.mixer(y)
        else:
            y = self.fc1(y)
            y = self.act(y)
            y = self.dwconv(y.transpose(1, 2)).transpose(1, 2).contiguous()
            y = self.fc2(y)
        tokens = tokens + self.gamma * y

        if self.mode == 'head':
            seq = tokens[:, self.num_stripes:]
        elif self.mode == 'tail':
            seq = tokens[:, :h * w]
        else:
            seq = self._remove_even(tokens, stripe_len)

        return seq.transpose(1, 2).contiguous().view(b, c, h, w)


class FSLoRAFLM(nn.Module):
    """Frequency learning module used inside layer-wise FSLoRA adapters."""

    def __init__(self, rank, low_cutoff=0.30, high_cutoff=0.40, transition=0.0):
        super().__init__()
        self.low_cutoff = float(low_cutoff)
        self.high_cutoff = float(high_cutoff)
        self.transition = max(float(transition), 0.0)
        hidden_dim = max(rank // 4, 4)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rank, hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1, bias=True),
        )
        self.register_buffer('last_freq_weight', torch.ones(2) * 0.5, persistent=False)

    def _masks(self, height, width, device, dtype):
        fy = torch.fft.fftfreq(height, device=device).abs().view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device).abs().view(1, width // 2 + 1)
        radius = torch.sqrt(fy * fy + fx * fx)
        radius = radius / radius.max().clamp_min(1e-6)
        if self.transition > 0:
            low = torch.sigmoid((self.low_cutoff - radius) / self.transition)
            high = torch.sigmoid((radius - self.high_cutoff) / self.transition)
        else:
            low = (radius <= self.low_cutoff).float()
            high = (radius >= self.high_cutoff).float()
        low = low.view(1, 1, height, width // 2 + 1).to(dtype=dtype)
        high = high.view(1, 1, height, width // 2 + 1).to(dtype=dtype)
        return low, high

    def forward(self, x):
        batch_size, _, height, width = x.shape
        freq = torch.fft.rfft2(x, norm='ortho')
        low, high = self._masks(height, width, x.device, x.dtype)
        weight = torch.softmax(self.router(x.float()), dim=1).to(dtype=freq.real.dtype)
        self.last_freq_weight = weight.detach().mean(dim=(0, 2, 3))
        low_weight = weight[:, 0:1].view(batch_size, 1, 1, 1)
        high_weight = weight[:, 1:2].view(batch_size, 1, 1, 1)
        freq = low_weight * freq * low + high_weight * freq * high
        return torch.fft.irfft2(freq, s=(height, width), norm='ortho')


class FSLoRASLM(nn.Module):
    """Spatial router that produces expert masks for FSLoRA."""

    def __init__(self, in_dim, num_experts):
        super().__init__()
        self.num_experts = int(num_experts)
        self.router = nn.Conv2d(in_dim, self.num_experts, 1, bias=False)

    def forward(self, x):
        return torch.softmax(self.router(x.float()), dim=1)


class FSLoRALinear(nn.Module):
    """LoRA wrapper for Linear weights: W0(x) + B(SLM(FLM(A(x))))."""

    accepts_spatial_context = True
    is_fslora_adapter = True

    def __init__(
        self,
        linear,
        rank=16,
        num_experts=2,
        init_gamma=1e-3,
        freq_low_cutoff=0.30,
        freq_high_cutoff=0.40,
        freq_transition=0.08,
        freeze_base=False,
    ):
        super().__init__()
        self.base = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = min(max(int(rank), 4), self.in_features, self.out_features)
        self.num_experts = max(int(num_experts), 1)
        self.fslora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.fslora_flm = FSLoRAFLM(
            self.rank,
            low_cutoff=freq_low_cutoff,
            high_cutoff=freq_high_cutoff,
            transition=freq_transition,
        )
        self.fslora_slm = FSLoRASLM(self.in_features, self.num_experts)
        self.fslora_B = nn.ModuleList([
            nn.Linear(self.rank, self.out_features, bias=False)
            for _ in range(self.num_experts)
        ])
        self.fslora_gamma = nn.Parameter(torch.ones(1) * float(init_gamma))
        self.fslora_gamma._no_weight_decay = True
        self._context = None
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fslora_A.weight, a=0, mode='fan_out')
        for expert in self.fslora_B:
            nn.init.zeros_(expert.weight)

    def set_spatial_context(self, H=None, W=None, batch_size=None, seq_len=None):
        self._context = (H, W, batch_size, seq_len)

    def _to_map(self, z, H, W, batch_size, seq_len):
        if H is None or W is None:
            return None
        H, W = int(H), int(W)
        if z.dim() == 3 and H * W == z.shape[1]:
            return z.transpose(1, 2).contiguous().view(z.shape[0], self.rank, H, W)
        if (
            z.dim() == 2
            and batch_size is not None
            and seq_len is not None
            and H * W == int(seq_len)
            and z.shape[0] == int(batch_size) * int(seq_len)
        ):
            return z.view(int(batch_size), int(seq_len), self.rank).transpose(1, 2).contiguous().view(
                int(batch_size), self.rank, H, W
            )
        return None

    def _from_map(self, z_map, ref, batch_size, seq_len):
        if ref.dim() == 3:
            return z_map.flatten(2).transpose(1, 2).contiguous()
        return z_map.flatten(2).transpose(1, 2).contiguous().view(int(batch_size) * int(seq_len), self.rank)

    def _router_mask(self, x, H, W, batch_size, seq_len):
        x_map = self._to_input_map(x.float(), H, W, batch_size, seq_len)
        if x_map is None:
            return None
        return self.fslora_slm(x_map)

    def _to_input_map(self, x, H, W, batch_size, seq_len):
        if H is None or W is None:
            return None
        H, W = int(H), int(W)
        if x.dim() == 3 and H * W == x.shape[1]:
            return x.transpose(1, 2).contiguous().view(x.shape[0], self.in_features, H, W)
        if (
            x.dim() == 2
            and batch_size is not None
            and seq_len is not None
            and H * W == int(seq_len)
            and x.shape[0] == int(batch_size) * int(seq_len)
        ):
            return x.view(int(batch_size), int(seq_len), self.in_features).transpose(1, 2).contiguous().view(
                int(batch_size), self.in_features, H, W
            )
        return None

    def _apply_experts(self, z, mask, batch_size, seq_len):
        expert_outs = [expert(z) for expert in self.fslora_B]
        if mask is None:
            return sum(expert_outs) / float(len(expert_outs))
        if z.dim() == 3:
            mask_seq = mask.flatten(2).transpose(1, 2).contiguous()
            out = 0
            for idx, expert_out in enumerate(expert_outs):
                out = out + mask_seq[:, :, idx:idx + 1] * expert_out
            return out
        mask_seq = mask.flatten(2).transpose(1, 2).contiguous().view(
            int(batch_size) * int(seq_len),
            self.num_experts,
        )
        out = 0
        for idx, expert_out in enumerate(expert_outs):
            out = out + mask_seq[:, idx:idx + 1] * expert_out
        return out

    def forward(self, x, H=None, W=None, batch_size=None, seq_len=None):
        out = self.base(x)
        if H is None and self._context is not None:
            H, W, batch_size, seq_len = self._context
        with torch.amp.autocast(x.device.type, enabled=False):
            x_float = x.float()
            z = self.fslora_A(x_float)
            z_map = self._to_map(z, H, W, batch_size, seq_len)
            if z_map is not None:
                z_map = self.fslora_flm(z_map)
                z = self._from_map(z_map, z, batch_size, seq_len)
            mask = self._router_mask(x_float, H, W, batch_size, seq_len)
            delta = self._apply_experts(z, mask, batch_size, seq_len)
            delta = self.fslora_gamma.float() * delta
        return out + delta.to(out.dtype)


class FSLoRAConv2d(nn.Module):
    """LoRA wrapper for ConvBlock convolutions in shallow MambaVision stages."""

    is_fslora_adapter = True

    def __init__(
        self,
        conv,
        rank=16,
        num_experts=2,
        init_gamma=1e-3,
        freq_low_cutoff=0.30,
        freq_high_cutoff=0.40,
        freq_transition=0.08,
        freeze_base=False,
    ):
        super().__init__()
        self.base = conv
        self.rank = min(max(int(rank), 4), conv.in_channels, conv.out_channels)
        self.num_experts = max(int(num_experts), 1)
        self.fslora_A = nn.Conv2d(conv.in_channels, self.rank, 1, bias=False)
        self.fslora_flm = FSLoRAFLM(
            self.rank,
            low_cutoff=freq_low_cutoff,
            high_cutoff=freq_high_cutoff,
            transition=freq_transition,
        )
        self.fslora_slm = FSLoRASLM(conv.in_channels, self.num_experts)
        self.fslora_B = nn.ModuleList([
            nn.Conv2d(self.rank, conv.out_channels, 1, bias=False)
            for _ in range(self.num_experts)
        ])
        self.fslora_gamma = nn.Parameter(torch.ones(1) * float(init_gamma))
        self.fslora_gamma._no_weight_decay = True
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fslora_A.weight, a=0, mode='fan_out')
        for expert in self.fslora_B:
            nn.init.zeros_(expert.weight)

    def forward(self, x):
        out = self.base(x)
        with torch.amp.autocast(x.device.type, enabled=False):
            x_float = x.float()
            z = self.fslora_A(x_float)
            z = self.fslora_flm(z)
            mask = self.fslora_slm(x_float)
            delta = 0
            for idx, expert in enumerate(self.fslora_B):
                delta = delta + mask[:, idx:idx + 1] * expert(z)
            delta = self.fslora_gamma.float().view(1, 1, 1, 1) * delta
        return out + delta.to(out.dtype)


def _wrap_fslora_children(
    module,
    rank,
    num_experts,
    init_gamma,
    freq_low_cutoff,
    freq_high_cutoff,
    freq_transition,
    freeze_base,
    wrap_conv,
):
    count = 0
    parent_type = module.__class__.__name__
    for name, child in list(module.named_children()):
        if getattr(child, 'is_fslora_adapter', False):
            continue
        if isinstance(child, nn.Linear) and (
            (parent_type == 'MambaVisionMixer' and name in ('in_proj', 'out_proj'))
            or (parent_type == 'Attention' and name in ('qkv', 'proj'))
            or (parent_type == 'Mlp' and name in ('fc1', 'fc2'))
        ):
            setattr(module, name, FSLoRALinear(
                child,
                rank=rank,
                num_experts=num_experts,
                init_gamma=init_gamma,
                freq_low_cutoff=freq_low_cutoff,
                freq_high_cutoff=freq_high_cutoff,
                freq_transition=freq_transition,
                freeze_base=freeze_base,
            ))
            count += 1
        elif (
            wrap_conv
            and isinstance(child, nn.Conv2d)
            and child.stride == (1, 1)
            and child.groups == 1
            and parent_type == 'ConvBlock'
            and name in ('conv1', 'conv2')
        ):
            setattr(module, name, FSLoRAConv2d(
                child,
                rank=rank,
                num_experts=num_experts,
                init_gamma=init_gamma,
                freq_low_cutoff=freq_low_cutoff,
                freq_high_cutoff=freq_high_cutoff,
                freq_transition=freq_transition,
                freeze_base=freeze_base,
            ))
            count += 1
        else:
            count += _wrap_fslora_children(
                child,
                rank=rank,
                num_experts=num_experts,
                init_gamma=init_gamma,
                freq_low_cutoff=freq_low_cutoff,
                freq_high_cutoff=freq_high_cutoff,
                freq_transition=freq_transition,
                freeze_base=freeze_base,
                wrap_conv=wrap_conv,
            )
    return count


def install_fslora_adapters(
    backbone,
    rank=16,
    num_experts=2,
    init_gamma=1e-3,
    freq_low_cutoff=0.30,
    freq_high_cutoff=0.40,
    freq_transition=0.08,
    freeze_base=False,
    wrap_conv=True,
    target_stages=None,
):
    if not hasattr(backbone, 'levels'):
        return 0
    if target_stages is None or len(target_stages) == 0:
        target_stages = set(range(len(backbone.levels)))
    else:
        target_stages = {int(stage) for stage in target_stages}

    count = 0
    for stage_idx, level in enumerate(backbone.levels):
        if stage_idx not in target_stages:
            continue
        if not hasattr(level, 'blocks'):
            continue
        for block in level.blocks:
            count += _wrap_fslora_children(
                block,
                rank=rank,
                num_experts=num_experts,
                init_gamma=init_gamma,
                freq_low_cutoff=freq_low_cutoff,
                freq_high_cutoff=freq_high_cutoff,
                freq_transition=freq_transition,
                freeze_base=freeze_base,
                wrap_conv=wrap_conv,
            )
    return count


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride)
            print('using resnet101_ibn_a as a backbone')

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.is_puzzle = 'puzzle' in cfg.MODEL.TRANSFORMER_TYPE.lower()
        self.is_mambavision = 'mamba' in cfg.MODEL.TRANSFORMER_TYPE.lower()
        self.use_sfm = False
        self.use_pam = cfg.INPUT.PAM.ENABLED
        
        self.pooling = None
        local_cfg = getattr(cfg.MODEL, 'LOCAL_STRIPE', None)
        token_cfg = getattr(local_cfg, 'TOKEN_INSERTION', None) if local_cfg is not None else None
        fslora_cfg = getattr(cfg.MODEL, 'FSLORA', None)
        self.fslora_enabled = bool(getattr(fslora_cfg, 'ENABLED', False))
        self.fslora_rank = int(getattr(fslora_cfg, 'RANK', 32))
        self.fslora_num_experts = int(getattr(fslora_cfg, 'NUM_EXPERTS', 2))
        self.fslora_init_gamma = float(getattr(fslora_cfg, 'INIT_GAMMA', 1e-3))
        self.fslora_freq_low_cutoff = float(getattr(fslora_cfg, 'FREQ_LOW_CUTOFF', 0.30))
        self.fslora_freq_high_cutoff = float(getattr(fslora_cfg, 'FREQ_HIGH_CUTOFF', 0.40))
        self.fslora_freq_transition = float(getattr(fslora_cfg, 'FREQ_TRANSITION', 0.0))
        self.fslora_wrap_conv = bool(getattr(fslora_cfg, 'WRAP_CONV', True))
        self.fslora_freeze_base = bool(getattr(fslora_cfg, 'FREEZE_BASE', False))
        self.fslora_target_stages = list(getattr(fslora_cfg, 'TARGET_STAGES', []))
        self.fslora_num_adapters = 0
        self.local_stripe_enabled = bool(getattr(local_cfg, 'ENABLED', False))
        self.local_num_stripes = int(getattr(local_cfg, 'NUM_STRIPES', 4))
        self.local_inference = str(getattr(local_cfg, 'INFERENCE', 'concat')).lower()
        self.local_pooling = None
        self.stripe_token_insertion = None
        self.local_token_insertion_enabled = bool(getattr(token_cfg, 'ENABLED', False))
        self.local_token_insertion_mixer_type = str(getattr(token_cfg, 'MIXER_TYPE', 'mambavision')).lower()
        default_token_pooling = str(getattr(local_cfg, 'POOLING_TYPE', 'gem')) if local_cfg is not None else 'gem'
        self.local_token_pooling_type = str(getattr(token_cfg, 'TOKEN_POOLING_TYPE', default_token_pooling)).lower()
        self.local_token_insertion_mode = str(getattr(token_cfg, 'MODE', 'even')).lower()
        self.local_token_insertion_kernel = int(getattr(token_cfg, 'KERNEL_SIZE', 3))
        self.local_token_insertion_mlp_ratio = float(getattr(token_cfg, 'MLP_RATIO', 2.0))
        self.local_token_insertion_init_scale = float(getattr(token_cfg, 'INIT_SCALE', 1e-3))

        if self.is_mambavision:
            # 获取配置
            global_stages = list(getattr(cfg.MODEL.MAMBAVISION, 'GLOBAL_STAGES', []))
            sasf_stages = list(getattr(cfg.MODEL.MAMBAVISION, 'SASF_STAGES', []))
            
            # SFM 配置
            self.use_sfm = getattr(cfg.MODEL.MAMBAVISION, 'USE_SFM', False)
            sfm_num_layers = getattr(cfg.MODEL.MAMBAVISION, 'SFM_NUM_LAYERS', 1)
            sfm_depths = list(getattr(cfg.MODEL.MAMBAVISION, 'SFM_DEPTHS', [1, 1]))
            sfm_drop_path = getattr(cfg.MODEL.MAMBAVISION, 'SFM_DROP_PATH', 0.0)
            
            # SIE 配置
            self.use_sie = getattr(cfg.MODEL, 'SIE_CAMERA', False)
            sie_xishu = getattr(cfg.MODEL, 'SIE_XISHU', 1.5)
            
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
                img_size=cfg.INPUT.SIZE_TRAIN,
                pretrained_path=model_path if pretrain_choice == 'imagenet' else '',
                sie_xishu=sie_xishu,
                camera=camera_num if self.use_sie else 0,
                view=view_num if self.use_sie else 0,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                drop_rate=cfg.MODEL.DROP_OUT,
                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                global_stages=global_stages,
                sasf_stages=sasf_stages,
                use_sfm=self.use_sfm,
                sfm_num_layers=sfm_num_layers,
                sfm_depths=sfm_depths,
                sfm_drop_path=sfm_drop_path,
            )
            # Get feature dimension from MambaVision
            self.in_planes = self.base.num_features
            pooling_type = getattr(cfg.MODEL, 'POOLING_TYPE', 'gem')
            self.pooling = create_pooling(pooling_type)
            print(f'[Model] Using pooling type: {pooling_type}')
            
            # SFM 多级聚合 heads
            if self.use_sfm:
                self.pooling_fused = nn.ModuleList()
                self.bottleneck_fused = nn.ModuleList()
                self.classifier_fused = nn.ModuleList()

        else:
            extra_kwargs = {}
            if 'mamba_hybrid' in cfg.MODEL.TRANSFORMER_TYPE.lower():
                m_layers = cfg.MODEL.get('MAMBA_LAYERS', None)
                if m_layers is not None:
                    extra_kwargs['mamba_layers'] = m_layers
                m_d_state = cfg.MODEL.get('MAMBA_D_STATE', None)
                if m_d_state is not None:
                    extra_kwargs.setdefault('mamba_cfg', {})
                    extra_kwargs['mamba_cfg']['d_state'] = m_d_state
                m_d_conv = cfg.MODEL.get('MAMBA_D_CONV', None)
                if m_d_conv is not None:
                    extra_kwargs.setdefault('mamba_cfg', {})
                    extra_kwargs['mamba_cfg']['d_conv'] = m_d_conv
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=3.0,
                                                            camera=0, view=0, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate= cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                            **extra_kwargs)
            if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
                self.in_planes = 384
            if pretrain_choice == 'imagenet':
                self.base.load_param(model_path)
                print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.fslora_enabled:
            if not self.is_mambavision:
                raise ValueError('FSLORA currently supports the MambaVision backbone only')
            if self.use_sfm:
                raise ValueError('FSLORA and SFM must be evaluated separately')
            self.fslora_num_adapters = install_fslora_adapters(
                self.base,
                rank=self.fslora_rank,
                num_experts=self.fslora_num_experts,
                init_gamma=self.fslora_init_gamma,
                freq_low_cutoff=self.fslora_freq_low_cutoff,
                freq_high_cutoff=self.fslora_freq_high_cutoff,
                freq_transition=self.fslora_freq_transition,
                freeze_base=self.fslora_freeze_base,
                wrap_conv=self.fslora_wrap_conv,
                target_stages=self.fslora_target_stages,
            )
            print(
                '[Model] FSLoRA enabled in backbone blocks: adapters={}, rank={}, experts={}, init_gamma={}, freq_cutoff=({:.2f},{:.2f}), wrap_conv={}, freeze_base={}, target_stages={}'.format(
                    self.fslora_num_adapters,
                    self.fslora_rank,
                    self.fslora_num_experts,
                    self.fslora_init_gamma,
                    self.fslora_freq_low_cutoff,
                    self.fslora_freq_high_cutoff,
                    self.fslora_wrap_conv,
                    self.fslora_freeze_base,
                    self.fslora_target_stages if self.fslora_target_stages else 'all',
                )
            )

        if self.local_stripe_enabled:
            if not self.is_mambavision:
                raise ValueError('LOCAL_STRIPE currently supports the MambaVision backbone only')
            if not self.use_pam:
                raise ValueError('LOCAL_STRIPE currently expects PAM training to be enabled')
            if self.use_sfm:
                raise ValueError('LOCAL_STRIPE and SFM must be evaluated separately')
            if self.local_num_stripes <= 0:
                raise ValueError('LOCAL_STRIPE.NUM_STRIPES must be positive')

            local_pooling_type = getattr(cfg.MODEL.LOCAL_STRIPE, 'POOLING_TYPE', 'gem')
            self.local_pooling = create_pooling(local_pooling_type)
            if self.local_token_insertion_enabled:
                self.stripe_token_insertion = StripeTokenInsertion(
                    dim=self.in_planes,
                    num_stripes=self.local_num_stripes,
                    mixer_type=self.local_token_insertion_mixer_type,
                    token_pooling_type=self.local_token_pooling_type,
                    mode=self.local_token_insertion_mode,
                    kernel_size=self.local_token_insertion_kernel,
                    mlp_ratio=self.local_token_insertion_mlp_ratio,
                    init_scale=self.local_token_insertion_init_scale,
                )

            self.local_bottlenecks = nn.ModuleList()
            self.local_classifiers = nn.ModuleList()
            for _ in range(self.local_num_stripes):
                bn = nn.BatchNorm1d(self.in_planes)
                bn.bias.requires_grad_(False)
                bn.apply(weights_init_kaiming)
                self.local_bottlenecks.append(bn)
                self.local_classifiers.append(copy.deepcopy(self.classifier))
            print(
                '[Model] LocalStripe enabled on BA branch: stripes={}, pooling={}, token_insertion={}, mixer={}, token_pooling={}'.format(
                    self.local_num_stripes,
                    local_pooling_type,
                    self.local_token_insertion_enabled,
                    self.local_token_insertion_mixer_type,
                    self.local_token_pooling_type,
                )
            )

        if self.use_pam:
            if not self.is_mambavision:
                raise ValueError('PAM currently supports the MambaVision backbone only')
            if self.use_sfm:
                raise ValueError('PAM and SFM must be evaluated separately')

            self.bottleneck_pam_crop = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_pam_crop.bias.requires_grad_(False)
            self.bottleneck_pam_crop.apply(weights_init_kaiming)

            self.bottleneck_pam_erase = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_pam_erase.bias.requires_grad_(False)
            self.bottleneck_pam_erase.apply(weights_init_kaiming)

            self.classifier_pam_crop = copy.deepcopy(self.classifier)
            self.classifier_pam_erase = copy.deepcopy(self.classifier)
            print('[Model] PAM enabled: shared MambaVision backbone with BA, CA and EA heads')
        
        # SFM fused分支的head (ModuleList 支持多级级联与深层监督)
        if self.is_mambavision and self.use_sfm:
            # 极致稳健的维度检测方案
            # 1. 优先从 backbone 实例属性获取
            dim = getattr(self.base, 'dim', None)
            
            # 2. 如果失败，尝试从 patch_embed 获取
            if dim is None and hasattr(self.base, 'patch_embed'):
                dim = getattr(self.base.patch_embed, 'dim', None)
                
            # 3. 如果仍然失败，从配置文件字符串猜测
            if dim is None:
                t_type = cfg.MODEL.TRANSFORMER_TYPE.lower()
                if 'tiny' in t_type: dim = 80
                elif 'small' in t_type: dim = 96
                elif 'base' in t_type: dim = 128
                else: dim = 80 # 最后的兜底
            
            print(f'[SFM Head Init] Infallible detection: Detected dim={dim} for transformer {cfg.MODEL.TRANSFORMER_TYPE}')
            
            s1_dim = dim * 2  # 160 (Tiny) or 192 (Small)
            s2_dim = dim * 4  # 320 (Tiny) or 384 (Small)
            s3_dim = dim * 4  # 320 (Tiny) or 384 (Small)
            s4_dim = self.in_planes # 通常是 512
            # fused_maps = [F12(s2_dim), F23(s3_dim), F34(s4_dim)]
            possible_dims = [s2_dim, s3_dim, s4_dim]
            
            sfm_depths = list(getattr(cfg.MODEL.MAMBAVISION, 'SFM_DEPTHS', [0, 0, 0]))
            while len(sfm_depths) < 3:
                sfm_depths.append(0)
            
            head_idx = 0
            sfm_pooling_type = getattr(cfg.MODEL.MAMBAVISION, 'SFM_POOLING_TYPE', 'gem')
            print(f'[SFM] Using pooling type: {sfm_pooling_type}')
            for i, d in enumerate(sfm_depths):
                if d > 0:
                    current_in_dim = possible_dims[i]
                    self.pooling_fused.append(create_pooling(sfm_pooling_type))
                    
                    bn = nn.BatchNorm1d(current_in_dim)
                    bn.bias.requires_grad_(False)
                    bn.apply(weights_init_kaiming)
                    self.bottleneck_fused.append(bn)
                    
                    cls = nn.Linear(current_in_dim, self.num_classes, bias=False)
                    cls.apply(weights_init_classifier)
                    self.classifier_fused.append(cls)
                    head_idx += 1
            print(f'[SFM] Initialized {head_idx} hierarchical fused heads')

    def _classify(self, classifier, feat, label):
        if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
            return classifier(feat, label)
        return classifier(feat)

    def _pool_feature_map(self, feature_map, pooling=None):
        pool_layer = self.pooling if pooling is None else pooling
        if pool_layer is None:
            pooled = nn.functional.adaptive_avg_pool2d(feature_map, 1)
        else:
            pooled = pool_layer(feature_map)
        return pooled.flatten(1)

    def _refine_local_tokens(self, feature_map):
        if self.stripe_token_insertion is not None:
            return self.stripe_token_insertion(feature_map)
        return feature_map

    def _pool_local_stripes(self, feature_map):
        if feature_map.shape[2] % self.local_num_stripes != 0:
            raise ValueError(
                "Feature-map height must be divisible by LOCAL_STRIPE.NUM_STRIPES"
            )
        stripes = torch.chunk(feature_map, self.local_num_stripes, dim=2)
        return [self._pool_feature_map(stripe, self.local_pooling) for stripe in stripes]

    def _local_head_index(self, stripe_idx):
        return stripe_idx

    def _append_local_outputs(self, scores, feats, feature_map, label):
        local_feats = self._pool_local_stripes(feature_map)
        for stripe_idx, local_feat in enumerate(local_feats):
            head_idx = self._local_head_index(stripe_idx)
            local_feat_bn = self.local_bottlenecks[head_idx](local_feat)
            scores.append(
                self._classify(self.local_classifiers[head_idx], local_feat_bn, label)
            )
            feats.append(local_feat)

    def _local_inference_features(self, feature_map):
        local_feats = self._pool_local_stripes(feature_map)
        if self.neck_feat == 'after':
            local_feats = [
                self.local_bottlenecks[self._local_head_index(idx)](feat)
                for idx, feat in enumerate(local_feats)
            ]
        return torch.cat(local_feats, dim=1)

    def _format_local_inference(self, global_feat, feature_map):
        local_feat = self._local_inference_features(feature_map)
        if self.local_inference == 'global':
            final_feat = global_feat
        elif self.local_inference == 'local':
            final_feat = local_feat
        else:
            final_feat = torch.cat([global_feat, local_feat], dim=1)
        return {
            'backbone': global_feat,
            'fused': local_feat,
            'concat': final_feat,
        }

    def _forward_pam(self, x, label, cam_label, view_label):
        if not isinstance(x, (tuple, list)) or len(x) != 3:
            raise ValueError('PAM training expects BA, CA and EA image tensors')

        img_base, img_crop, img_erase = x
        stacked_img = torch.cat([img_base, img_crop, img_erase], dim=0)
        stacked_cam = torch.cat([cam_label] * 3, dim=0) if cam_label is not None else None
        stacked_view = torch.cat([view_label] * 3, dim=0) if view_label is not None else None

        output = self.base(stacked_img, cam_label=stacked_cam, view_label=stacked_view)
        if isinstance(output, dict):
            if 'fused_maps' in output:
                raise ValueError('PAM and SFM must be evaluated separately')
            output = output['backbone_map']
        map_base, map_crop, map_erase = output.chunk(3, dim=0)

        if self.local_stripe_enabled:
            map_base = self._refine_local_tokens(map_base)

        feat_base = self._pool_feature_map(map_base)
        feat_crop = self._pool_feature_map(map_crop)
        feat_erase = self._pool_feature_map(map_erase)

        feat_base_bn = self.bottleneck(feat_base)
        feat_crop_bn = self.bottleneck_pam_crop(feat_crop)
        feat_erase_bn = self.bottleneck_pam_erase(feat_erase)

        scores = [
            self._classify(self.classifier, feat_base_bn, label),
            self._classify(self.classifier_pam_crop, feat_crop_bn, label),
            self._classify(self.classifier_pam_erase, feat_erase_bn, label),
        ]
        feats = [feat_base, feat_crop, feat_erase]
        if self.local_stripe_enabled:
            self._append_local_outputs(scores, feats, map_base, label)
        return scores, feats

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.use_pam and self.training:
            return self._forward_pam(x, label, cam_label, view_label)

        if self.is_mambavision or self.is_puzzle:
            # Get backbone output
            output = self.base(x, cam_label=cam_label, view_label=view_label)
            
            # Check if SFM mode (output is dict) or normal mode (output is tensor)
            if isinstance(output, dict):
                # SFM mode: hierarchical multi-branch processing
                backbone_map = output['backbone_map']  # (B, 512, 16, 8)
                fused_maps = output['fused_maps']     # List of intermediate fused maps
                
                # Backbone head
                feat_backbone = self.pooling(backbone_map).flatten(1)
                feat_backbone_bn = self.bottleneck(feat_backbone)
                
                # Fused heads (Loop over all active fusion stages)
                all_feats = [feat_backbone]
                all_feats_bn = [feat_backbone_bn]
                
                for i, f_map in enumerate(fused_maps):
                    f_feat = self.pooling_fused[i](f_map).flatten(1)
                    f_feat_bn = self.bottleneck_fused[i](f_feat)
                    all_feats.append(f_feat)
                    all_feats_bn.append(f_feat_bn)
                
                if self.training:
                    all_scores = []
                    # 1. Backbone score
                    if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                        all_scores.append(self.classifier(all_feats_bn[0], label))
                    else:
                        all_scores.append(self.classifier(all_feats_bn[0]))
                    
                    # 2. Fused branch scores (Deep Supervision)
                    for i in range(len(fused_maps)):
                        if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                            all_scores.append(self.classifier_fused[i](all_feats_bn[i+1], label))
                        else:
                            all_scores.append(self.classifier_fused[i](all_feats_bn[i+1]))
                    
                    return all_scores, all_feats
                else:
                    # Inference: return dict with separate features for evaluation
                    if self.neck_feat == 'after':
                        backbone_feat = all_feats_bn[0]
                        fused_feat = all_feats_bn[-1]
                    else:
                        backbone_feat = all_feats[0]
                        fused_feat = all_feats[-1]
                    
                    concat_feat = torch.cat([backbone_feat, fused_feat], dim=1)
                    
                    # Return dict for multi-feature evaluation
                    return {
                        'backbone': backbone_feat,
                        'fused': fused_feat,
                        'concat': concat_feat,
                    }
            else:
                # Normal mode: single tensor output
                feature_map = output
                if self.local_stripe_enabled:
                    feature_map = self._refine_local_tokens(feature_map)
                
                # Apply pooling
                global_feat = self._pool_feature_map(feature_map)
                
                feat = self.bottleneck(global_feat)

                if self.training:
                    cls_score = self._classify(self.classifier, feat, label)
                    
                    return cls_score, global_feat
                else:
                    if self.neck_feat == 'after':
                        global_output = feat
                    else:
                        global_output = global_feat
                    if self.local_stripe_enabled:
                        return self._format_local_inference(global_output, feature_map)
                    return global_output

        else:
            # ViT/DeiT backbone
            global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
            feat = self.bottleneck(global_feat)

            if self.training:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat
            else:
                if self.neck_feat == 'after':
                    return feat
                else:
                    return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


_factory_osnet = {
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
}


class SameScaleFrequencyMambaBlock(nn.Module):
    """Spatial Mamba block for a same-scale feature map."""

    def __init__(
        self,
        dim,
        mlp_ratio=2.0,
        init_scale=0.1,
        d_state=8,
        d_conv=3,
        bidirectional=True,
    ):
        super().__init__()
        hidden_dim = max(int(dim * mlp_ratio), dim)
        self.bidirectional = bidirectional
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaVisionMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,
            use_sasf=False,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gamma1 = nn.Parameter(init_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_scale * torch.ones(dim))
        self.gamma1._no_weight_decay = True
        self.gamma2._no_weight_decay = True

    def forward(self, feature_map):
        b, c, h, w = feature_map.shape
        seq = feature_map.flatten(2).transpose(1, 2).contiguous()
        seq_norm = self.norm1(seq)
        mixed = self.mixer(seq_norm, H=h, W=w)
        if self.bidirectional:
            rev_norm = torch.flip(seq_norm, dims=(1,)).contiguous()
            mixed_rev = self.mixer(rev_norm, H=h, W=w)
            mixed = 0.5 * (mixed + torch.flip(mixed_rev, dims=(1,)))
        seq = seq + self.gamma1 * mixed
        seq = seq + self.gamma2 * self.mlp(self.norm2(seq))
        return seq.transpose(1, 2).reshape(b, c, h, w).contiguous()


class MSEFBlock(nn.Module):
    """Multi-Scale Enhancement Fusion block from Multinex.

    Spatial-channel product interaction with tanh-gated channel recalibration.
    DWConv_3x3(LN(x)) * tanh_SE(LN(x)) + x
    """

    def __init__(self, ch, reduction_ratio=16, use_res_scale=False, res_scale_init=0.1):
        super().__init__()
        hidden = max(ch // int(reduction_ratio), 1)
        self.layer_norm = nn.LayerNorm(ch)
        self.depthwise_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(ch, hidden)
        self.se_fc2 = nn.Linear(hidden, ch)
        self.res_scale = nn.Parameter(torch.ones(1) * float(res_scale_init)) if use_res_scale else None
        if self.res_scale is not None:
            self.res_scale._no_weight_decay = True

    def forward(self, x):
        identity = x
        x_norm = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_pool(x_norm).flatten(1)
        x2 = F.relu(self.se_fc1(x2))
        x2 = torch.tanh(self.se_fc2(x2)).view(x.shape[0], x.shape[1], 1, 1)
        delta = x1 * x2
        if self.res_scale is not None:
            delta = self.res_scale.to(dtype=delta.dtype) * delta
        return identity + delta


class SameScaleFrequencyMambaFusion(nn.Module):
    """Same-scale OSNet/Mamba map fusion followed by SpatialMamba refinement.

    The FDMF interface is kept for config/checkpoint compatibility, but the
    frequency low/high decomposition path has been removed. The module now uses:
        F_init = Conv1x1(cat(MambaMap, OSNetMap))
        F_out  = SpatialMamba(F_init)
    """

    def __init__(
        self,
        dim,
        osnet_dim,
        compressed_channels=64,
        lowpass_kernel=5,
        highpass_kernel=3,
        use_hamming=True,
        filter_type='none',
        mamba_depth=1,
        mamba_d_state=8,
        mamba_d_conv=3,
        mamba_init_scale=0.1,
        mamba_bidirectional=True,
        mlp_ratio=2.0,
        msef_enabled=True,
        msef_reduction_ratio=16,
        msef_res_scale_enabled=False,
        msef_res_scale_init=0.1,
    ):
        super().__init__()
        if mamba_depth < 0:
            raise ValueError('FDMF_MAMBA_DEPTH must be >= 0')
        self.dim = dim
        self.filter_type = 'none'
        self.osnet_map_proj = (
            nn.Identity()
            if osnet_dim == dim
            else nn.Sequential(
                nn.Conv2d(osnet_dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
            )
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )
        self.mamba_blocks = nn.ModuleList([
            SameScaleFrequencyMambaBlock(
                dim,
                mlp_ratio=mlp_ratio,
                init_scale=mamba_init_scale,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                bidirectional=mamba_bidirectional,
            )
            for _ in range(mamba_depth)
        ])
        self.msef = (
            MSEFBlock(
                dim,
                reduction_ratio=msef_reduction_ratio,
                use_res_scale=msef_res_scale_enabled,
                res_scale_init=msef_res_scale_init,
            )
            if msef_enabled
            else nn.Identity()
        )

        self.osnet_map_proj.apply(weights_init_kaiming)
        self.fuse_proj.apply(weights_init_kaiming)

    def forward(self, mamba_map, osnet_map):
        if osnet_map.shape[-2:] != mamba_map.shape[-2:]:
            osnet_map = F.interpolate(
                osnet_map,
                size=mamba_map.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        osnet_map = self.osnet_map_proj(osnet_map)

        fused_map = self.fuse_proj(torch.cat([mamba_map, osnet_map], dim=1))
        for block in self.mamba_blocks:
            fused_map = block(fused_map)
        fused_map = self.msef(fused_map)
        return fused_map


class StageFCU(nn.Module):
    """Bidirectional stage-level coupling inspired by Conformer FCU."""

    def __init__(
        self,
        mamba_dim,
        osnet_dim,
        init_scale=0.1,
        direction='bidirectional',
        gate_type='none',
        gate_reduction=16,
        gate_init_bias=0.0,
    ):
        super().__init__()
        self.direction = str(direction).lower()
        if self.direction not in ('bidirectional', 'osnet_to_mamba', 'mamba_to_osnet'):
            raise ValueError("FCU direction must be 'bidirectional', 'osnet_to_mamba', or 'mamba_to_osnet'")
        self.gate_type = str(gate_type).lower()
        if self.gate_type not in ('none', 'channel'):
            raise ValueError("FCU gate_type must be 'none' or 'channel'")
        self.mamba_from_osnet = nn.Sequential(
            nn.Conv2d(osnet_dim, mamba_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mamba_dim),
            nn.GELU(),
        )
        self.osnet_from_mamba = nn.Sequential(
            nn.Conv2d(mamba_dim, osnet_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(osnet_dim),
            nn.GELU(),
        )
        self.fusion_scale_mamba = nn.Parameter(torch.ones(1) * float(init_scale))
        self.fusion_scale_osnet = nn.Parameter(torch.ones(1) * float(init_scale))
        if self.gate_type == 'channel':
            self.mamba_gate = self._make_channel_gate(mamba_dim, gate_reduction, gate_init_bias)
            self.osnet_gate = self._make_channel_gate(osnet_dim, gate_reduction, gate_init_bias)
        else:
            self.mamba_gate = None
            self.osnet_gate = None

    @staticmethod
    def _make_channel_gate(dim, reduction, init_bias):
        hidden_dim = max(int(dim) // max(int(reduction), 1), 4)
        gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(gate[3].weight)
        nn.init.constant_(gate[3].bias, float(init_bias))
        return gate

    def _gate_delta(self, target_map, delta_map, gate):
        if gate is None:
            return delta_map
        gate_input = torch.cat([target_map, delta_map], dim=1).float()
        return delta_map * gate(gate_input).to(delta_map.dtype)

    def forward(self, mamba_map, osnet_map):
        new_mamba_map = mamba_map
        new_osnet_map = osnet_map

        if self.direction in ('bidirectional', 'osnet_to_mamba'):
            osnet_to_mamba = self.mamba_from_osnet(osnet_map)
            if osnet_to_mamba.shape[-2:] != mamba_map.shape[-2:]:
                osnet_to_mamba = F.interpolate(
                    osnet_to_mamba,
                    size=mamba_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            osnet_to_mamba = self._gate_delta(mamba_map, osnet_to_mamba, self.mamba_gate)
            new_mamba_map = mamba_map + self.fusion_scale_mamba.to(mamba_map.dtype) * osnet_to_mamba

        if self.direction in ('bidirectional', 'mamba_to_osnet'):
            mamba_to_osnet = self.osnet_from_mamba(mamba_map)
            if mamba_to_osnet.shape[-2:] != osnet_map.shape[-2:]:
                mamba_to_osnet = F.interpolate(
                    mamba_to_osnet,
                    size=osnet_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            mamba_to_osnet = self._gate_delta(osnet_map, mamba_to_osnet, self.osnet_gate)
            new_osnet_map = osnet_map + self.fusion_scale_osnet.to(osnet_map.dtype) * mamba_to_osnet

        return new_mamba_map, new_osnet_map


class MambaOSNetFusion(nn.Module):
    """MambaVision + OSNet fusion."""

    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super().__init__()
        if cfg.INPUT.PAM.ENABLED:
            raise ValueError('OSNET_FUSION first version expects INPUT.PAM.ENABLED=False')
        if getattr(cfg.MODEL.MAMBAVISION, 'USE_SFM', False):
            raise ValueError('OSNET_FUSION and SFM must be evaluated separately')

        fusion_cfg = cfg.MODEL.OSNET_FUSION
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.cosine_scale = cfg.SOLVER.COSINE_SCALE
        self.cosine_margin = cfg.SOLVER.COSINE_MARGIN
        self.fusion_type = str(getattr(fusion_cfg, 'FUSION_TYPE', 'descriptor')).lower()
        self.fusion_norm = str(getattr(fusion_cfg, 'FUSION_NORM', 'none')).lower()
        self.fusion_beta = float(getattr(fusion_cfg, 'FUSION_BETA', 1.0))
        if self.fusion_type not in ('descriptor', 'stage_fcu', 'fdmf'):
            raise ValueError("MODEL.OSNET_FUSION.FUSION_TYPE must be 'descriptor', 'stage_fcu', or 'fdmf'")
        if self.fusion_norm not in ('none', 'branch', 'weighted_branch'):
            raise ValueError("MODEL.OSNET_FUSION.FUSION_NORM must be one of: none, branch, weighted_branch")
        if self.fusion_type == 'fdmf' and self.fusion_norm != 'none':
            raise ValueError("MODEL.OSNET_FUSION.FUSION_NORM must be 'none' for fdmf")
        if self.fusion_beta < 0:
            raise ValueError('MODEL.OSNET_FUSION.FUSION_BETA must be non-negative')
        self.fdmf_fused_form = str(getattr(fusion_cfg, 'FDMF_FUSED_FORM', 'raw_fdmf')).lower()
        if self.fdmf_fused_form not in ('mamba_fdmf', 'raw_fdmf', 'fdmf_only'):
            raise ValueError("MODEL.OSNET_FUSION.FDMF_FUSED_FORM must be 'mamba_fdmf', 'raw_fdmf', or 'fdmf_only'")

        self.mamba = build_transformer(num_classes, camera_num, view_num, cfg, factory)
        self.mamba_dim = self.mamba.in_planes

        osnet_type = str(getattr(fusion_cfg, 'OSNET_TYPE', 'osnet_x1_0'))
        if osnet_type not in _factory_osnet:
            raise ValueError('Unknown OSNet type: {}'.format(osnet_type))
        self.osnet = _factory_osnet[osnet_type](
            num_classes=num_classes,
            pretrained=False,
            loss='triplet',
        )
        self.osnet_dim = self.osnet.feature_dim
        self.osnet_map_dim = self.osnet.conv5.conv.out_channels
        self.osnet_stage2_dim = self.osnet.conv3[0].conv3.conv.out_channels
        self.osnet_stage3_dim = self.osnet.conv4[0].conv3.conv.out_channels

        osnet_pretrain = str(getattr(fusion_cfg, 'PRETRAIN_PATH', '')).strip()
        if osnet_pretrain:
            if not os.path.exists(osnet_pretrain):
                raise FileNotFoundError('OSNet pretrained weight not found: {}'.format(osnet_pretrain))
            self.osnet.load_param(osnet_pretrain)
        else:
            print('[Model] OSNet fusion branch is randomly initialized; set MODEL.OSNET_FUSION.PRETRAIN_PATH for ImageNet weights')

        if bool(getattr(fusion_cfg, 'FREEZE_OSNET', False)):
            for param in self.osnet.parameters():
                param.requires_grad = False
            print('[Model] OSNet fusion branch frozen')

        self.osnet_bottleneck = nn.BatchNorm1d(self.osnet_dim)
        self.osnet_bottleneck.bias.requires_grad_(False)
        self.osnet_bottleneck.apply(weights_init_kaiming)
        self.osnet_classifier = self._make_classifier(self.osnet_dim)

        self.fdmf_refiner = None
        if self.fusion_type == 'fdmf':
            self.fdmf_refiner = SameScaleFrequencyMambaFusion(
                dim=self.mamba_dim,
                osnet_dim=self.osnet_map_dim,
                compressed_channels=int(getattr(fusion_cfg, 'FDMF_COMPRESSED_CHANNELS', 64)),
                lowpass_kernel=int(getattr(fusion_cfg, 'FDMF_LOWPASS_KERNEL', 5)),
                highpass_kernel=int(getattr(fusion_cfg, 'FDMF_HIGHPASS_KERNEL', 3)),
                use_hamming=bool(getattr(fusion_cfg, 'FDMF_HAMMING_WINDOW', True)),
                filter_type=str(getattr(fusion_cfg, 'FDMF_FILTER_TYPE', 'dynamic')),
                mamba_depth=int(getattr(fusion_cfg, 'FDMF_MAMBA_DEPTH', 1)),
                mamba_d_state=int(getattr(fusion_cfg, 'FDMF_MAMBA_D_STATE', 8)),
                mamba_d_conv=int(getattr(fusion_cfg, 'FDMF_MAMBA_D_CONV', 3)),
                mamba_init_scale=float(getattr(fusion_cfg, 'FDMF_MAMBA_INIT_SCALE', 0.1)),
                mamba_bidirectional=bool(getattr(fusion_cfg, 'FDMF_MAMBA_BIDIRECTIONAL', True)),
                mlp_ratio=float(getattr(fusion_cfg, 'FDMF_MLP_RATIO', 2.0)),
                msef_enabled=bool(getattr(fusion_cfg, 'FDMF_MSEF_ENABLED', True)),
                msef_reduction_ratio=int(getattr(fusion_cfg, 'FDMF_MSEF_REDUCTION_RATIO', 16)),
                msef_res_scale_enabled=bool(getattr(fusion_cfg, 'FDMF_MSEF_RES_SCALE_ENABLED', False)),
                msef_res_scale_init=float(getattr(fusion_cfg, 'FDMF_MSEF_RES_SCALE_INIT', 0.1)),
            )
            if self.fdmf_fused_form == 'raw_fdmf':
                self.fusion_dim = self.mamba_dim + self.osnet_dim + self.mamba_dim
            elif self.fdmf_fused_form == 'mamba_fdmf':
                self.fusion_dim = self.mamba_dim * 2
            else:
                self.fusion_dim = self.mamba_dim
        else:
            self.fusion_dim = self.mamba_dim + self.osnet_dim
        self.fusion_bottleneck = nn.BatchNorm1d(self.fusion_dim)
        self.fusion_bottleneck.bias.requires_grad_(False)
        self.fusion_bottleneck.apply(weights_init_kaiming)
        self.fusion_classifier = self._make_classifier(self.fusion_dim)

        self.use_stage_fcu_exchange = self.fusion_type == 'stage_fcu' or bool(getattr(fusion_cfg, 'FCU_ENABLED', False))
        if self.use_stage_fcu_exchange:
            if not hasattr(self.mamba.base, 'levels') or len(self.mamba.base.levels) < 4:
                raise ValueError('stage_fcu requires a MambaVision backbone with four stages')
            self.fcu_stages = [int(stage) for stage in getattr(fusion_cfg, 'FCU_STAGES', [2, 3])]
            invalid_fcu_stages = sorted(set(self.fcu_stages) - {2, 3})
            if invalid_fcu_stages:
                raise ValueError('MODEL.OSNET_FUSION.FCU_STAGES only supports stages 2 and 3')
            if not self.fcu_stages:
                raise ValueError('MODEL.OSNET_FUSION.FCU_STAGES must contain at least one stage for stage_fcu')
            mamba_stage2_dim = self._mamba_stage_out_dim(self.mamba.base.levels[1])
            mamba_stage3_dim = self._mamba_stage_out_dim(self.mamba.base.levels[2])
            fcu_init_scale = float(getattr(fusion_cfg, 'FCU_INIT_SCALE', 0.1))
            fcu_gate_type = str(getattr(fusion_cfg, 'FCU_GATE_TYPE', 'none')).lower()
            fcu_gate_reduction = int(getattr(fusion_cfg, 'FCU_GATE_REDUCTION', 16))
            fcu_gate_init_bias = float(getattr(fusion_cfg, 'FCU_GATE_INIT_BIAS', 0.0))
            self.fcu_direction = str(getattr(fusion_cfg, 'FCU_DIRECTION', 'bidirectional')).lower()
            self.fcu_stage2_direction = str(getattr(fusion_cfg, 'FCU_STAGE2_DIRECTION', '')).lower()
            self.fcu_stage3_direction = str(getattr(fusion_cfg, 'FCU_STAGE3_DIRECTION', '')).lower()
            if not self.fcu_stage2_direction:
                self.fcu_stage2_direction = self.fcu_direction
            if not self.fcu_stage3_direction:
                self.fcu_stage3_direction = self.fcu_direction
            if 2 in self.fcu_stages:
                self.stage2_fcu = StageFCU(
                    mamba_stage2_dim,
                    self.osnet_stage2_dim,
                    init_scale=fcu_init_scale,
                    direction=self.fcu_stage2_direction,
                    gate_type=fcu_gate_type,
                    gate_reduction=fcu_gate_reduction,
                    gate_init_bias=fcu_gate_init_bias,
                )
            if 3 in self.fcu_stages:
                self.stage3_fcu = StageFCU(
                    mamba_stage3_dim,
                    self.osnet_stage3_dim,
                    init_scale=fcu_init_scale,
                    direction=self.fcu_stage3_direction,
                    gate_type=fcu_gate_type,
                    gate_reduction=fcu_gate_reduction,
                    gate_init_bias=fcu_gate_init_bias,
                )

        print(
            '[Model] Mamba-OSNet fusion enabled: type={}, mamba_dim={}, osnet_type={}, osnet_dim={}, fusion_dim={}, fusion_norm={}, beta={:.2f}'.format(
                self.fusion_type,
                self.mamba_dim,
                osnet_type,
                self.osnet_dim,
                self.fusion_dim,
                self.fusion_norm,
                self.fusion_beta,
            )
        )
        if self.fusion_type == 'fdmf':
            print(
                '[Model] FDMF enabled: osnet_map_dim={}, form={}, filter={}, bidirectional={}, msef={}, msef_res_scale={}'.format(
                    self.osnet_map_dim,
                    self.fdmf_fused_form,
                    self.fdmf_refiner.filter_type,
                    bool(getattr(fusion_cfg, 'FDMF_MAMBA_BIDIRECTIONAL', True)),
                    bool(getattr(fusion_cfg, 'FDMF_MSEF_ENABLED', True)),
                    bool(getattr(fusion_cfg, 'FDMF_MSEF_RES_SCALE_ENABLED', False)),
                )
            )
        if self.use_stage_fcu_exchange:
            print(
                '[Model] Stage-FCU exchange enabled at stages={}, direction={}, stage2_direction={}, stage3_direction={}, init_scale={:.3f}, gate_type={}, gate_reduction={}, gate_init_bias={:.2f}, stage2_dim={}/{}, stage3_dim={}/{}'.format(
                    self.fcu_stages,
                    self.fcu_direction,
                    self.fcu_stage2_direction,
                    self.fcu_stage3_direction,
                    fcu_init_scale,
                    fcu_gate_type,
                    fcu_gate_reduction,
                    fcu_gate_init_bias,
                    mamba_stage2_dim,
                    self.osnet_stage2_dim,
                    mamba_stage3_dim,
                    self.osnet_stage3_dim,
                )
            )
    def _make_classifier(self, in_planes):
        if self.ID_LOSS_TYPE == 'arcface':
            return Arcface(in_planes, self.num_classes, s=self.cosine_scale, m=self.cosine_margin)
        if self.ID_LOSS_TYPE == 'cosface':
            return Cosface(in_planes, self.num_classes, s=self.cosine_scale, m=self.cosine_margin)
        if self.ID_LOSS_TYPE == 'amsoftmax':
            return AMSoftmax(in_planes, self.num_classes, s=self.cosine_scale, m=self.cosine_margin)
        if self.ID_LOSS_TYPE == 'circle':
            return CircleLoss(in_planes, self.num_classes, s=self.cosine_scale, m=self.cosine_margin)
        classifier = nn.Linear(in_planes, self.num_classes, bias=False)
        classifier.apply(weights_init_classifier)
        return classifier

    @staticmethod
    def _mamba_stage_out_dim(level):
        if getattr(level, 'downsample', None) is not None:
            return level.downsample.reduction[0].out_channels
        block = level.blocks[0]
        if hasattr(block, 'norm1') and hasattr(block.norm1, 'normalized_shape'):
            return block.norm1.normalized_shape[0]
        if hasattr(block, 'conv1'):
            return block.conv1.out_channels
        raise ValueError('Unable to infer MambaVision stage output channels')

    def _classify(self, classifier, feat, label):
        if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
            return classifier(feat, label)
        return classifier(feat)

    def _normalize_feature(self, x):
        return F.normalize(x.float(), p=2, dim=1).to(x.dtype)

    def _make_descriptor_fused_feat(self, mamba_feat, osnet_feat):
        if self.fusion_norm == 'none':
            return torch.cat([mamba_feat, osnet_feat], dim=1)

        mamba_feat_n = self._normalize_feature(mamba_feat)
        osnet_feat_n = self._normalize_feature(osnet_feat)
        if self.fusion_norm == 'weighted_branch':
            osnet_feat_n = self.fusion_beta * osnet_feat_n
        return torch.cat([mamba_feat_n, osnet_feat_n], dim=1)

    def _make_fdmf_fused_feat(self, mamba_feat, osnet_feat, mamba_map, osnet_map, return_fdmf_feat=False):
        fdmf_map = self.fdmf_refiner(mamba_map, osnet_map)
        fdmf_feat = self.mamba._pool_feature_map(fdmf_map)
        if self.fdmf_fused_form == 'raw_fdmf':
            fused_feat = torch.cat([mamba_feat, osnet_feat, fdmf_feat], dim=1)
        elif self.fdmf_fused_form == 'mamba_fdmf':
            fused_feat = torch.cat([mamba_feat, fdmf_feat], dim=1)
        else:
            fused_feat = fdmf_feat
        if return_fdmf_feat:
            return fused_feat, fdmf_feat
        return fused_feat

    def _forward_mamba_branch(self, x, label=None, cam_label=None, view_label=None, return_map=False):
        output = self.mamba.base(x, cam_label=cam_label, view_label=view_label)
        if isinstance(output, dict):
            raise ValueError('OSNET_FUSION expects a single MambaVision feature map')
        feature_map = output
        global_feat = self.mamba._pool_feature_map(feature_map)
        feat_bn = self.mamba.bottleneck(global_feat)
        if self.training:
            score = self.mamba._classify(self.mamba.classifier, feat_bn, label)
            if return_map:
                return score, global_feat, feat_bn, feature_map
            return score, global_feat, feat_bn
        if return_map:
            return global_feat, feat_bn, feature_map
        return global_feat, feat_bn

    def _forward_osnet_branch(self, x, label=None, return_map=False):
        feature_map = self.osnet.featuremaps(x)
        global_feat = self.osnet.global_avgpool(feature_map)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.osnet.fc is not None:
            global_feat = self.osnet.fc(global_feat)
        feat_bn = self.osnet_bottleneck(global_feat)
        if self.training:
            score = self._classify(self.osnet_classifier, feat_bn, label)
            if return_map:
                return score, global_feat, feat_bn, feature_map
            return score, global_feat, feat_bn
        if return_map:
            return global_feat, feat_bn, feature_map
        return global_feat, feat_bn

    def _mamba_patch_embed_with_sie(self, x, cam_label=None, view_label=None):
        base = self.mamba.base
        x = base.patch_embed(x)
        if base.sie_embed is None or cam_label is None:
            return x

        max_idx = base.sie_embed.shape[0] - 1
        if base.cam_num > 1 and base.view_num > 1 and view_label is not None:
            idx = cam_label * base.view_num + view_label
            idx = idx.clamp(0, max_idx)
            sie = base.sie_xishu * base.sie_embed[idx]
        elif base.cam_num > 1:
            idx = cam_label.clamp(0, max_idx)
            sie = base.sie_xishu * base.sie_embed[idx]
        elif base.view_num > 1 and view_label is not None:
            idx = view_label.clamp(0, max_idx)
            sie = base.sie_xishu * base.sie_embed[idx]
        else:
            sie = 0
        return x + sie

    def _forward_stage_fcu_maps(self, x, cam_label=None, view_label=None):
        base = self.mamba.base

        mamba_map = self._mamba_patch_embed_with_sie(x, cam_label=cam_label, view_label=view_label)
        osnet_map = self.osnet.conv1(x)
        osnet_map = self.osnet.maxpool(osnet_map)
        osnet_map = self.osnet.conv2(osnet_map)

        mamba_map = base.levels[0](mamba_map)
        mamba_map = base.levels[1](mamba_map)
        osnet_map = self.osnet.conv3(osnet_map)
        if 2 in self.fcu_stages:
            mamba_map, osnet_map = self.stage2_fcu(mamba_map, osnet_map)

        mamba_map = base.levels[2](mamba_map)
        osnet_map = self.osnet.conv4(osnet_map)
        if 3 in self.fcu_stages:
            mamba_map, osnet_map = self.stage3_fcu(mamba_map, osnet_map)

        mamba_map = base.main_proj(mamba_map)
        mamba_map = base.levels[3](mamba_map)
        osnet_map = self.osnet.conv5(osnet_map)
        return mamba_map, osnet_map

    def _forward_stage_fcu_branches(self, x, label=None, cam_label=None, view_label=None, return_map=False):
        mamba_map, osnet_map = self._forward_stage_fcu_maps(
            x,
            cam_label=cam_label,
            view_label=view_label,
        )

        mamba_feat = self.mamba._pool_feature_map(mamba_map)
        mamba_bn = self.mamba.bottleneck(mamba_feat)

        osnet_feat = self.osnet.global_avgpool(osnet_map)
        osnet_feat = osnet_feat.view(osnet_feat.size(0), -1)
        if self.osnet.fc is not None:
            osnet_feat = self.osnet.fc(osnet_feat)
        osnet_bn = self.osnet_bottleneck(osnet_feat)

        if self.training:
            mamba_score = self.mamba._classify(self.mamba.classifier, mamba_bn, label)
            osnet_score = self._classify(self.osnet_classifier, osnet_bn, label)
            if return_map:
                return mamba_score, mamba_feat, mamba_bn, osnet_score, osnet_feat, osnet_bn, mamba_map, osnet_map
            return mamba_score, mamba_feat, mamba_bn, osnet_score, osnet_feat, osnet_bn
        if return_map:
            return mamba_feat, mamba_bn, osnet_feat, osnet_bn, mamba_map, osnet_map
        return mamba_feat, mamba_bn, osnet_feat, osnet_bn

    def forward(self, x, label=None, cam_label=None, view_label=None):
        use_fdmf = self.fusion_type == 'fdmf'
        if self.training:
            if self.use_stage_fcu_exchange:
                stage_out = self._forward_stage_fcu_branches(
                    x,
                    label=label,
                    cam_label=cam_label,
                    view_label=view_label,
                    return_map=use_fdmf,
                )
                if use_fdmf:
                    (
                        mamba_score,
                        mamba_feat,
                        mamba_bn,
                        osnet_score,
                        osnet_feat,
                        osnet_bn,
                        mamba_map,
                        osnet_map,
                    ) = stage_out
                    fused_feat = self._make_fdmf_fused_feat(mamba_feat, osnet_feat, mamba_map, osnet_map)
                else:
                    (
                        mamba_score,
                        mamba_feat,
                        mamba_bn,
                        osnet_score,
                        osnet_feat,
                        osnet_bn,
                    ) = stage_out
            else:
                mamba_out = self._forward_mamba_branch(
                    x,
                    label=label,
                    cam_label=cam_label,
                    view_label=view_label,
                    return_map=use_fdmf,
                )
                osnet_out = self._forward_osnet_branch(x, label=label, return_map=use_fdmf)
                if use_fdmf:
                    mamba_score, mamba_feat, mamba_bn, mamba_map = mamba_out
                    osnet_score, osnet_feat, osnet_bn, osnet_map = osnet_out
                    fused_feat = self._make_fdmf_fused_feat(mamba_feat, osnet_feat, mamba_map, osnet_map)
                else:
                    mamba_score, mamba_feat, mamba_bn = mamba_out
                    osnet_score, osnet_feat, osnet_bn = osnet_out
                    fused_feat = self._make_descriptor_fused_feat(mamba_feat, osnet_feat)
            if self.use_stage_fcu_exchange and not use_fdmf:
                fused_feat = self._make_descriptor_fused_feat(mamba_feat, osnet_feat)
            fused_bn = self.fusion_bottleneck(fused_feat)
            fused_score = self._classify(self.fusion_classifier, fused_bn, label)
            return [mamba_score, osnet_score, fused_score], [mamba_feat, osnet_feat, fused_feat]

        if self.use_stage_fcu_exchange:
            stage_out = self._forward_stage_fcu_branches(
                x,
                cam_label=cam_label,
                view_label=view_label,
                return_map=use_fdmf,
            )
            if use_fdmf:
                mamba_feat, mamba_bn, osnet_feat, osnet_bn, mamba_map, osnet_map = stage_out
                fused_feat, fdmf_feat = self._make_fdmf_fused_feat(
                    mamba_feat,
                    osnet_feat,
                    mamba_map,
                    osnet_map,
                    return_fdmf_feat=True,
                )
            else:
                mamba_feat, mamba_bn, osnet_feat, osnet_bn = stage_out
                fused_feat = self._make_descriptor_fused_feat(mamba_feat, osnet_feat)
        else:
            mamba_out = self._forward_mamba_branch(
                x,
                cam_label=cam_label,
                view_label=view_label,
                return_map=use_fdmf,
            )
            osnet_out = self._forward_osnet_branch(x, return_map=use_fdmf)
            if use_fdmf:
                mamba_feat, mamba_bn, mamba_map = mamba_out
                osnet_feat, osnet_bn, osnet_map = osnet_out
                fused_feat, fdmf_feat = self._make_fdmf_fused_feat(
                    mamba_feat,
                    osnet_feat,
                    mamba_map,
                    osnet_map,
                    return_fdmf_feat=True,
                )
            else:
                mamba_feat, mamba_bn = mamba_out
                osnet_feat, osnet_bn = osnet_out
                fused_feat = self._make_descriptor_fused_feat(mamba_feat, osnet_feat)
        if self.use_stage_fcu_exchange and not use_fdmf:
            fused_feat = self._make_descriptor_fused_feat(mamba_feat, osnet_feat)
        fused_bn = self.fusion_bottleneck(fused_feat)

        if self.neck_feat == 'after':
            mamba_out = mamba_bn
            osnet_out = osnet_bn
            fused_out = fused_bn
        else:
            mamba_out = mamba_feat
            osnet_out = osnet_feat
            fused_out = fused_feat

        output = {
            'backbone': mamba_out,
            'osnet': osnet_out,
        }
        if use_fdmf:
            output['raw_concat'] = torch.cat([mamba_out, osnet_out], dim=1)
            output['fdmf'] = fused_out
            output['mamba_fdmf_osnet'] = torch.cat([mamba_out, fdmf_feat, osnet_out], dim=1)
        else:
            output['concat'] = fused_out
        return output

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        if isinstance(param_dict, dict) and 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        own_state = self.state_dict()
        matched, skipped = 0, []
        for key, value in param_dict.items():
            key = key.replace('module.', '')
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key].copy_(value)
                matched += 1
            else:
                skipped.append(key)
        print('Loading pretrained model from {} (matched={}, skipped={})'.format(
            trained_path,
            matched,
            len(skipped),
        ))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'mambavision_tiny_reid': mambavision_tiny_reid,
    'mambavision_small_reid': mambavision_small_reid,
    'mambavision_base_reid': mambavision_base_reid,
    'mambavision_tiny_TransReID': mambavision_tiny_reid,
    'mambavision_small_TransReID': mambavision_small_reid,
    'mambavision_base_TransReID': mambavision_base_reid,
}


def make_model(cfg, num_class, camera_num, view_num):
    model = None
    osnet_fusion_enabled = bool(getattr(getattr(cfg.MODEL, 'OSNET_FUSION', None), 'ENABLED', False))
    if osnet_fusion_enabled:
        model = MambaOSNetFusion(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========building Mamba-OSNet fusion===========')
    elif cfg.MODEL.NAME == 'transformer':
        model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    
    if model is None:
        raise RuntimeError("Model was not initialized correctly in make_model!")
        
    return model

