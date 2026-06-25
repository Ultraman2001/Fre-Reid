import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .backbones.mambavision.mamba_vision_reid import (
    MambaVisionMixer,
    mambavision_tiny_reid,
    mambavision_small_reid,
    mambavision_base_reid,
)
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
    if cfg.MODEL.NAME == 'transformer':
        model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    
    if model is None:
        raise RuntimeError("Model was not initialized correctly in make_model!")
        
    return model

