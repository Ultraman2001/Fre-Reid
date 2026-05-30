import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .backbones.mambavision.mamba_vision_reid import mambavision_tiny_reid, mambavision_small_reid, mambavision_base_reid
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


METRIC_CLASSIFIERS = ('arcface', 'cosface', 'amsoftmax', 'circle')


def create_classifier_head(id_loss_type, in_planes, num_classes, cfg, prefix=''):
    if id_loss_type == 'arcface':
        if prefix:
            print('[{}] using arcface with s:{}, m: {}'.format(prefix, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        return Arcface(in_planes, num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
    if id_loss_type == 'cosface':
        if prefix:
            print('[{}] using cosface with s:{}, m: {}'.format(prefix, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        return Cosface(in_planes, num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
    if id_loss_type == 'amsoftmax':
        if prefix:
            print('[{}] using amsoftmax with s:{}, m: {}'.format(prefix, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        return AMSoftmax(in_planes, num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
    if id_loss_type == 'circle':
        if prefix:
            print('[{}] using circle with s:{}, m: {}'.format(prefix, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        return CircleLoss(in_planes, num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)

    classifier = nn.Linear(in_planes, num_classes, bias=False)
    classifier.apply(weights_init_classifier)
    return classifier


def classifier_forward(classifier, id_loss_type, feat, label=None):
    if id_loss_type in METRIC_CLASSIFIERS:
        return classifier(feat, label)
    return classifier(feat)


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
        self.use_fd = False
        
        self.pooling = None

        if self.is_mambavision:
            # 获取配置
            global_stages = list(getattr(cfg.MODEL.MAMBAVISION, 'GLOBAL_STAGES', []))
            sasf_stages = list(getattr(cfg.MODEL.MAMBAVISION, 'SASF_STAGES', []))
            
            # SFM 配置
            self.use_sfm = getattr(cfg.MODEL.MAMBAVISION, 'USE_SFM', False)
            self.use_fd = getattr(cfg.MODEL.MAMBAVISION, 'USE_FD', False)
            if self.use_sfm and self.use_fd:
                raise ValueError('USE_SFM and USE_FD should not be enabled at the same time.')
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
                use_fd=self.use_fd,
            )
            # Get feature dimension from MambaVision
            self.in_planes = self.base.num_features
            pooling_type = getattr(cfg.MODEL, 'POOLING_TYPE', 'gem')
            self.pooling = create_pooling(pooling_type)
            print(f'[Model] Using pooling type: {pooling_type}')
            if self.use_fd:
                fd_pooling_type = getattr(cfg.MODEL.MAMBAVISION, 'FD_POOLING_TYPE', pooling_type)
                self.fd_pooling = create_pooling(fd_pooling_type)
                print(f'[FD] Using frequency pooling type: {fd_pooling_type}')
            
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

        if self.is_mambavision and self.use_fd:
            self.fd_in_planes = self.in_planes * 2

            self.fd_bottleneck = nn.BatchNorm1d(self.fd_in_planes)
            self.fd_bottleneck.bias.requires_grad_(False)
            self.fd_bottleneck.apply(weights_init_kaiming)

            self.fd_spa_bottleneck = nn.BatchNorm1d(self.in_planes)
            self.fd_spa_bottleneck.bias.requires_grad_(False)
            self.fd_spa_bottleneck.apply(weights_init_kaiming)

            self.fd_freq_bottleneck = nn.BatchNorm1d(self.in_planes)
            self.fd_freq_bottleneck.bias.requires_grad_(False)
            self.fd_freq_bottleneck.apply(weights_init_kaiming)

            self.fd_classifier = create_classifier_head(
                self.ID_LOSS_TYPE, self.fd_in_planes, self.num_classes, cfg, prefix='FD final'
            )
            self.fd_spa_classifier = create_classifier_head(
                self.ID_LOSS_TYPE, self.in_planes, self.num_classes, cfg, prefix='FD spatial'
            )
            self.fd_freq_classifier = create_classifier_head(
                self.ID_LOSS_TYPE, self.in_planes, self.num_classes, cfg, prefix='FD frequency'
            )
            print(f'[FD] Initialized Stage2 DWT dual-branch heads: final_dim={self.fd_in_planes}')
        
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

    def _pool_feature_map(self, feature_map, pooling_layer=None):
        if pooling_layer is None:
            pooling_layer = self.pooling
        if pooling_layer is None:
            pooled = nn.functional.adaptive_avg_pool2d(feature_map, 1)
        else:
            pooled = pooling_layer(feature_map)
        return pooled.flatten(1)

    def _forward_fd(self, spa_map, freq_map, label=None):
        spa_feat = self._pool_feature_map(spa_map, self.pooling)
        freq_feat = self._pool_feature_map(freq_map, self.fd_pooling)
        final_feat = torch.cat([spa_feat, freq_feat], dim=1)

        final_bn = self.fd_bottleneck(final_feat)
        spa_bn = self.fd_spa_bottleneck(spa_feat)
        freq_bn = self.fd_freq_bottleneck(freq_feat)

        if self.training:
            final_score = classifier_forward(self.fd_classifier, self.ID_LOSS_TYPE, final_bn, label)
            spa_score = classifier_forward(self.fd_spa_classifier, self.ID_LOSS_TYPE, spa_bn, label)
            freq_score = classifier_forward(self.fd_freq_classifier, self.ID_LOSS_TYPE, freq_bn, label)
            return [final_score, spa_score, freq_score], [final_feat, spa_feat, freq_feat]

        return final_bn if self.neck_feat == 'after' else final_feat

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.is_mambavision or self.is_puzzle:
            # Get backbone output
            output = self.base(x, cam_label=cam_label, view_label=view_label)
            
            # Check if SFM mode (output is dict) or normal mode (output is tensor)
            if isinstance(output, dict):
                if 'freq_map' in output:
                    return self._forward_fd(
                        output['backbone_map'],
                        output['freq_map'],
                        label=label,
                    )

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
                
                # Apply pooling
                global_feat = self._pool_feature_map(feature_map)
                
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

