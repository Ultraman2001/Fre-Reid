# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ratr_loss import RATRLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    
    # RATR Loss 初始化
    ratr_fn = None
    if getattr(cfg.SOLVER, 'RATR_ENABLED', False):
        pk = cfg.DATALOADER.NUM_INSTANCE  # K 值
        p = cfg.SOLVER.IMS_PER_BATCH // pk  # P 值
        tau = getattr(cfg.SOLVER, 'RATR_TAU', 0.1)
        ratr_fn = RATRLoss(num_branches=2, num_classes=p, samples_per_class=pk, tau=tau)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                # 支持多分支模式 (PMS 或 SFM)
                if isinstance(score, list):
                    use_pam = cfg.INPUT.PAM.ENABLED
                    use_sfm = getattr(cfg.MODEL.MAMBAVISION, 'USE_SFM', False)
                    osnet_fusion_cfg = getattr(cfg.MODEL, 'OSNET_FUSION', None)
                    use_osnet_fusion = bool(getattr(osnet_fusion_cfg, 'ENABLED', False))

                    if use_pam:
                        local_cfg = getattr(cfg.MODEL, 'LOCAL_STRIPE', None)
                        local_enabled = bool(getattr(local_cfg, 'ENABLED', False))
                        local_num_stripes = int(getattr(local_cfg, 'NUM_STRIPES', 4))
                        expected_len = 3 + local_num_stripes if local_enabled else 3
                        if len(score) != expected_len or len(feat) != expected_len:
                            raise ValueError(
                                'PAM loss expects BA, CA, EA plus configured local auxiliary branches'
                            )

                        pam_weight = cfg.SOLVER.PAM_AUGMENTED_LOSS_WEIGHT
                        branch_names = ('ba', 'ca', 'ea')
                        id_losses = []
                        tri_losses = []

                        for branch_score, branch_feat in zip(score[:3], feat[:3]):
                            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                                id_losses.append(xent(branch_score.float(), target))
                            else:
                                id_losses.append(F.cross_entropy(branch_score.float(), target))
                            tri_losses.append(triplet(branch_feat.float(), target)[0])

                        pam_denominator = 1.0 + 2.0 * pam_weight
                        ID_LOSS = (id_losses[0] + pam_weight * (id_losses[1] + id_losses[2])) / \
                                  pam_denominator
                        TRI_LOSS = (tri_losses[0] + pam_weight * (tri_losses[1] + tri_losses[2])) / \
                                   pam_denominator

                        loss_detail = {'pam_weight': pam_weight}
                        for name, id_loss, tri_loss in zip(branch_names, id_losses, tri_losses):
                            loss_detail[f'id_{name}'] = id_loss.item()
                            loss_detail[f'tri_{name}'] = tri_loss.item()

                        aux_start = 3
                        if local_enabled:
                            local_weight = float(getattr(local_cfg, 'LOSS_WEIGHT', 0.2))
                            local_use_triplet = bool(getattr(local_cfg, 'USE_TRIPLET', True))
                            local_id_losses = []
                            local_tri_losses = []

                            aux_end = aux_start + local_num_stripes
                            for local_score, local_feat in zip(score[aux_start:aux_end], feat[aux_start:aux_end]):
                                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                                    local_id_losses.append(xent(local_score.float(), target))
                                else:
                                    local_id_losses.append(F.cross_entropy(local_score.float(), target))
                                if local_use_triplet:
                                    local_tri_losses.append(triplet(local_feat.float(), target)[0])

                            LOCAL_ID_LOSS = sum(local_id_losses) / len(local_id_losses)
                            ID_LOSS = ID_LOSS + local_weight * LOCAL_ID_LOSS

                            if local_use_triplet:
                                LOCAL_TRI_LOSS = sum(local_tri_losses) / len(local_tri_losses)
                                TRI_LOSS = TRI_LOSS + local_weight * LOCAL_TRI_LOSS
                                loss_detail['tri_local'] = LOCAL_TRI_LOSS.item()
                            else:
                                loss_detail['tri_local'] = 0.0

                            loss_detail['local_weight'] = local_weight
                            loss_detail['id_local'] = LOCAL_ID_LOSS.item()

                        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                        return total_loss, loss_detail
                    
                    if use_osnet_fusion:
                        if len(score) != 3 or len(feat) != 3:
                            raise ValueError('OSNET_FUSION loss expects Mamba, OSNet and concat branches')

                        branch_names = ('mamba', 'osnet', 'concat')
                        branch_weights = (
                            1.0,
                            float(getattr(osnet_fusion_cfg, 'OSNET_LOSS_WEIGHT', 0.5)),
                            float(getattr(osnet_fusion_cfg, 'FUSED_LOSS_WEIGHT', 1.0)),
                        )
                        weight_sum = sum(branch_weights)
                        if weight_sum <= 0:
                            raise ValueError('OSNET_FUSION branch weights must sum to a positive value')

                        id_losses = []
                        tri_losses = []
                        for branch_score, branch_feat in zip(score, feat):
                            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                                id_losses.append(xent(branch_score.float(), target))
                            else:
                                id_losses.append(F.cross_entropy(branch_score.float(), target))
                            tri_losses.append(triplet(branch_feat.float(), target)[0])

                        ID_LOSS = sum(w * l for w, l in zip(branch_weights, id_losses)) / weight_sum
                        TRI_LOSS = sum(w * l for w, l in zip(branch_weights, tri_losses)) / weight_sum

                        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                        loss_detail = {
                            'fusion_mode': 'osnet',
                            'w_mamba': branch_weights[0],
                            'w_osnet': branch_weights[1],
                            'w_concat': branch_weights[2],
                        }
                        for name, id_loss, tri_loss in zip(branch_names, id_losses, tri_losses):
                            loss_detail[f'id_{name}'] = id_loss.item()
                            loss_detail[f'tri_{name}'] = tri_loss.item()

                        return total_loss, loss_detail

                    if use_sfm and len(score) >= 2:
                        # SFM 模式：HAT风格多级聚合损失 (归一化版本)
                        sfm_lambda = getattr(cfg.SOLVER, 'SFM_LAMBDA', 1.0)
                        sfm_lambda_aux = getattr(cfg.SOLVER, 'SFM_LAMBDA_AUX', 0.5)
                        
                        # 1. Backbone 分支 (权重 = 1.0)
                        if cfg.MODEL.IF_LABELSMOOTH == 'on':
                            id_b = xent(score[0].float(), target)
                        else:
                            id_b = F.cross_entropy(score[0].float(), target)
                        tri_b = triplet(feat[0].float(), target)[0]
                        
                        # 2. Fused 分支 (循环处理所有聚合层级)
                        id_f_list = []
                        tri_f_list = []
                        num_fused = len(score) - 1
                        
                        total_id_loss = id_b
                        total_tri_loss = tri_b
                        total_weight = 1.0
                        
                        loss_detail = {
                            'id_backbone': id_b.item(),
                            'tri_backbone': tri_b.item(),
                            'sfm_lambda': sfm_lambda,
                        }
                        
                        for i in range(1, len(score)):
                            # 最后一级使用 SFM_LAMBDA，中间级使用 SFM_LAMBDA_AUX
                            is_last = (i == len(score) - 1)
                            w = sfm_lambda if is_last else sfm_lambda_aux
                            
                            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                                id_fi = xent(score[i].float(), target)
                            else:
                                id_fi = F.cross_entropy(score[i].float(), target)
                            tri_fi = triplet(feat[i].float(), target)[0]
                            
                            total_id_loss = total_id_loss + w * id_fi
                            total_tri_loss = total_tri_loss + w * tri_fi
                            total_weight += w
                            
                            # 记录详情
                            loss_detail[f'id_fused_{i}'] = id_fi.item()
                            loss_detail[f'tri_fused_{i}'] = tri_fi.item()
                            
                        # 兼容性补充：将最后一级存为 id_fused 和 tri_fused
                        num_fused = len(score) - 1
                        if num_fused > 0:
                            loss_detail['id_fused'] = loss_detail[f'id_fused_{num_fused}']
                            loss_detail['tri_fused'] = loss_detail[f'tri_fused_{num_fused}']
                        
                        # 取消归一化，直接使用加权和
                        ID_LOSS = total_id_loss
                        TRI_LOSS = total_tri_loss
                        
                        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                        
                        # ===== RATR 损失 =====
                        nonlocal ratr_fn
                        if ratr_fn is not None:
                            ratr_lambda = getattr(cfg.SOLVER, 'RATR_LAMBDA', 1.0)
                            
                            # L2 归一化特征 (backbone + 最终融合)
                            backbone_feat_norm = F.normalize(feat[0].float(), dim=1)
                            fused_feat_norm = F.normalize(feat[-1].float(), dim=1)
                            
                            # 确保 RATR 模块在正确的 device 上
                            ratr_fn = ratr_fn.to(target.device)
                            
                            ratr_loss = ratr_fn([backbone_feat_norm, fused_feat_norm], target)
                            total_loss = total_loss + ratr_lambda * ratr_loss
                            
                            loss_detail['ratr'] = ratr_loss.item()
                        
                        return total_loss, loss_detail

                    raise ValueError('Unhandled multi-branch loss configuration')
                else:
                    # 单分支逻辑
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        ID_LOSS = xent(score.float(), target)
                    else:
                        ID_LOSS = F.cross_entropy(score.float(), target)
                    TRI_LOSS = triplet(feat.float(), target)[0]
                    
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))


    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
