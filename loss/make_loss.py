# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss, euclidean_dist, hard_example_mining
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

    def _ce_vector(inputs, targets):
        inputs = inputs.float()
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            log_probs = F.log_softmax(inputs, dim=1)
            one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            smooth_targets = (1 - xent.epsilon) * one_hot + xent.epsilon / num_classes
            return (-smooth_targets * log_probs).sum(dim=1)
        return F.cross_entropy(inputs, targets, reduction='none')

    def _id_loss(inputs, target_a, target_b=None, lam=None):
        loss_a = _ce_vector(inputs, target_a)
        if target_b is None or lam is None:
            return loss_a.mean()
        lam = lam.to(device=inputs.device, dtype=loss_a.dtype).view(-1)
        loss_b = _ce_vector(inputs, target_b.to(inputs.device))
        return (lam * loss_a + (1.0 - lam) * loss_b).mean()

    def _triplet_vector(global_feat, labels):
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        dist_ap *= (1.0 + triplet.hard_factor)
        dist_an *= (1.0 - triplet.hard_factor)
        y = dist_an.new_ones(dist_an.size())
        if triplet.margin is not None:
            return F.margin_ranking_loss(dist_an, dist_ap, y, margin=triplet.margin, reduction='none')
        return F.soft_margin_loss(dist_an - dist_ap, y, reduction='none')

    def _tri_loss(global_feat, target_a, target_b=None, lam=None):
        global_feat = global_feat.float()
        # OSBBM mixed labels are valid for ID/CE loss, but not for hard triplet
        # mining: donor labels do not preserve the PK sampler structure.
        return triplet(global_feat, target_a)[0]
    
    # RATR Loss 初始化
    ratr_fn = None
    if getattr(cfg.SOLVER, 'RATR_ENABLED', False):
        pk = cfg.DATALOADER.NUM_INSTANCE  # K 值
        p = cfg.SOLVER.IMS_PER_BATCH // pk  # P 值
        tau = getattr(cfg.SOLVER, 'RATR_TAU', 0.1)
        ratr_fn = RATRLoss(num_branches=2, num_classes=p, samples_per_class=pk, tau=tau)

    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam=None, osbbm_target=None, osbbm_lambda=None):
            return _id_loss(score, target, osbbm_target, osbbm_lambda)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam=None, osbbm_target=None, osbbm_lambda=None):
            nonlocal ratr_fn
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

                        osbbm_apply_to = str(getattr(getattr(cfg.INPUT, 'OSBBM', None), 'APPLY_TO', 'base')).lower()
                        for branch_idx, (branch_score, branch_feat) in enumerate(zip(score[:3], feat[:3])):
                            use_osbbm_labels = osbbm_target is not None and (
                                branch_idx == 0 or osbbm_apply_to == 'all'
                            )
                            branch_target_b = osbbm_target if use_osbbm_labels else None
                            branch_lambda = osbbm_lambda if use_osbbm_labels else None
                            id_losses.append(_id_loss(branch_score, target, branch_target_b, branch_lambda))
                            tri_losses.append(_tri_loss(branch_feat, target, branch_target_b, branch_lambda))

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
                            local_target_b = osbbm_target
                            local_lambda = osbbm_lambda
                            for local_score, local_feat in zip(score[aux_start:aux_end], feat[aux_start:aux_end]):
                                local_id_losses.append(_id_loss(local_score, target, local_target_b, local_lambda))
                                if local_use_triplet:
                                    local_tri_losses.append(_tri_loss(local_feat, target, local_target_b, local_lambda))

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
                        fusion_type = str(getattr(osnet_fusion_cfg, 'FUSION_TYPE', 'descriptor')).lower()
                        use_stage3_local = bool(getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_ENABLED', False))
                        use_stage4_local = bool(getattr(osnet_fusion_cfg, 'STAGE4_STRIPE_LOCAL_ENABLED', False))
                        use_stage3_part_sup = bool(
                            getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_PART_SUPERVISION_ENABLED', False)
                        )
                        local_num_stripes = int(getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_NUM_STRIPES', 4))
                        if not use_stage3_local:
                            use_stage3_part_sup = False
                            local_num_stripes = 0
                        expected_len = 3 + (1 if use_stage3_local else 0) + (
                            local_num_stripes if use_stage3_part_sup else 0
                        ) + (1 if use_stage4_local else 0)
                        if len(score) != expected_len or len(feat) != expected_len:
                            raise ValueError(
                                'OSNET_FUSION loss expects Mamba, OSNet, fused{}{}{} branches'.format(
                                    ', stage3_stripe_local' if use_stage3_local else '',
                                    ', and per-stripe locals' if use_stage3_part_sup else '',
                                    ', and stage4_stripe_local' if use_stage4_local else '',
                                )
                            )

                        fused_branch_name = 'fdmf' if fusion_type == 'fdmf' else 'concat'
                        branch_names = ['mamba', 'osnet', fused_branch_name]
                        branch_weights = [
                            1.0,
                            float(getattr(osnet_fusion_cfg, 'OSNET_LOSS_WEIGHT', 0.5)),
                            float(getattr(osnet_fusion_cfg, 'FUSED_LOSS_WEIGHT', 1.0)),
                        ]
                        if use_stage3_local:
                            branch_names.append('stage3_stripe_local')
                            branch_weights.append(float(getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_LOSS_WEIGHT', 0.2)))
                        if use_stage3_part_sup:
                            part_weight = float(getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_PART_LOSS_WEIGHT', 0.05))
                            for idx in range(local_num_stripes):
                                branch_names.append('stage3_part_{}'.format(idx + 1))
                                branch_weights.append(part_weight)
                        if use_stage4_local:
                            branch_names.append('stage4_stripe_local')
                            branch_weights.append(float(getattr(osnet_fusion_cfg, 'STAGE4_STRIPE_LOCAL_LOSS_WEIGHT', 0.2)))
                        weight_sum = sum(branch_weights)
                        if weight_sum <= 0:
                            raise ValueError('OSNET_FUSION branch weights must sum to a positive value')

                        id_losses = []
                        tri_losses = []
                        for branch_score, branch_feat in zip(score, feat):
                            id_losses.append(_id_loss(branch_score, target, osbbm_target, osbbm_lambda))
                            tri_losses.append(_tri_loss(branch_feat, target, osbbm_target, osbbm_lambda))

                        ID_LOSS = sum(w * l for w, l in zip(branch_weights, id_losses)) / weight_sum
                        TRI_LOSS = sum(w * l for w, l in zip(branch_weights, tri_losses)) / weight_sum

                        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                        loss_detail = {
                            'fusion_mode': 'osnet',
                            'fusion_type': fusion_type,
                            'fused_branch_key': fused_branch_name,
                            'w_mamba': branch_weights[0],
                            'w_osnet': branch_weights[1],
                            'w_concat': branch_weights[2],
                            'w_fdmf': branch_weights[2],
                        }
                        if use_stage3_local:
                            loss_detail['w_stage3_stripe_local'] = branch_weights[3]
                        if use_stage3_part_sup:
                            part_start = 4
                            part_id_losses = id_losses[part_start:part_start + local_num_stripes]
                            part_tri_losses = tri_losses[part_start:part_start + local_num_stripes]
                            loss_detail['w_stage3_part'] = float(
                                getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_PART_LOSS_WEIGHT', 0.05)
                            )
                            loss_detail['num_stage3_parts'] = local_num_stripes
                            loss_detail['id_stage3_part_avg'] = (
                                sum(loss.item() for loss in part_id_losses) / max(len(part_id_losses), 1)
                            )
                            loss_detail['tri_stage3_part_avg'] = (
                                sum(loss.item() for loss in part_tri_losses) / max(len(part_tri_losses), 1)
                            )
                        if use_stage4_local:
                            stage4_idx = len(branch_names) - 1
                            loss_detail['w_stage4_stripe_local'] = branch_weights[stage4_idx]
                        for name, id_loss, tri_loss in zip(branch_names, id_losses, tri_losses):
                            loss_detail[f'id_{name}'] = id_loss.item()
                            loss_detail[f'tri_{name}'] = tri_loss.item()

                        if ratr_fn is not None:
                            ratr_pair = str(getattr(cfg.SOLVER, 'RATR_BRANCH_PAIR', 'mamba_osnet')).lower()
                            pair_to_indices = {
                                'mamba_osnet': (0, 1),
                                'mamba_fused': (0, 2),
                                'osnet_fused': (1, 2),
                            }
                            if ratr_pair not in pair_to_indices:
                                raise ValueError(
                                    "SOLVER.RATR_BRANCH_PAIR must be one of: mamba_osnet, mamba_fused, osnet_fused"
                                )
                            idx_a, idx_b = pair_to_indices[ratr_pair]
                            ratr_lambda = float(getattr(cfg.SOLVER, 'RATR_LAMBDA', 1.0))
                            ratr_fn = ratr_fn.to(target.device)
                            ratr_loss = ratr_fn([
                                F.normalize(feat[idx_a].float(), dim=1),
                                F.normalize(feat[idx_b].float(), dim=1),
                            ], target)
                            total_loss = total_loss + ratr_lambda * ratr_loss
                            loss_detail['ratr'] = ratr_loss.item()
                            loss_detail['ratr_pair'] = ratr_pair
                            loss_detail['ratr_lambda'] = ratr_lambda

                        return total_loss, loss_detail

                    if use_sfm and len(score) >= 2:
                        # SFM 模式：HAT风格多级聚合损失 (归一化版本)
                        sfm_lambda = getattr(cfg.SOLVER, 'SFM_LAMBDA', 1.0)
                        sfm_lambda_aux = getattr(cfg.SOLVER, 'SFM_LAMBDA_AUX', 0.5)
                        
                        # 1. Backbone 分支 (权重 = 1.0)
                        id_b = _id_loss(score[0], target, osbbm_target, osbbm_lambda)
                        tri_b = _tri_loss(feat[0], target, osbbm_target, osbbm_lambda)
                        
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
                            
                            id_fi = _id_loss(score[i], target, osbbm_target, osbbm_lambda)
                            tri_fi = _tri_loss(feat[i], target, osbbm_target, osbbm_lambda)
                            
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
                    ID_LOSS = _id_loss(score, target, osbbm_target, osbbm_lambda)
                    TRI_LOSS = _tri_loss(feat, target, osbbm_target, osbbm_lambda)
                    
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))


    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
