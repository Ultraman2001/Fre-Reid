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
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def id_loss_value(score, target):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            return xent(score.float(), target)
        return F.cross_entropy(score.float(), target)

    ratr_fn = None
    if getattr(cfg.SOLVER, 'RATR_ENABLED', False):
        pk = cfg.DATALOADER.NUM_INSTANCE
        p = cfg.SOLVER.IMS_PER_BATCH // pk
        tau = getattr(cfg.SOLVER, 'RATR_TAU', 0.1)
        ratr_fn = RATRLoss(num_branches=2, num_classes=p, samples_per_class=pk, tau=tau)

    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam=None):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if isinstance(score, list):
                    use_pam = cfg.INPUT.PAM.ENABLED
                    use_sfm = getattr(cfg.MODEL.MAMBAVISION, 'USE_SFM', False)

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
                            id_losses.append(id_loss_value(branch_score, target))
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
                                local_id_losses.append(id_loss_value(local_score, target))
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

                    if use_sfm and len(score) >= 2:
                        sfm_lambda = getattr(cfg.SOLVER, 'SFM_LAMBDA', 1.0)
                        sfm_lambda_aux = getattr(cfg.SOLVER, 'SFM_LAMBDA_AUX', 0.5)

                        id_b = id_loss_value(score[0], target)
                        tri_b = triplet(feat[0].float(), target)[0]

                        total_id_loss = id_b
                        total_tri_loss = tri_b

                        loss_detail = {
                            'id_backbone': id_b.item(),
                            'tri_backbone': tri_b.item(),
                            'sfm_lambda': sfm_lambda,
                        }

                        for i in range(1, len(score)):
                            is_last = (i == len(score) - 1)
                            w = sfm_lambda if is_last else sfm_lambda_aux

                            id_fi = id_loss_value(score[i], target)
                            tri_fi = triplet(feat[i].float(), target)[0]

                            total_id_loss = total_id_loss + w * id_fi
                            total_tri_loss = total_tri_loss + w * tri_fi

                            loss_detail[f'id_fused_{i}'] = id_fi.item()
                            loss_detail[f'tri_fused_{i}'] = tri_fi.item()

                        num_fused = len(score) - 1
                        if num_fused > 0:
                            loss_detail['id_fused'] = loss_detail[f'id_fused_{num_fused}']
                            loss_detail['tri_fused'] = loss_detail[f'tri_fused_{num_fused}']

                        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * total_id_loss + \
                                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * total_tri_loss

                        nonlocal ratr_fn
                        if ratr_fn is not None:
                            ratr_lambda = getattr(cfg.SOLVER, 'RATR_LAMBDA', 1.0)
                            backbone_feat_norm = F.normalize(feat[0].float(), dim=1)
                            fused_feat_norm = F.normalize(feat[-1].float(), dim=1)
                            ratr_fn = ratr_fn.to(target.device)
                            ratr_loss = ratr_fn([backbone_feat_norm, fused_feat_norm], target)
                            total_loss = total_loss + ratr_lambda * ratr_loss
                            loss_detail['ratr'] = ratr_loss.item()

                        return total_loss, loss_detail

                ID_LOSS = id_loss_value(score, target)
                TRI_LOSS = triplet(feat.float(), target)[0]

                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            print('expected METRIC_LOSS_TYPE should be triplet'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func, center_criterion
