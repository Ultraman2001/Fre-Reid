# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss, euclidean_dist, hard_example_mining
from .center_loss import CenterLoss
from .ratr_loss import RATRLoss


def make_loss(cfg, num_classes):
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

    def _tri_loss(global_feat, target_a, target_b=None, lam=None):
        global_feat = global_feat.float()
        # OSBBM mixed labels are valid for ID/CE loss, but not for hard triplet
        # mining: donor labels do not preserve the PK sampler structure.
        return triplet(global_feat, target_a)[0]

    def _guided_tri_loss(local_feat, guide_feat, labels):
        """Mine hard pairs in a detached guide space, optimize them in local space."""
        local_feat = local_feat.float()
        with torch.no_grad():
            guide_dist = euclidean_dist(guide_feat.detach().float(), guide_feat.detach().float())
            _, _, positive_index, negative_index = hard_example_mining(
                guide_dist,
                labels,
                return_inds=True,
            )

        local_dist = euclidean_dist(local_feat, local_feat)
        anchor_index = torch.arange(labels.shape[0], device=labels.device)
        dist_ap = local_dist[anchor_index, positive_index]
        dist_an = local_dist[anchor_index, negative_index]
        dist_ap = dist_ap * (1.0 + triplet.hard_factor)
        dist_an = dist_an * (1.0 - triplet.hard_factor)
        if triplet.margin is not None:
            loss = triplet.ranking_loss(dist_an, dist_ap, torch.ones_like(dist_an))
        else:
            loss = F.softplus(dist_ap - dist_an).mean()
        return loss, dist_ap.detach().mean(), dist_an.detach().mean()

    def _part_avg_tri_loss(part_features, labels, confidence=None):
        if part_features.ndim != 3:
            raise ValueError('Part-averaged triplet expects features shaped [B, K, C]')
        features = part_features.float()
        pairwise = (
            features[:, None, :, :] - features[None, :, :, :]
        ).square().sum(dim=-1).clamp_min(1e-12).sqrt()
        if confidence is None:
            dist_mat = pairwise.mean(dim=-1)
        else:
            confidence = confidence.detach().float().clamp(0.0, 1.0)
            if confidence.shape != part_features.shape[:2]:
                raise ValueError('Local confidence must be shaped [B, K]')
            mutual = (
                confidence[:, None, :] * confidence[None, :, :]
            ).clamp_min(0.0).sqrt()
            dist_mat = (pairwise * mutual).sum(dim=-1) / mutual.sum(dim=-1).clamp_min(1e-6)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        dist_ap = dist_ap * (1.0 + triplet.hard_factor)
        dist_an = dist_an * (1.0 - triplet.hard_factor)
        if triplet.margin is not None:
            return triplet.ranking_loss(dist_an, dist_ap, torch.ones_like(dist_an))
        return F.softplus(dist_ap - dist_an).mean()

    def _pairwise_squared_distance(x):
        x = x.float()
        squared_norm = x.square().sum(dim=1, keepdim=True)
        return (squared_norm + squared_norm.t() - 2.0 * x.matmul(x.t())).clamp_min(0.0)

    def _peer_complement_losses(peer_features, labels, peer_cfg):
        include_local = bool(getattr(peer_cfg, 'INCLUDE_LOCAL', False))
        names = ['mamba', 'fdmf', 'osnet']
        if include_local:
            names.append('local')
        missing = [name for name in names if name not in peer_features]
        if missing:
            raise ValueError('PEER_COMPLEMENT missing features: {}'.format(', '.join(missing)))

        weights = {
            'mamba': float(getattr(peer_cfg, 'MAMBA_WEIGHT', 1.0)),
            'fdmf': float(getattr(peer_cfg, 'FDMF_WEIGHT', 0.75)),
            'osnet': float(getattr(peer_cfg, 'OSNET_WEIGHT', 0.4)),
            'local': float(getattr(peer_cfg, 'LOCAL_WEIGHT', 0.3)),
        }
        branch_features = {
            name: F.normalize(peer_features[name].float(), p=2, dim=1)
            for name in names
        }
        descriptor = F.normalize(
            torch.cat([weights[name] * branch_features[name] for name in names], dim=1),
            p=2,
            dim=1,
        )
        fused_distance = _pairwise_squared_distance(descriptor)
        same_identity = labels.view(-1, 1).eq(labels.view(1, -1))
        same_identity.fill_diagonal_(False)
        different_identity = ~labels.view(-1, 1).eq(labels.view(1, -1))
        valid = same_identity.any(dim=1) & different_identity.any(dim=1)
        if not valid.any():
            zero = descriptor.new_zeros(())
            return zero, zero, {'valid_triplets': 0}

        positive_index = fused_distance.detach().masked_fill(~same_identity, float('-inf')).argmax(dim=1)
        negative_index = fused_distance.detach().masked_fill(~different_identity, float('inf')).argmin(dim=1)
        anchor_index = torch.arange(labels.shape[0], device=labels.device)[valid]
        positive_index = positive_index[valid]
        negative_index = negative_index[valid]
        margin_target = float(getattr(peer_cfg, 'MARGIN', 0.3))
        temperature = max(float(getattr(peer_cfg, 'TEMPERATURE', 0.1)), 1e-6)

        fused_positive = fused_distance[anchor_index, positive_index]
        fused_negative = fused_distance[anchor_index, negative_index]
        fused_margin = fused_negative - fused_positive
        fused_loss = F.relu(margin_target - fused_margin).mean()

        branch_margins = {}
        for name in names:
            feature = branch_features[name]
            positive_distance = (feature[anchor_index] - feature[positive_index]).square().sum(dim=1)
            negative_distance = (feature[anchor_index] - feature[negative_index]).square().sum(dim=1)
            branch_margins[name] = negative_distance - positive_distance

        branch_losses = []
        stats = {
            'valid_triplets': int(anchor_index.numel()),
            'fused_margin': fused_margin.detach().mean().item(),
        }
        for name in names:
            peer_margin = torch.stack([
                branch_margins[peer_name].detach()
                for peer_name in names
                if peer_name != name
            ]).mean(dim=0)
            peer_deficit = torch.sigmoid((margin_target - peer_margin) / temperature)
            branch_violation = F.relu(margin_target - branch_margins[name])
            branch_losses.append((peer_deficit * branch_violation).mean())
            stats['{}_margin'.format(name)] = branch_margins[name].detach().mean().item()
            stats['{}_deficit'.format(name)] = peer_deficit.detach().mean().item()
        peer_loss = torch.stack(branch_losses).mean()
        return fused_loss, peer_loss, stats
    
    # RATR Loss 初始化
    ratr_fn = None
    if getattr(cfg.SOLVER, 'RATR_ENABLED', False):
        pk = cfg.DATALOADER.NUM_INSTANCE  # K 值
        p = cfg.SOLVER.IMS_PER_BATCH // pk  # P 值
        tau = getattr(cfg.SOLVER, 'RATR_TAU', 0.1)
        ratr_fn = RATRLoss(
            num_branches=2,
            num_classes=p,
            samples_per_class=pk,
            tau=tau,
            mode=getattr(cfg.SOLVER, 'RATR_MODE', 'raw'),
            intra_target=getattr(cfg.SOLVER, 'RATR_INTRA_TARGET', 0.5),
            inter_target=getattr(cfg.SOLVER, 'RATR_INTER_TARGET', 0.3),
        )

    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam=None, osbbm_target=None, osbbm_lambda=None, epoch=None):
            return _id_loss(score, target, osbbm_target, osbbm_lambda)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam=None, osbbm_target=None, osbbm_lambda=None, epoch=None):
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
                        complementarity_cfg = getattr(osnet_fusion_cfg, 'COMPLEMENTARITY', None)
                        complementarity_mode = str(
                            getattr(complementarity_cfg, 'MODE', 'none')
                        ).lower()
                        use_complementarity = complementarity_mode != 'none'
                        complementarity_aux = None
                        peer_complement_aux = None
                        stage3_local_regularization = None
                        stage3_local_parts = None
                        role_specialization_aux = None
                        branch_feats = feat
                        if feat and isinstance(feat[-1], dict):
                            complementarity_aux = feat[-1].get('complementarity')
                            peer_complement_aux = feat[-1].get('peer_complement')
                            stage3_local_regularization = feat[-1].get('stage3_local_regularization')
                            stage3_local_parts = feat[-1].get('stage3_local_parts')
                            role_specialization_aux = feat[-1].get('role_specialization')
                            branch_feats = feat[:-1]
                        use_stage3_local = bool(getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_ENABLED', False))
                        base_expected_len = 3 + (1 if use_stage3_local else 0)
                        expected_len = base_expected_len + (1 if use_complementarity else 0)
                        if len(score) != expected_len or len(branch_feats) != expected_len:
                            raise ValueError(
                                'OSNET_FUSION loss expects Mamba, OSNet, fused{}{} branches'.format(
                                    ', stage3_stripe_local' if use_stage3_local else '',
                                    ', and complementarity joint descriptor' if use_complementarity else '',
                                )
                            )
                        if use_complementarity and complementarity_aux is None:
                            raise ValueError('Complementarity mode requires model auxiliary loss data')
                        role_cfg = getattr(
                            osnet_fusion_cfg,
                            'ROLE_SPECIALIZATION',
                            None,
                        )
                        role_specialization_enabled = bool(
                            getattr(role_cfg, 'ENABLED', False)
                        )
                        if (
                            role_specialization_enabled
                            and role_specialization_aux is None
                        ):
                            raise ValueError(
                                'ROLE_SPECIALIZATION requires model auxiliary loss data'
                            )

                        fused_branch_name = 'fdmf' if fusion_type == 'fdmf' else 'concat'
                        branch_names = ['mamba', 'osnet', fused_branch_name]
                        base_branch_weights = [
                            1.0,
                            float(getattr(osnet_fusion_cfg, 'OSNET_LOSS_WEIGHT', 0.5)),
                            float(getattr(osnet_fusion_cfg, 'FUSED_LOSS_WEIGHT', 1.0)),
                        ]
                        base_weight_sum = sum(base_branch_weights)
                        if base_weight_sum <= 0:
                            raise ValueError('OSNET_FUSION branch weights must sum to a positive value')

                        id_losses = []
                        tri_losses = []
                        for branch_score, branch_feat in zip(score[:3], branch_feats[:3]):
                            id_losses.append(_id_loss(branch_score, target, osbbm_target, osbbm_lambda))
                            tri_losses.append(_tri_loss(branch_feat, target, osbbm_target, osbbm_lambda))

                        id_numerator = sum(
                            weight * branch_loss
                            for weight, branch_loss in zip(base_branch_weights, id_losses)
                        )
                        tri_numerator = sum(
                            weight * branch_loss
                            for weight, branch_loss in zip(base_branch_weights, tri_losses)
                        )
                        local_id_weight = 0.0
                        local_tri_weight = 0.0
                        id_denominator = base_weight_sum
                        tri_denominator = base_weight_sum
                        if use_stage3_local:
                            legacy_local_weight = float(
                                getattr(osnet_fusion_cfg, 'STAGE3_STRIPE_LOCAL_LOSS_WEIGHT', 0.2)
                            )
                            local_id_weight = float(
                                getattr(osnet_fusion_cfg, 'STAGE3_LOCAL_ID_LOSS_WEIGHT', -1.0)
                            )
                            local_tri_weight = float(
                                getattr(osnet_fusion_cfg, 'STAGE3_LOCAL_TRIPLET_LOSS_WEIGHT', -1.0)
                            )
                            if local_id_weight < 0:
                                local_id_weight = legacy_local_weight
                            if local_tri_weight < 0:
                                local_tri_weight = legacy_local_weight
                            if local_id_weight < 0 or local_tri_weight < 0:
                                raise ValueError('Local ID/triplet weights must be non-negative')

                            flat_local_id = _id_loss(
                                score[3], target, osbbm_target, osbbm_lambda
                            )
                            flat_local_tri = _tri_loss(
                                branch_feats[3], target, osbbm_target, osbbm_lambda
                            )
                            selected_local_id = flat_local_id
                            selected_local_tri = flat_local_tri
                            part_id_mode = str(
                                getattr(osnet_fusion_cfg, 'STAGE3_LOCAL_PART_ID_MODE', 'none')
                            ).lower()
                            triplet_mode = str(
                                getattr(osnet_fusion_cfg, 'STAGE3_LOCAL_TRIPLET_MODE', 'flat')
                            ).lower()
                            confidence_mode = str(
                                getattr(osnet_fusion_cfg, 'STAGE3_LOCAL_CONFIDENCE_MODE', 'none')
                            ).lower()
                            if part_id_mode not in ('none', 'replace', 'joint'):
                                raise ValueError(
                                    'STAGE3_LOCAL_PART_ID_MODE must be none, replace, or joint'
                                )
                            if triplet_mode not in ('flat', 'part_avg'):
                                raise ValueError(
                                    'STAGE3_LOCAL_TRIPLET_MODE must be flat or part_avg'
                                )
                            part_id_loss = None
                            part_tri_loss = None
                            guided_local_tri = None
                            guided_local_ap = None
                            guided_local_an = None
                            if part_id_mode != 'none' or triplet_mode == 'part_avg':
                                if stage3_local_parts is None:
                                    raise ValueError(
                                        'Part-specific local supervision requires semantic-detail part outputs'
                                    )
                            if part_id_mode != 'none':
                                part_scores = stage3_local_parts.get('scores', [])
                                if not part_scores:
                                    raise ValueError(
                                        'Part ID mode requires part-specific classifiers in the model'
                                    )
                                part_id_loss = torch.stack([
                                    _id_loss(part_score, target, osbbm_target, osbbm_lambda)
                                    for part_score in part_scores
                                ]).mean()
                                if part_id_mode == 'replace':
                                    selected_local_id = part_id_loss
                                else:
                                    joint_fraction = float(
                                        getattr(
                                            osnet_fusion_cfg,
                                            'STAGE3_LOCAL_PART_ID_JOINT_FRACTION',
                                            0.5,
                                        )
                                    )
                                    if not 0.0 <= joint_fraction <= 1.0:
                                        raise ValueError(
                                            'STAGE3_LOCAL_PART_ID_JOINT_FRACTION must be in [0, 1]'
                                        )
                                    selected_local_id = (
                                        (1.0 - joint_fraction) * flat_local_id
                                        + joint_fraction * part_id_loss
                                    )
                            if triplet_mode == 'part_avg':
                                part_confidence = None
                                if confidence_mode in ('triplet', 'descriptor'):
                                    part_confidence = stage3_local_parts.get('confidence')
                                part_tri_loss = _part_avg_tri_loss(
                                    stage3_local_parts['features'],
                                    target,
                                    confidence=part_confidence,
                                )
                                selected_local_tri = part_tri_loss

                            guided_mix = float(
                                getattr(
                                    osnet_fusion_cfg,
                                    'STAGE3_LOCAL_GUIDED_TRIPLET_MIX',
                                    0.0,
                                )
                            )
                            guided_source = str(
                                getattr(
                                    osnet_fusion_cfg,
                                    'STAGE3_LOCAL_GUIDED_TRIPLET_SOURCE',
                                    'main',
                                )
                            ).lower()
                            if not 0.0 <= guided_mix <= 1.0:
                                raise ValueError(
                                    'STAGE3_LOCAL_GUIDED_TRIPLET_MIX must be in [0, 1]'
                                )
                            guided_source_indices = {'mamba': 0, 'osnet': 1, 'fdmf': 2}
                            if guided_source not in ('main', *guided_source_indices):
                                raise ValueError(
                                    'STAGE3_LOCAL_GUIDED_TRIPLET_SOURCE must be main, mamba, osnet, or fdmf'
                                )
                            if guided_mix > 0.0:
                                if guided_source == 'main':
                                    if bool(getattr(osnet_fusion_cfg, 'FDMF_BYPASS', False)):
                                        guide_feat = torch.cat([
                                            float(getattr(complementarity_cfg, 'MAMBA_WEIGHT', 1.0))
                                            * F.normalize(branch_feats[0].float(), p=2, dim=1),
                                            float(getattr(complementarity_cfg, 'OSNET_WEIGHT', 0.4))
                                            * F.normalize(branch_feats[1].float(), p=2, dim=1),
                                        ], dim=1)
                                    else:
                                        guide_feat = torch.cat([
                                            float(getattr(complementarity_cfg, 'MAMBA_WEIGHT', 1.0))
                                            * F.normalize(branch_feats[0].float(), p=2, dim=1),
                                            float(getattr(complementarity_cfg, 'FDMF_WEIGHT', 0.75))
                                            * F.normalize(branch_feats[2].float(), p=2, dim=1),
                                            float(getattr(complementarity_cfg, 'OSNET_WEIGHT', 0.4))
                                            * F.normalize(branch_feats[1].float(), p=2, dim=1),
                                        ], dim=1)
                                else:
                                    guide_feat = branch_feats[guided_source_indices[guided_source]]
                                guided_local_tri, guided_local_ap, guided_local_an = _guided_tri_loss(
                                    branch_feats[3],
                                    guide_feat,
                                    target,
                                )
                                selected_local_tri = (
                                    (1.0 - guided_mix) * selected_local_tri
                                    + guided_mix * guided_local_tri
                                )

                            id_losses.append(selected_local_id)
                            tri_losses.append(selected_local_tri)
                            branch_names.append('stage3_stripe_local')
                            id_numerator = id_numerator + local_id_weight * selected_local_id
                            tri_numerator = tri_numerator + local_tri_weight * selected_local_tri
                            id_denominator = base_weight_sum + local_id_weight
                            tri_denominator = base_weight_sum + local_tri_weight

                        denominator_override = float(
                            getattr(osnet_fusion_cfg, 'STAGE3_LOCAL_LOSS_DENOMINATOR', 0.0)
                        )
                        if denominator_override > 0:
                            id_denominator = denominator_override
                            tri_denominator = denominator_override

                        if id_denominator <= 0 or tri_denominator <= 0:
                            raise ValueError('OSNET_FUSION loss denominators must be positive')
                        ID_LOSS = id_numerator / id_denominator
                        TRI_LOSS = tri_numerator / tri_denominator

                        total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                        loss_detail = {
                            'fusion_mode': 'osnet',
                            'fusion_type': fusion_type,
                            'fused_branch_key': fused_branch_name,
                            'w_mamba': base_branch_weights[0],
                            'w_osnet': base_branch_weights[1],
                            'w_concat': base_branch_weights[2],
                            'w_fdmf': base_branch_weights[2],
                            'id_denominator': id_denominator,
                            'tri_denominator': tri_denominator,
                        }
                        if role_specialization_enabled:
                            role_compensation_enabled = bool(
                                getattr(role_cfg, 'COMPENSATION_ENABLED', True)
                            )
                            role_compensation_weight = (
                                float(
                                    getattr(
                                        role_cfg,
                                        'COMPENSATION_LOSS_WEIGHT',
                                        0.05,
                                    )
                                )
                                if role_compensation_enabled
                                else 0.0
                            )
                            role_compensation_start_epoch = int(
                                getattr(
                                    role_cfg,
                                    'COMPENSATION_START_EPOCH',
                                    0,
                                )
                            )
                            role_compensation_ramp_epochs = int(
                                getattr(
                                    role_cfg,
                                    'COMPENSATION_RAMP_EPOCHS',
                                    0,
                                )
                            )
                            if role_compensation_start_epoch < 0:
                                raise ValueError(
                                    'COMPENSATION_START_EPOCH must be non-negative'
                                )
                            if role_compensation_ramp_epochs < 0:
                                raise ValueError(
                                    'COMPENSATION_RAMP_EPOCHS must be non-negative'
                                )
                            current_epoch = (
                                int(epoch)
                                if epoch is not None
                                else role_compensation_start_epoch
                                + role_compensation_ramp_epochs
                            )
                            if current_epoch < role_compensation_start_epoch:
                                role_compensation_scale = 0.0
                            elif role_compensation_ramp_epochs > 0:
                                role_compensation_scale = min(
                                    float(
                                        current_epoch
                                        - role_compensation_start_epoch
                                        + 1
                                    )
                                    / float(role_compensation_ramp_epochs),
                                    1.0,
                                )
                            else:
                                role_compensation_scale = 1.0
                            effective_role_compensation_weight = (
                                role_compensation_weight
                                * role_compensation_scale
                            )
                            role_compensation_loss = role_specialization_aux[
                                'compensation_loss'
                            ]
                            total_loss = total_loss + (
                                effective_role_compensation_weight
                                * role_compensation_loss
                            )
                            loss_detail.update({
                                'role_branch': role_specialization_aux['branch'],
                                'role_location': role_specialization_aux['location'],
                                'role_mask_type': role_specialization_aux['mask_type'],
                                'role_compensation_enabled': role_compensation_enabled,
                                'role_compensation_loss': role_compensation_loss.item(),
                                'role_compensation_base_weight': role_compensation_weight,
                                'role_compensation_weight': effective_role_compensation_weight,
                                'role_compensation_scale': role_compensation_scale,
                                'role_drop_fraction': role_specialization_aux[
                                    'drop_fraction'
                                ].item(),
                                'role_pre_cosine': role_specialization_aux[
                                    'pre_cosine'
                                ].item(),
                                'role_post_cosine': role_specialization_aux[
                                    'post_cosine'
                                ].item(),
                                'role_recovery_gain': role_specialization_aux[
                                    'recovery_gain'
                                ].item(),
                            })
                        if use_stage3_local:
                            loss_detail.update({
                                'w_stage3_local_id': local_id_weight,
                                'w_stage3_local_tri': local_tri_weight,
                                'local_part_id_mode': part_id_mode,
                                'local_triplet_mode': triplet_mode,
                                'local_confidence_mode': confidence_mode,
                                'local_guided_triplet_mix': guided_mix,
                                'local_guided_triplet_source': guided_source,
                                'id_stage3_stripe_local_flat': flat_local_id.item(),
                                'tri_stage3_stripe_local_flat': flat_local_tri.item(),
                            })
                            if part_id_loss is not None:
                                loss_detail['id_stage3_stripe_local_parts'] = part_id_loss.item()
                            if part_tri_loss is not None:
                                loss_detail['tri_stage3_stripe_local_parts'] = part_tri_loss.item()
                            if guided_local_tri is not None:
                                loss_detail.update({
                                    'tri_stage3_stripe_local_guided': guided_local_tri.item(),
                                    'stage3_local_guided_ap': guided_local_ap.item(),
                                    'stage3_local_guided_an': guided_local_an.item(),
                                })
                        if use_complementarity:
                            joint_id_weight = float(
                                getattr(complementarity_cfg, 'JOINT_ID_LOSS_WEIGHT', 0.0)
                            )
                            joint_triplet_weight = float(
                                getattr(complementarity_cfg, 'JOINT_TRIPLET_LOSS_WEIGHT', 0.0)
                            )
                            joint_score = score[base_expected_len]
                            joint_feat = branch_feats[base_expected_len]
                            joint_id_loss = (
                                _id_loss(joint_score, target, osbbm_target, osbbm_lambda)
                                if joint_id_weight > 0
                                else joint_feat.new_zeros(())
                            )
                            joint_tri_loss = (
                                _tri_loss(joint_feat, target, osbbm_target, osbbm_lambda)
                                if joint_triplet_weight > 0
                                else joint_feat.new_zeros(())
                            )
                            total_loss = total_loss + (
                                joint_id_weight * cfg.MODEL.ID_LOSS_WEIGHT * joint_id_loss
                                + joint_triplet_weight * cfg.MODEL.TRIPLET_LOSS_WEIGHT * joint_tri_loss
                            )

                            prediction_weight = float(
                                getattr(complementarity_cfg, 'PREDICTION_LOSS_WEIGHT', 0.1)
                            )
                            energy_weight = float(
                                getattr(complementarity_cfg, 'RESIDUAL_ENERGY_WEIGHT', 0.05)
                            )
                            covariance_weight = float(
                                getattr(complementarity_cfg, 'COVARIANCE_LOSS_WEIGHT', 0.02)
                            )
                            prediction_loss = complementarity_aux['prediction_loss']
                            energy_loss = complementarity_aux['residual_energy_loss']
                            covariance_loss = complementarity_aux['covariance_loss']
                            total_loss = total_loss + (
                                prediction_weight * prediction_loss
                                + energy_weight * energy_loss
                                + covariance_weight * covariance_loss
                            )
                            loss_detail.update({
                                'complementarity_mode': complementarity_mode,
                                'joint_id_weight': joint_id_weight,
                                'joint_triplet_weight': joint_triplet_weight,
                                'id_joint': joint_id_loss.item(),
                                'tri_joint': joint_tri_loss.item(),
                                'prediction_loss': prediction_loss.item(),
                                'prediction_weight': prediction_weight,
                                'residual_energy_loss': energy_loss.item(),
                                'residual_energy_weight': energy_weight,
                                'covariance_loss': covariance_loss.item(),
                                'covariance_weight': covariance_weight,
                                'dropped_branch': complementarity_aux.get('dropped_branch', 'none'),
                            })
                        peer_cfg = getattr(osnet_fusion_cfg, 'PEER_COMPLEMENT', None)
                        if bool(getattr(peer_cfg, 'ENABLED', False)):
                            if osbbm_target is not None:
                                raise ValueError('PEER_COMPLEMENT does not support OSBBM mixed-label batches')
                            if peer_complement_aux is None:
                                raise ValueError('PEER_COMPLEMENT requires model auxiliary features')
                            fused_metric_loss, peer_loss, peer_stats = _peer_complement_losses(
                                peer_complement_aux,
                                target,
                                peer_cfg,
                            )
                            start_epoch = int(getattr(peer_cfg, 'START_EPOCH', 40))
                            ramp_epochs = max(int(getattr(peer_cfg, 'RAMP_EPOCHS', 20)), 1)
                            current_epoch = int(epoch) if epoch is not None else start_epoch + ramp_epochs
                            if current_epoch < start_epoch:
                                ramp_scale = 0.0
                            else:
                                ramp_scale = min(
                                    float(current_epoch - start_epoch + 1) / float(ramp_epochs),
                                    1.0,
                                )
                            fused_metric_weight = float(
                                getattr(peer_cfg, 'FUSED_METRIC_WEIGHT', 0.1)
                            )
                            peer_loss_weight = float(
                                getattr(peer_cfg, 'PEER_LOSS_WEIGHT', 0.05)
                            )
                            total_loss = total_loss + ramp_scale * (
                                fused_metric_weight * fused_metric_loss
                                + peer_loss_weight * peer_loss
                            )
                            loss_detail.update({
                                'peer_complement_scale': ramp_scale,
                                'peer_fused_metric': fused_metric_loss.item(),
                                'peer_fused_metric_weight': fused_metric_weight,
                                'peer_loss': peer_loss.item(),
                                'peer_loss_weight': peer_loss_weight,
                                'peer_stats': peer_stats,
                            })
                        if stage3_local_regularization is not None:
                            balance_weight = float(
                                getattr(osnet_fusion_cfg, 'STAGE3_SOFT_BALANCE_WEIGHT', 0.01)
                            )
                            order_weight = float(
                                getattr(osnet_fusion_cfg, 'STAGE3_SOFT_ORDER_WEIGHT', 0.01)
                            )
                            balance_loss = stage3_local_regularization['balance_loss']
                            order_loss = stage3_local_regularization['order_loss']
                            total_loss = total_loss + (
                                balance_weight * balance_loss
                                + order_weight * order_loss
                            )
                            loss_detail.update({
                                'stage3_soft_balance': balance_loss.item(),
                                'stage3_soft_balance_weight': balance_weight,
                                'stage3_soft_order': order_loss.item(),
                                'stage3_soft_order_weight': order_weight,
                                'stage3_soft_fraction': stage3_local_regularization['part_fraction'].cpu().tolist(),
                                'stage3_soft_centers': stage3_local_regularization['part_centers'].cpu().tolist(),
                            })
                            if 'part_confidence' in stage3_local_regularization:
                                loss_detail['stage3_part_confidence'] = (
                                    stage3_local_regularization['part_confidence'].cpu().tolist()
                                )
                        if use_stage3_local:
                            loss_detail['w_stage3_stripe_local'] = legacy_local_weight
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
                            loss_detail['ratr_intra_corr'] = ratr_fn.last_intra
                            loss_detail['ratr_inter_corr'] = ratr_fn.last_inter
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
                        
                        total_id_loss = id_b
                        total_tri_loss = tri_b
                        
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
