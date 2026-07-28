import logging
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.osbbm import apply_osbbm_batch
from utils.feature_complementarity import (
    analyze_feature_complementarity,
    attach_retrieval_results,
    attach_weight_sweep_results,
    build_branch_normalized_descriptors,
    iter_weighted_branch_descriptors,
    log_feature_complementarity,
    save_feature_complementarity,
)
from torch import amp
import torch.distributed as dist


def _top_level_tensor_health(value, prefix, names=None):
    """Summarize branch tensors only when a non-finite loss is detected."""
    if torch.is_tensor(value):
        items = [(prefix, value)]
    elif isinstance(value, (list, tuple)):
        items = []
        for idx, item in enumerate(value):
            if not torch.is_tensor(item):
                continue
            name = names[idx] if names is not None and idx < len(names) else str(idx)
            items.append(('{}.{}'.format(prefix, name), item))
    elif isinstance(value, dict):
        items = [
            ('{}.{}'.format(prefix, key), item)
            for key, item in value.items()
            if torch.is_tensor(item)
        ]
    else:
        return []

    summaries = []
    for name, tensor in items:
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        total = detached.numel()
        finite_count = int(finite.sum().item())
        if finite_count:
            abs_max = detached[finite].float().abs().max().item()
        else:
            abs_max = float('nan')
        summaries.append(
            '{} shape={} finite={}/{} nan={} inf={} absmax={:.3e}'.format(
                name,
                tuple(detached.shape),
                finite_count,
                total,
                int(torch.isnan(detached).sum().item()),
                int(torch.isinf(detached).sum().item()),
                abs_max,
            )
        )
    return summaries


def _nonfinite_loss_details(loss_detail):
    if not isinstance(loss_detail, dict):
        return []
    bad = []
    for key, value in loss_detail.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            bad.append('{}={}'.format(key, value))
    return bad


def _nonfinite_parameter_names(model, limit=12):
    bad = []
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and not torch.isfinite(parameter.detach()).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _osbbm_active_for_epoch(osbbm_cfg, epoch, max_epochs):
    schedule = str(getattr(osbbm_cfg, 'SCHEDULE', 'always')).lower()
    start_epoch = int(getattr(osbbm_cfg, 'START_EPOCH', 1))
    end_epoch = int(getattr(osbbm_cfg, 'END_EPOCH', 0))
    if end_epoch <= 0:
        end_epoch = max_epochs

    if schedule == 'always':
        return True
    if epoch < start_epoch or epoch > end_epoch:
        return False
    if schedule == 'range':
        return True
    if schedule == 'cycle':
        period = max(int(getattr(osbbm_cfg, 'PERIOD_EPOCHS', 20)), 1)
        on_epochs = int(getattr(osbbm_cfg, 'ON_EPOCHS', 10))
        on_epochs = min(max(on_epochs, 0), period)
        cycle_pos = (epoch - start_epoch) % period
        return cycle_pos < on_epochs
    raise ValueError("INPUT.OSBBM.SCHEDULE must be 'always', 'range', or 'cycle'")


def _select_eval_feature(cfg, feat):
    if not isinstance(feat, dict):
        return feat

    feat_mode = str(getattr(getattr(cfg, 'TEST', None), 'FEAT_MODE', 'concat')).lower()
    if feat_mode in feat:
        return feat[feat_mode]
    for fallback in ('fdmf', 'concat', 'raw_concat', 'backbone'):
        if fallback in feat:
            return feat[fallback]
    raise KeyError('No usable feature key found in model output: {}'.format(sorted(feat.keys())))


def _cosine_with_padding(x, y):
    if x.shape[1] < y.shape[1]:
        x = F.pad(x, (0, y.shape[1] - x.shape[1]))
    elif y.shape[1] < x.shape[1]:
        y = F.pad(y, (0, x.shape[1] - y.shape[1]))
    return F.cosine_similarity(x, y, dim=1).mean().item()


def _fdmf_training_stats(cfg, feat):
    osnet_fusion_cfg = getattr(getattr(cfg, 'MODEL', None), 'OSNET_FUSION', None)
    if not bool(getattr(osnet_fusion_cfg, 'ENABLED', False)):
        return None
    if str(getattr(osnet_fusion_cfg, 'FUSION_TYPE', 'descriptor')).lower() != 'fdmf':
        return None
    if not isinstance(feat, list) or len(feat) < 3:
        return None

    mamba_feat = feat[0].detach().float()
    osnet_feat = feat[1].detach().float()
    fused_feat = feat[2].detach().float()
    mamba_dim = mamba_feat.shape[1]
    osnet_dim = osnet_feat.shape[1]
    fused_form = str(getattr(osnet_fusion_cfg, 'FDMF_FUSED_FORM', 'raw_fdmf')).lower()

    if fused_form == 'fdmf_only':
        fdmf_feat = fused_feat
    elif fused_form == 'mamba_fdmf':
        fdmf_feat = fused_feat[:, mamba_dim:]
    elif fused_feat.shape[1] >= mamba_dim + osnet_dim:
        fdmf_feat = fused_feat[:, mamba_dim + osnet_dim:]
    else:
        return None

    raw_concat = torch.cat([mamba_feat, osnet_feat], dim=1)
    stats = {
        'mamba_norm': torch.norm(mamba_feat, p=2, dim=1).mean().item(),
        'fdmf_norm': torch.norm(fdmf_feat, p=2, dim=1).mean().item(),
        'cos_mamba_fdmf': F.cosine_similarity(mamba_feat, fdmf_feat, dim=1).mean().item()
        if mamba_feat.shape[1] == fdmf_feat.shape[1] else 0.0,
        'cos_fused_raw': _cosine_with_padding(fused_feat, raw_concat),
    }
    return stats


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             model_ema=None):  # Added model_ema
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    pam_enabled = cfg.INPUT.PAM.ENABLED
    dual_view_enabled = bool(getattr(cfg.INPUT.DUAL_VIEW, 'ENABLED', False))
    if pam_enabled and dual_view_enabled:
        raise ValueError('INPUT.PAM and INPUT.DUAL_VIEW are mutually exclusive')

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    osbbm_cfg = getattr(cfg.INPUT, 'OSBBM', None)
    osbbm_enabled = bool(getattr(osbbm_cfg, 'ENABLED', False))
    if dual_view_enabled and osbbm_enabled:
        raise ValueError(
            'INPUT.DUAL_VIEW and INPUT.OSBBM must be evaluated separately'
        )
    osbbm_mixed_label = bool(getattr(osbbm_cfg, 'MIXED_LABEL', True))
    if osbbm_enabled:
        osbbm_end_epoch = int(getattr(osbbm_cfg, 'END_EPOCH', 0))
        if osbbm_end_epoch <= 0:
            osbbm_end_epoch = epochs
        logger.info(
            "[OSBBM] enabled: prob={:.2f}, blocks={}, mix_blocks={}, gray_prob={:.2f}, gray_scope={}, sample={}, donor={}, block_mode={}, apply_to={}, mixed_label={}, schedule={}, start={}, end={}, period={}, on={}".format(
                float(getattr(osbbm_cfg, 'PROB', 0.5)),
                int(getattr(osbbm_cfg, 'NUM_BLOCKS', 8)),
                int(getattr(osbbm_cfg, 'NUM_MIX_BLOCKS', 2)),
                float(getattr(osbbm_cfg, 'GRAY_PROB', 0.5)),
                str(getattr(osbbm_cfg, 'GRAY_SCOPE', 'all')).lower(),
                str(getattr(osbbm_cfg, 'SAMPLE_MODE', 'random')).lower(),
                str(getattr(osbbm_cfg, 'DONOR_MODE', 'random')).lower(),
                str(getattr(osbbm_cfg, 'BLOCK_MODE', 'random')).lower(),
                str(getattr(osbbm_cfg, 'APPLY_TO', 'base')).lower(),
                osbbm_mixed_label,
                str(getattr(osbbm_cfg, 'SCHEDULE', 'always')).lower(),
                int(getattr(osbbm_cfg, 'START_EPOCH', 1)),
                osbbm_end_epoch,
                int(getattr(osbbm_cfg, 'PERIOD_EPOCHS', 20)),
                int(getattr(osbbm_cfg, 'ON_EPOCHS', 10)),
            )
        )
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler('cuda')
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        osbbm_active = osbbm_enabled and _osbbm_active_for_epoch(osbbm_cfg, epoch, epochs)
        if osbbm_enabled and (epoch == 1 or str(getattr(osbbm_cfg, 'SCHEDULE', 'always')).lower() != 'always'):
            logger.info("[OSBBM] Epoch {} active={}".format(epoch, osbbm_active))
        for n_iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            osbbm_target_b = None
            osbbm_lambda = None
            if pam_enabled:
                img_base, img_crop, img_erase, vid, target_cam, target_view = batch
                target = vid.to(device)
                img_base = img_base.to(device)
                img_crop = img_crop.to(device)
                img_erase = img_erase.to(device)
                if osbbm_active:
                    apply_to = str(getattr(osbbm_cfg, 'APPLY_TO', 'base')).lower()
                    if apply_to not in ('base', 'all'):
                        raise ValueError("INPUT.OSBBM.APPLY_TO must be 'base' or 'all'")
                    img_base, osbbm_target_b, osbbm_lambda, osbbm_info = apply_osbbm_batch(
                        img_base,
                        target,
                        cfg.INPUT.PIXEL_MEAN,
                        cfg.INPUT.PIXEL_STD,
                        prob=float(getattr(osbbm_cfg, 'PROB', 0.5)),
                        num_blocks=int(getattr(osbbm_cfg, 'NUM_BLOCKS', 8)),
                        num_mix_blocks=int(getattr(osbbm_cfg, 'NUM_MIX_BLOCKS', 2)),
                        gray_prob=float(getattr(osbbm_cfg, 'GRAY_PROB', 0.5)),
                        gray_scope=str(getattr(osbbm_cfg, 'GRAY_SCOPE', 'all')),
                        sample_mode=str(getattr(osbbm_cfg, 'SAMPLE_MODE', 'random')),
                        donor_mode=str(getattr(osbbm_cfg, 'DONOR_MODE', 'random')),
                        block_mode=str(getattr(osbbm_cfg, 'BLOCK_MODE', 'random')),
                        return_info=True,
                    )
                    if apply_to == 'all':
                        img_crop = apply_osbbm_batch(
                            img_crop,
                            target,
                            cfg.INPUT.PIXEL_MEAN,
                            cfg.INPUT.PIXEL_STD,
                            prob=float(getattr(osbbm_cfg, 'PROB', 0.5)),
                            num_blocks=int(getattr(osbbm_cfg, 'NUM_BLOCKS', 8)),
                            num_mix_blocks=int(getattr(osbbm_cfg, 'NUM_MIX_BLOCKS', 2)),
                            gray_prob=float(getattr(osbbm_cfg, 'GRAY_PROB', 0.5)),
                            gray_scope=str(getattr(osbbm_cfg, 'GRAY_SCOPE', 'all')),
                            mix_info=osbbm_info,
                        )
                        img_erase = apply_osbbm_batch(
                            img_erase,
                            target,
                            cfg.INPUT.PIXEL_MEAN,
                            cfg.INPUT.PIXEL_STD,
                            prob=float(getattr(osbbm_cfg, 'PROB', 0.5)),
                            num_blocks=int(getattr(osbbm_cfg, 'NUM_BLOCKS', 8)),
                            num_mix_blocks=int(getattr(osbbm_cfg, 'NUM_MIX_BLOCKS', 2)),
                            gray_prob=float(getattr(osbbm_cfg, 'GRAY_PROB', 0.5)),
                            gray_scope=str(getattr(osbbm_cfg, 'GRAY_SCOPE', 'all')),
                            mix_info=osbbm_info,
                        )
                img = (
                    img_base,
                    img_crop,
                    img_erase,
                )
                if not osbbm_mixed_label:
                    osbbm_target_b = None
                    osbbm_lambda = None
                batch_size = img_base.shape[0]
            elif dual_view_enabled:
                img_mamba, img_osnet, vid, target_cam, target_view = batch
                target = vid.to(device)
                img = (
                    img_mamba.to(device),
                    img_osnet.to(device),
                )
                batch_size = img[0].shape[0]
            else:
                img, vid, target_cam, target_view = batch
                img = img.to(device)
                target = vid.to(device)
                if osbbm_active:
                    img, osbbm_target_b, osbbm_lambda, _ = apply_osbbm_batch(
                        img,
                        target,
                        cfg.INPUT.PIXEL_MEAN,
                        cfg.INPUT.PIXEL_STD,
                        prob=float(getattr(osbbm_cfg, 'PROB', 0.5)),
                        num_blocks=int(getattr(osbbm_cfg, 'NUM_BLOCKS', 8)),
                        num_mix_blocks=int(getattr(osbbm_cfg, 'NUM_MIX_BLOCKS', 2)),
                        gray_prob=float(getattr(osbbm_cfg, 'GRAY_PROB', 0.5)),
                        gray_scope=str(getattr(osbbm_cfg, 'GRAY_SCOPE', 'all')),
                        sample_mode=str(getattr(osbbm_cfg, 'SAMPLE_MODE', 'random')),
                        donor_mode=str(getattr(osbbm_cfg, 'DONOR_MODE', 'random')),
                        block_mode=str(getattr(osbbm_cfg, 'BLOCK_MODE', 'random')),
                        return_info=True,
                    )
                    if not osbbm_mixed_label:
                        osbbm_target_b = None
                        osbbm_lambda = None
                batch_size = img.shape[0]
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast('cuda'):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            
            # Loss 计算在 FP32 下进行，避免 FP16 溢出导致 NaN
            loss_result = loss_fn(
                score,
                feat,
                target,
                target_cam,
                osbbm_target_b,
                osbbm_lambda,
                epoch=epoch,
            )
            if isinstance(loss_result, tuple):
                loss, loss_detail = loss_result
            else:
                loss, loss_detail = loss_result, None
            if not torch.isfinite(loss).all():
                branch_names = ('mamba', 'osnet', 'fdmf', 'stage3_local')
                input_stats = _top_level_tensor_health(img, 'input')
                score_stats = _top_level_tensor_health(score, 'score', branch_names)
                feat_stats = _top_level_tensor_health(feat, 'feat', branch_names)
                bad_loss_terms = _nonfinite_loss_details(loss_detail)
                bad_parameters = _nonfinite_parameter_names(model)
                raise FloatingPointError(
                    'Non-finite loss at epoch {} iteration {}; optimizer step aborted. '
                    'OSBBM_active={}; nonfinite_loss_terms={}; nonfinite_parameters={}; '
                    'tensor_health={}'.format(
                        epoch,
                        n_iter + 1,
                        osbbm_active,
                        bad_loss_terms or 'none reported',
                        bad_parameters or 'none',
                        ' | '.join(input_stats + score_stats + feat_stats),
                    )
                )
            fdmf_stats = _fdmf_training_stats(cfg, feat)
            scaler.scale(loss).backward()

            # Add Gradient Clipping (Configurable)
            if cfg.SOLVER.CLIP_GRAD_NORM > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            if model_ema is not None:
                model_ema.update(model)

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            
            # Multi-branch modes use their primary branch for training accuracy.
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # 基础日志
                log_msg = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                    epoch, (n_iter + 1), len(train_loader),
                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0])
                
                # Multi-branch modes: append concise branch details.
                if loss_detail is not None:
                    if 'pam_weight' in loss_detail:
                        log_msg += " | PAM[w={:.2f}]".format(loss_detail['pam_weight'])
                        for name in ('ba', 'ca', 'ea'):
                            log_msg += " | {}[ID={:.2f}, Tri={:.4f}]".format(
                                name.upper(),
                                loss_detail[f'id_{name}'],
                                loss_detail[f'tri_{name}'],
                            )
                        if 'local_weight' in loss_detail:
                            log_msg += " | BA-Local[w={:.2f}, ID={:.2f}, Tri={:.4f}]".format(
                                loss_detail['local_weight'],
                                loss_detail['id_local'],
                                loss_detail['tri_local'],
                            )
                    elif loss_detail.get('fusion_mode') == 'osnet':
                        fused_name = str(loss_detail.get('fusion_type', 'descriptor')).upper()
                        if fused_name == 'DESCRIPTOR':
                            fused_name = 'CONCAT'
                        fused_key = str(loss_detail.get('fused_branch_key', 'concat')).lower()
                        log_msg += " | OSFusion[w={:.1f}/{:.1f}/{:.1f}]".format(
                            loss_detail.get('w_mamba', 1.0),
                            loss_detail.get('w_osnet', 0.5),
                            loss_detail.get('w_concat', 1.0),
                        )
                        for name, key in (('MAMBA', 'mamba'), ('OSNET', 'osnet'), (fused_name, fused_key)):
                            log_msg += " | {}[ID={:.2f}, Tri={:.4f}]".format(
                                name,
                                loss_detail.get(f'id_{key}', 0.0),
                                loss_detail.get(f'tri_{key}', 0.0),
                            )
                        if 'w_stage3_stripe_local' in loss_detail:
                            log_msg += " | S3Local[w={:.2f}, ID={:.2f}, Tri={:.4f}]".format(
                                loss_detail.get('w_stage3_stripe_local', 0.0),
                                loss_detail.get('id_stage3_stripe_local', 0.0),
                                loss_detail.get('tri_stage3_stripe_local', 0.0),
                            )
                            if 'tri_stage3_stripe_local_guided' in loss_detail:
                                log_msg += " | GuidedTri[src={},mix={:.2f},loss={:.4f},ap/an={:.3f}/{:.3f}]".format(
                                    loss_detail.get('local_guided_triplet_source', 'main'),
                                    loss_detail.get('local_guided_triplet_mix', 0.0),
                                    loss_detail.get('tri_stage3_stripe_local_guided', 0.0),
                                    loss_detail.get('stage3_local_guided_ap', 0.0),
                                    loss_detail.get('stage3_local_guided_an', 0.0),
                                )
                        if 'role_compensation_loss' in loss_detail:
                            log_msg += (
                                " | RoleSpec[{}/{}/{} drop={:.3f}] "
                                "Comp[w={:.3f}x{:.2f},loss={:.4f},"
                                "cos={:.3f}->{:.3f},gain={:+.3f}]"
                            ).format(
                                loss_detail.get('role_branch', 'none'),
                                loss_detail.get('role_location', 'none'),
                                loss_detail.get('role_mask_type', 'none'),
                                loss_detail.get('role_drop_fraction', 0.0),
                                loss_detail.get('role_compensation_base_weight', 0.0),
                                loss_detail.get('role_compensation_scale', 0.0),
                                loss_detail.get('role_compensation_loss', 0.0),
                                loss_detail.get('role_pre_cosine', 0.0),
                                loss_detail.get('role_post_cosine', 0.0),
                                loss_detail.get('role_recovery_gain', 0.0),
                            )
                        if 'ratr' in loss_detail:
                            log_msg += " | RATR[{}]={:.4f}(intra={:.3f},inter={:.3f})".format(
                                loss_detail.get('ratr_pair', 'pair'),
                                loss_detail['ratr'],
                                loss_detail.get('ratr_intra_corr', 0.0),
                                loss_detail.get('ratr_inter_corr', 0.0),
                            )
                        if 'complementarity_mode' in loss_detail:
                            log_msg += " | Comp[{},id_w={:.2f},tri_w={:.2f},drop={}] ID={:.2f} Tri={:.4f} Pred={:.4f} Energy={:.4f} Cov={:.4f}".format(
                                loss_detail['complementarity_mode'],
                                loss_detail.get('joint_id_weight', 0.0),
                                loss_detail.get('joint_triplet_weight', 0.0),
                                loss_detail.get('dropped_branch', 'none'),
                                loss_detail.get('id_joint', 0.0),
                                loss_detail.get('tri_joint', 0.0),
                                loss_detail.get('prediction_loss', 0.0),
                                loss_detail.get('residual_energy_loss', 0.0),
                                loss_detail.get('covariance_loss', 0.0),
                            )
                        if 'peer_loss' in loss_detail:
                            peer_stats = loss_detail.get('peer_stats', {})
                            log_msg += " | PeerComp[s={:.2f},fused={:.4f},peer={:.4f},margin={:.3f}]".format(
                                loss_detail.get('peer_complement_scale', 0.0),
                                loss_detail.get('peer_fused_metric', 0.0),
                                loss_detail.get('peer_loss', 0.0),
                                peer_stats.get('fused_margin', 0.0),
                            )
                        if 'stage3_soft_balance' in loss_detail:
                            log_msg += " | SoftPart[bal={:.4f},ord={:.4f},mass={},center={}]".format(
                                loss_detail.get('stage3_soft_balance', 0.0),
                                loss_detail.get('stage3_soft_order', 0.0),
                                [round(value, 3) for value in loss_detail.get('stage3_soft_fraction', [])],
                                [round(value, 3) for value in loss_detail.get('stage3_soft_centers', [])],
                            )
                        if fdmf_stats is not None:
                            log_msg += " | FDMFStats[M_norm={:.3f}, F_norm={:.3f}, cos(M,F)={:.3f}, cos(Fused,RawCat)={:.3f}]".format(
                                fdmf_stats['mamba_norm'],
                                fdmf_stats['fdmf_norm'],
                                fdmf_stats['cos_mamba_fdmf'],
                                fdmf_stats['cos_fused_raw'],
                            )
                    else:
                        s_lambda = loss_detail.get('sfm_lambda', 0.0)
                        id_b = loss_detail.get('id_backbone', 0.0)
                        tri_b = loss_detail.get('tri_backbone', 0.0)

                        log_msg += " | SFM[λ={:.1f}]: ID_b={:.2f}, Tri_b={:.4f}".format(s_lambda, id_b, tri_b)

                        # 打印各融合分支的 Loss (不含 CS 相似度)
                        if isinstance(feat, list) and len(feat) > 1:
                            for i in range(1, len(feat)):
                                id_val = loss_detail.get(f'id_fused_{i}', loss_detail.get('id_fused', 0.0))
                                tri_val = loss_detail.get(f'tri_fused_{i}', loss_detail.get('tri_fused', 0.0))
                                log_msg += " | F{}[ID={:.2f}, Tri={:.4f}]".format(i, id_val, tri_val)

                        # RATR 损失
                        if 'ratr' in loss_detail:
                            log_msg += " | RATR={:.4f}".format(loss_detail['ratr'])
                
                logger.info(log_msg)


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        
        # Get model reference for SASF monitoring
        if hasattr(model, 'module'):
            _model = model.module
        else:
            _model = model
        
        # SASF Injection Strength Monitor (every 20 epochs)
        if epoch % 20 == 0:
            sasf_stats = []
            base_model = _model.base if hasattr(_model, 'base') else _model
            if hasattr(base_model, 'levels'):
                for lvl_idx, level in enumerate(base_model.levels):
                    if hasattr(level, 'blocks'):
                        for blk_idx, block in enumerate(level.blocks):
                            if hasattr(block, 'mixer') and hasattr(block.mixer, 'use_sasf') and block.mixer.use_sasf:
                                mixer = block.mixer
                                sf = mixer.state_fusion
                                # Collect SASF statistics (Two-Branch: dw3 + dw5)
                                stats = {
                                    'lvl': lvl_idx,
                                    'blk': blk_idx,
                                    'sasf_scale': mixer.sasf_scale.item() if hasattr(mixer, 'sasf_scale') else 0,
                                    'k3_norm': sf.dw3.weight.norm().item(),
                                    'k5_norm': sf.dw5.weight.norm().item(),
                                }
                                sasf_stats.append(stats)
            if sasf_stats:
                logger.info(f"[SASF Monitor] Epoch {epoch}:")
                logger.info(f"  {'Layer':<15} {'Scale':<8} {'K3(3x3)':<10} {'K5(5x5)':<10}")
                logger.info(f"  {'-'*43}")
                for s in sasf_stats:
                    layer_name = f"Stage{s['lvl']+1}.Blk{s['blk']}"
                    logger.info(f"  {layer_name:<15} {s['sasf_scale']:>6.4f} {s['k3_norm']:>8.2f} {s['k5_norm']:>8.2f}")

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                    if model_ema is not None:
                         torch.save(model_ema.module.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ema_{}.pth'.format(epoch)))
            else:
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                if model_ema is not None:
                         torch.save(model_ema.module.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ema_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            # Helper function for evaluation
            def run_eval(eval_model, desc="Validation"):
                eval_model.eval()
                evaluator.reset()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = eval_model(img, cam_label=camids, view_label=target_view)
                        feat = _select_eval_feature(cfg, feat)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info(f"{desc} Results - Epoch: {epoch}")
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                return cmc, mAP

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    run_eval(model, "Validation (Regular)")
                    if model_ema is not None:
                         run_eval(model_ema.module, "Validation (EMA)")
                    torch.cuda.empty_cache()
            else:
                run_eval(model, "Validation (Regular)")
                if model_ema is not None:
                        run_eval(model_ema.module, "Validation (EMA)")
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    branch_norm_betas = [
        float(beta)
        for beta in getattr(getattr(cfg, 'TEST', None), 'BRANCH_NORM_BETAS', [])
    ]
    eval_all_feats = bool(getattr(getattr(cfg, 'TEST', None), 'EVAL_ALL_FEATS', True))
    preferred_feat = str(getattr(getattr(cfg, 'TEST', None), 'FEAT_MODE', 'concat')).lower()
    weight_sweep_enabled = bool(
        getattr(getattr(cfg, 'TEST', None), 'COMPLEMENTARITY_WEIGHT_SWEEP', False)
    )
    complementarity_enabled = weight_sweep_enabled or bool(
        getattr(getattr(cfg, 'TEST', None), 'COMPLEMENTARITY_ANALYSIS', False)
    )
    
    # Collect features for all reported branches.
    all_feats = {}
    branch_norm_stats = {'mamba': [], 'osnet': []}
    all_pids = []
    all_camids = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            
            # Handle dict output (multi-branch modes) or tensor output (normal mode)
            if isinstance(feat, dict):
                if eval_all_feats:
                    for key in feat.keys():
                        all_feats.setdefault(key, [])
                        all_feats[key].append(feat[key].cpu())
                elif complementarity_enabled:
                    analysis_keys = {
                        'backbone',
                        'fdmf_only',
                        'osnet',
                        'stage3_stripe_local',
                        preferred_feat,
                    }
                    for key in analysis_keys:
                        if key in feat:
                            all_feats.setdefault(key, [])
                            all_feats[key].append(feat[key].cpu())
                else:
                    selected_feat = _select_eval_feature(cfg, feat)
                    all_feats.setdefault(preferred_feat, [])
                    all_feats[preferred_feat].append(selected_feat.cpu())

                if (eval_all_feats or complementarity_enabled) and 'backbone' in feat and 'osnet' in feat:
                    mamba_feat = feat['backbone']
                    osnet_feat = feat['osnet']
                    branch_norm_stats['mamba'].append(torch.norm(mamba_feat, p=2, dim=1).cpu())
                    branch_norm_stats['osnet'].append(torch.norm(osnet_feat, p=2, dim=1).cpu())
                    if eval_all_feats:
                        branch_norm_concat = torch.cat(
                            [
                                F.normalize(mamba_feat, p=2, dim=1),
                                F.normalize(osnet_feat, p=2, dim=1),
                            ],
                            dim=1,
                        )
                        all_feats.setdefault('branch_norm_concat', [])
                        all_feats['branch_norm_concat'].append(branch_norm_concat.cpu())
                        for beta in branch_norm_betas:
                            beta_key = 'weighted_branch_norm_concat_b{:03d}'.format(int(round(beta * 100)))
                            weighted_branch_norm_concat = torch.cat(
                                [
                                    F.normalize(mamba_feat, p=2, dim=1),
                                    beta * F.normalize(osnet_feat, p=2, dim=1),
                                ],
                                dim=1,
                            )
                            all_feats.setdefault(beta_key, [])
                            all_feats[beta_key].append(weighted_branch_norm_concat.cpu())
            else:
                # Normal mode: only one descriptor is available.
                all_feats.setdefault('concat', [])
                all_feats['concat'].append(feat.cpu())
            
            all_pids.extend(pid.tolist() if hasattr(pid, 'tolist') else pid)
            all_camids.extend(camid.tolist() if hasattr(camid, 'tolist') else camid)
            img_path_list.extend(imgpath)

    # Convert to numpy arrays for evaluation
    import numpy as np
    all_pids = np.array(all_pids)
    all_camids = np.array(all_camids)

    feature_tensors = {
        name: torch.cat(feature_list, dim=0)
        for name, feature_list in all_feats.items()
        if feature_list
    }
    complementarity_report = None
    if complementarity_enabled:
        complementarity_report = analyze_feature_complementarity(
            feature_tensors,
            num_query=num_query,
            topk=int(getattr(cfg.TEST, 'COMPLEMENTARITY_TOPK', 10)),
            cka_samples=int(getattr(cfg.TEST, 'COMPLEMENTARITY_CKA_SAMPLES', 2048)),
            query_samples=int(getattr(cfg.TEST, 'COMPLEMENTARITY_QUERY_SAMPLES', 256)),
            gallery_samples=int(getattr(cfg.TEST, 'COMPLEMENTARITY_GALLERY_SAMPLES', 4096)),
            seed=int(getattr(cfg.SOLVER, 'SEED', 42)),
        )
        if not weight_sweep_enabled:
            feature_tensors.update(build_branch_normalized_descriptors(feature_tensors))

    if branch_norm_stats['mamba'] and branch_norm_stats['osnet']:
        mamba_norm = torch.cat(branch_norm_stats['mamba'], dim=0)
        osnet_norm = torch.cat(branch_norm_stats['osnet'], dim=0)
        norm_ratio = mamba_norm / osnet_norm.clamp_min(1e-12)
        logger.info(
            "Feature norm stats - Mamba: mean={:.4f}, std={:.4f}; OSNet: mean={:.4f}, std={:.4f}; M/O ratio: mean={:.4f}, std={:.4f}".format(
                mamba_norm.mean().item(),
                mamba_norm.std(unbiased=False).item(),
                osnet_norm.mean().item(),
                osnet_norm.std(unbiased=False).item(),
                norm_ratio.mean().item(),
                norm_ratio.std(unbiased=False).item(),
            )
        )
    
    # Evaluate each feature type
    results = {}
    def evaluate_descriptor(feat_name, feats, verbose=True):
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()

        evaluator.update((feats, all_pids, all_camids))

        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        result = {'cmc': cmc, 'mAP': mAP}
        if verbose:
            logger.info(f"=== {feat_name.upper()} Results ===")
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return result

    for feat_name, feats in feature_tensors.items():
        results[feat_name] = evaluate_descriptor(feat_name, feats)

    weight_metadata = {}
    if weight_sweep_enabled:
        for feat_name, feats, weights in iter_weighted_branch_descriptors(
            feature_tensors,
            fdmf_weights=list(getattr(cfg.TEST, 'COMPLEMENTARITY_FDMF_WEIGHTS', [])),
            osnet_weights=list(getattr(cfg.TEST, 'COMPLEMENTARITY_OSNET_WEIGHTS', [])),
            local_weights=list(getattr(cfg.TEST, 'COMPLEMENTARITY_LOCAL_WEIGHTS', [])),
        ):
            weight_metadata[feat_name] = weights
            results[feat_name] = evaluate_descriptor(feat_name, feats, verbose=False)
            result = results[feat_name]
            logger.info(
                'WeightSweep {} mAP={:.2f}% R1={:.2f}%'.format(
                    feat_name,
                    100.0 * result['mAP'],
                    100.0 * result['cmc'][0],
                )
            )

    if complementarity_report is not None:
        base_results = {
            name: result
            for name, result in results.items()
            if not name.startswith('ws_')
        }
        attach_retrieval_results(complementarity_report, base_results)
        if weight_sweep_enabled:
            attach_weight_sweep_results(complementarity_report, weight_metadata, results)
            weighted = complementarity_report.get('weight_sweep', {})
            if weighted:
                best_map_name, best_map = max(weighted.items(), key=lambda item: item[1]['mAP'])
                best_r1_name, best_r1 = max(weighted.items(), key=lambda item: item[1]['R1'])
                logger.info(
                    'Best weight-sweep mAP: {} mAP={:.2f}% R1={:.2f}%'.format(
                        best_map_name,
                        100.0 * best_map['mAP'],
                        100.0 * best_map['R1'],
                    )
                )
                logger.info(
                    'Best weight-sweep R1: {} mAP={:.2f}% R1={:.2f}%'.format(
                        best_r1_name,
                        100.0 * best_r1['mAP'],
                        100.0 * best_r1['R1'],
                    )
                )
        log_feature_complementarity(logger, complementarity_report)
        output_name = str(
            getattr(cfg.TEST, 'COMPLEMENTARITY_OUTPUT', 'feature_complementarity.json')
        )
        output_path = output_name if os.path.isabs(output_name) else os.path.join(
            cfg.OUTPUT_DIR,
            output_name,
        )
        save_feature_complementarity(complementarity_report, output_path)
        logger.info('Complementarity report saved to {}'.format(output_path))
    
    for key in (preferred_feat, 'fdmf', 'concat', 'raw_concat', 'backbone'):
        if key in results:
            return results[key]['cmc'][0], results[key]['cmc'][4]
    return 0.0, 0.0


