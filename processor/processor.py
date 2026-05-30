import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch import amp
import torch.distributed as dist

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

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
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
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast('cuda'):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            
            # Loss 计算在 FP32 下进行，避免 FP16 溢出导致 NaN
            loss_result = loss_fn(score, feat, target, target_cam)
            if isinstance(loss_result, tuple):
                loss, loss_detail = loss_result
            else:
                loss, loss_detail = loss_result, None

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
            
            # SFM 模式 (score 是 list 时，使用 backbone 分支计算准确率)
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # 基础日志
                log_msg = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                    epoch, (n_iter + 1), len(train_loader),
                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0])
                
                # SFM模式：追加各分支loss详情 (精简版)
                if loss_detail is not None:
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

                raw_model = model.module if hasattr(model, "module") else model
                if hasattr(raw_model, "sdm_last_stats"):
                    st = raw_model.sdm_last_stats
                    log_msg += (
                        " | SDM: stab_mean={:.3f}, var={:.5f}, min={:.3f}, max={:.3f}, freq_w={:.3f}"
                        .format(
                            st["mean"].item(),
                            st["var"].item(),
                            st["min"].item(),
                            st["max"].item(),
                            st["freq_w"].item(),
                        )
                    )
                
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
                        # Handle dict output (SFM mode) - use concat for training validation
                        if isinstance(feat, dict):
                            feat = feat['concat']
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
    
    # Collect features for all branches
    all_feats = {'backbone': [], 'fused': [], 'concat': []}
    all_pids = []
    all_camids = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            
            # Handle dict output (SFM mode) or tensor output (normal mode)
            if isinstance(feat, dict):
                for key in all_feats.keys():
                    all_feats[key].append(feat[key].cpu())
            else:
                # Normal mode: only concat available
                all_feats['concat'].append(feat.cpu())
            
            all_pids.extend(pid.tolist() if hasattr(pid, 'tolist') else pid)
            all_camids.extend(camid.tolist() if hasattr(camid, 'tolist') else camid)
            img_path_list.extend(imgpath)

    # Convert to numpy arrays for evaluation
    import numpy as np
    all_pids = np.array(all_pids)
    all_camids = np.array(all_camids)
    
    # Evaluate each feature type
    results = {}
    for feat_name, feat_list in all_feats.items():
        if len(feat_list) == 0:
            continue
            
        feats = torch.cat(feat_list, dim=0)
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
        
        # Update evaluator with all features at once
        for i in range(len(all_pids)):
            evaluator.update((feats[i:i+1], [all_pids[i]], [all_camids[i]]))
        
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        results[feat_name] = {'cmc': cmc, 'mAP': mAP}
        
        logger.info(f"=== {feat_name.upper()} Results ===")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    # Return concat results for backward compatibility
    if 'concat' in results:
        return results['concat']['cmc'][0], results['concat']['cmc'][4]
    else:
        return 0.0, 0.0


