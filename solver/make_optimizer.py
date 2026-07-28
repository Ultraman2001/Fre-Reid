import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    printed_high_lr_keys = set()
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        
        if getattr(value, "_no_weight_decay", False):
            weight_decay = 0.0
            print(f"[No Weight Decay] {key}")

        if "fslora" in key:
            fslora_lr_factor = getattr(cfg.SOLVER, 'FSLORA_LR_FACTOR', 2.0)
            lr = cfg.SOLVER.BASE_LR * fslora_lr_factor
            if "fslora" not in printed_high_lr_keys:
                print(f"Using {fslora_lr_factor}x learning rate for FSLoRA adapter: {key}")
                printed_high_lr_keys.add("fslora")

        elif key.startswith("osnet."):
            osnet_lr_factor = getattr(cfg.SOLVER, 'OSNET_LR_FACTOR', 1.0)
            lr = cfg.SOLVER.BASE_LR * osnet_lr_factor
            osnet_weight_decay = float(getattr(cfg.SOLVER, 'OSNET_WEIGHT_DECAY', -1.0))
            if osnet_weight_decay >= 0.0:
                weight_decay = osnet_weight_decay
            if "osnet" not in printed_high_lr_keys:
                print(
                    f"Using {osnet_lr_factor}x learning rate and weight_decay={weight_decay} "
                    f"for OSNet branch: {key}"
                )
                printed_high_lr_keys.add("osnet")

        elif (
            key.startswith("stage1_fcu")
            or key.startswith("stage2_fcu")
            or key.startswith("stage3_fcu")
        ):
            legacy_factor = getattr(cfg.SOLVER, 'OSNET_FUSION_LR_FACTOR', 2.0)
            fcu_lr_factor = float(getattr(cfg.SOLVER, 'FCU_LR_FACTOR', -1.0))
            if fcu_lr_factor < 0.0:
                fcu_lr_factor = legacy_factor
            lr = cfg.SOLVER.BASE_LR * fcu_lr_factor
            if "fcu" not in printed_high_lr_keys:
                print(f"Using {fcu_lr_factor}x learning rate for FCU modules: {key}")
                printed_high_lr_keys.add("fcu")

        elif key.startswith("fdmf_refiner"):
            legacy_factor = getattr(cfg.SOLVER, 'OSNET_FUSION_LR_FACTOR', 2.0)
            fdmf_lr_factor = float(getattr(cfg.SOLVER, 'FDMF_LR_FACTOR', -1.0))
            if fdmf_lr_factor < 0.0:
                fdmf_lr_factor = legacy_factor
            lr = cfg.SOLVER.BASE_LR * fdmf_lr_factor
            if "fdmf_refiner" not in printed_high_lr_keys:
                print(f"Using {fdmf_lr_factor}x learning rate for FDMF refiner: {key}")
                printed_high_lr_keys.add("fdmf_refiner")

        elif (
            key.startswith("osnet_bottleneck")
            or key.startswith("osnet_classifier")
            or key.startswith("fusion_bottleneck")
            or key.startswith("fusion_classifier")
            or key.startswith("stage3_stripe_local")
        ):
            fusion_lr_factor = getattr(cfg.SOLVER, 'OSNET_FUSION_LR_FACTOR', 2.0)
            lr = cfg.SOLVER.BASE_LR * fusion_lr_factor
            if "osnet_fusion_heads" not in printed_high_lr_keys:
                print(f"Using {fusion_lr_factor}x learning rate for OSNet fusion heads: {key}")
                printed_high_lr_keys.add("osnet_fusion_heads")

        elif "state_fusion" in key or "sasf_scale" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.SASF_LR_FACTOR
            if "state_fusion" not in printed_high_lr_keys:
                print(f"Using {cfg.SOLVER.SASF_LR_FACTOR}x learning rate for StateFusion: {key}")
                printed_high_lr_keys.add("state_fusion")
                
        elif "levels.3" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.STAGE4_LR_FACTOR
            if "levels.3" not in printed_high_lr_keys:
                print(f"Using {cfg.SOLVER.STAGE4_LR_FACTOR}x learning rate for Stage 4 (levels.3): {key}")
                printed_high_lr_keys.add("levels.3")

        if "bias" in key:
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if key.startswith("osnet."):
                osnet_bias_weight_decay = float(
                    getattr(cfg.SOLVER, 'OSNET_WEIGHT_DECAY_BIAS', -1.0)
                )
                if osnet_bias_weight_decay >= 0.0:
                    weight_decay = osnet_bias_weight_decay

        if "fusion_scale" in key:
            weight_decay = 0.0
            if key.startswith(("stage1_fcu", "stage2_fcu", "stage3_fcu")):
                if "fcu_scale" not in printed_high_lr_keys:
                    print(f"Using FCU learning rate and no weight decay for FCU fusion scale: {key}")
                    printed_high_lr_keys.add("fcu_scale")
            else:
                fusion_lr_factor = getattr(cfg.SOLVER, 'OSNET_FUSION_LR_FACTOR', 2.0)
                lr = cfg.SOLVER.BASE_LR * fusion_lr_factor
                if "fusion_scale" not in printed_high_lr_keys:
                    print(f"Using {fusion_lr_factor}x LR and no weight decay for fusion_scale: {key}")
                    printed_high_lr_keys.add("fusion_scale")

        if cfg.SOLVER.LARGE_FC_LR and ("classifier" in key or "arcface" in key):
            lr = cfg.SOLVER.BASE_LR * 2
            print('Using two times learning rate for fc ')

        if "sfm_s" in key:
            sfm_lr_factor = getattr(cfg.SOLVER, 'SFM_LR_FACTOR', 0.5)
            lr = cfg.SOLVER.BASE_LR * sfm_lr_factor
            if "sfm_blocks" not in printed_high_lr_keys:
                print(f"Using {sfm_lr_factor}x learning rate for SFM Blocks: {key}")
                printed_high_lr_keys.add("sfm_blocks")
        
        elif "pooling_fused" in key or "bottleneck_fused" in key or "classifier_fused" in key:
            lr = cfg.SOLVER.BASE_LR * 2.0
            if "sfm_heads" not in printed_high_lr_keys:
                print(f"Using 2.0x learning rate for SFM Heads: {key}")
                printed_high_lr_keys.add("sfm_heads")

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center

