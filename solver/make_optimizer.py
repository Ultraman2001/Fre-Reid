import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    # Track printed keys to avoid excessive logging
    printed_high_lr_keys = set()
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        
        # Respect _no_weight_decay attribute set by Mamba SSM components (A_log, D)
        # This aligns with the official MambaVision design intent
        if getattr(value, "_no_weight_decay", False):
            weight_decay = 0.0
            print(f"[No Weight Decay] {key}")
        
        # High Learning Rate Strategy for Randomly Initialized Modules
        # Priority: SASF modules first (independent LR), then Stage 4 backbone
        # 1. state_fusion / sasf_scale: StateFusion modules (uses SASF_LR_FACTOR)
        # 2. levels.3 (Stage 4 backbone): Random init due to structure change
        
        if "fslora" in key:
            fslora_lr_factor = getattr(cfg.SOLVER, 'FSLORA_LR_FACTOR', 2.0)
            lr = cfg.SOLVER.BASE_LR * fslora_lr_factor
            if "fslora" not in printed_high_lr_keys:
                print(f"Using {fslora_lr_factor}x learning rate for FSLoRA adapter: {key}")
                printed_high_lr_keys.add("fslora")

        elif key.startswith("osnet."):
            osnet_lr_factor = getattr(cfg.SOLVER, 'OSNET_LR_FACTOR', 1.0)
            lr = cfg.SOLVER.BASE_LR * osnet_lr_factor
            if "osnet" not in printed_high_lr_keys:
                print(f"Using {osnet_lr_factor}x learning rate for OSNet branch: {key}")
                printed_high_lr_keys.add("osnet")

        elif (
            key.startswith("osnet_bottleneck")
            or key.startswith("osnet_classifier")
            or key.startswith("fusion_bottleneck")
            or key.startswith("fusion_classifier")
            or key.startswith("stage2_fcu")
            or key.startswith("stage3_fcu")
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
            # Scale the *current* LR (whether base or high) by the bias factor
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        
        # fusion_scale is a gate scalar, should not have weight decay
        # and needs higher LR to learn effectively
        if "fusion_scale" in key:
            weight_decay = 0.0
            lr = cfg.SOLVER.BASE_LR * 3.0
            print(f"Using 3x LR and no weight decay for fusion_scale: {key}")
            
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        # SFM 3.0 Hierarchical Fusion Learning Rate
        # 1. SFM Blocks inside backbone (Typical: 0.5x LR)
        if "sfm_s" in key:
            sfm_lr_factor = getattr(cfg.SOLVER, 'SFM_LR_FACTOR', 0.5)
            lr = cfg.SOLVER.BASE_LR * sfm_lr_factor
            if "sfm_blocks" not in printed_high_lr_keys:
                print(f"Using {sfm_lr_factor}x learning rate for SFM Blocks: {key}")
                printed_high_lr_keys.add("sfm_blocks")
        
        # 2. SFM Deep Supervision Heads (2.0x LR for randomly initialized new layers)
        # Only match the exact head names to avoid false positives
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

