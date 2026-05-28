""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    
    # Aggressive Schedule for High LR Training:
    # - Higher lr_min (2%): Prevents underfitting in final epochs
    # - warmup from 1% to peak over WARMUP_EPOCHS
    lr_min = 0.02 * cfg.SOLVER.BASE_LR          # 2% of peak (was 0.2%)
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR  # Start at 1% of peak

    warmup_t = cfg.SOLVER.WARMUP_EPOCHS  # Now configurable via YAML (recommend 25)
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler
