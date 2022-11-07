import math
from config import cfg
import numpy as np


def cos_warmup_lr(steps_per_epoch):
    total_steps = cfg.MODEL.MAX_EPOCH * steps_per_epoch
    global_step = 0
    while global_step < total_steps:
        cur_epoch = math.floor(global_step / steps_per_epoch)
        global_step += 1
        yield get_epoch_lr(cur_epoch)


def lr_fun_cos(cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.MODEL.MAX_EPOCH))
    return (1.0 - cfg.MODEL.MIN_LR) * lr + cfg.MODEL.MIN_LR


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch according to the policy."""
    # Get lr and scale by by BASE_LR
    lr = lr_fun_cos(cur_epoch) * cfg.MODEL.INITIAL_LR
    # Linear warmup
    if cur_epoch < cfg.MODEL.WARM_UP_EPOCH:
        alpha = cur_epoch / cfg.MODEL.WARM_UP_EPOCH
        warmup_factor = cfg.MODEL.WARM_UP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr