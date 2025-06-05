# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_flood(optimizer, current_iter, max_iters, args):
    """Adjust learning rate based on polynomial decay with warmup."""
    initial_lr = args.lr
    warmup_iters = args.accum_iter
    warmup_ratio = 1e-6
    min_lr = args.min_lr
    power = 1.0
    
    if current_iter < warmup_iters:
        lr = warmup_ratio * initial_lr + (initial_lr - warmup_ratio * initial_lr) * (current_iter / warmup_iters)
    # else:
    #     decay_iter = current_iter - warmup_iters
    #     decay_total = max_iters - warmup_iters
    #     lr = (initial_lr - min_lr) * ((1 - decay_iter / decay_total) ** power) + min_lr
    else:
        lr = initial_lr * (0.99 ** (current_iter // 1000))
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    return lr
