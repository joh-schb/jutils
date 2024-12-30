""" Adapted from
https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/optimization.py
"""
import math
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_iter_exponential_schedule(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, final_ratio: float):
    """
    Iteration-wise exponential scheduler.

    Args:
        num_warmup_steps (int): Number of warmup steps
        num_training_steps (int): Total number of training steps
        final_ratio (float): Expected LR ratio at n_iter = num_training_steps
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        elif current_step >= num_training_steps:
            return final_ratio
        else:
            actual_iter = current_step - num_warmup_steps
            alpha = np.exp(
                actual_iter / (num_training_steps - num_warmup_steps) * np.log(final_ratio)
            )
            return alpha
    
    return LambdaLR(optimizer, lr_lambda)


def get_exponential_decay_schedule(optimizer: Optimizer, num_warmup_steps: int, t_decay: int):
    """
    Warmup-Stable-Decay (WSD) learning rate schedule according to the paper:
    'MiniCPM: Unveiling the Potential of Small Language Models with Scalable
    Training Strategies' - Hu et al. (2024)

    Max steps in WSD annealing phase: 3 * t_decay
    
    Args:
        num_warmup_steps (int): Number of warmup steps
        t_decay (int): Step at which the learning rate is halved (set to ~2% of the training steps)
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        actual_iter = current_step - num_warmup_steps
        return 0.5 ** ((actual_iter) / t_decay)

    return LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    base_lr = 1e-4
    warmup_steps = 1000
    training_steps = 10000

    schedules = {
        get_constant_schedule_with_warmup: dict(num_warmup_steps=warmup_steps),
        get_cosine_schedule_with_warmup: dict(num_warmup_steps=warmup_steps, num_training_steps=training_steps),
        get_iter_exponential_schedule: dict(num_warmup_steps=warmup_steps, num_training_steps=training_steps, final_ratio=0.01),
        get_exponential_decay_schedule: dict(num_warmup_steps=warmup_steps, t_decay=1000)
    }

    fig = plt.figure(figsize=(10, 6))
    for schedule, kwargs in schedules.items():
        opt = torch.optim.Adam(torch.nn.Linear(10, 10).parameters(), lr=base_lr)
        scheduler = schedule(opt, **kwargs)
        lrs = []
        for i in tqdm(range(training_steps)):
            scheduler.step()
            lrs.append(get_lr(opt))
        kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        lbl = f"{schedule.__name__}({kwargs_str})"
        plt.plot(lrs, label=lbl)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
    fig.savefig('lr-schedulers.png', bbox_inches='tight')
