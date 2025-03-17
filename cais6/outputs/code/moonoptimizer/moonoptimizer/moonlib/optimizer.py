import torch
from torch.optim import Optimizer
from typing import Dict, List, Tuple
import torch.nn as nn
import numpy as np

class DARMSprop(Optimizer):
    """
    Implements Directional-Adaptive RMSprop optimizer.
    """

    def __init__(self, params, lr: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0):
        """
        Initialize the DA-RMSprop optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            beta_1 (float, optional): Coefficient used for computing running averages of gradient. Defaults to 0.9.
            beta_2 (float, optional): Coefficient used for computing running averages of squared gradient. Defaults to 0.999.
            eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta_1 parameter: {}".format(beta_1))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta_2 parameter: {}".format(beta_2))
        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, eps=eps, weight_decay=weight_decay)
        super(DARMSprop, self).__init__(params, defaults)

        # Initialize online hessian approximation parameters (example)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Directional Momentum
                state['moving_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns: the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('DA-RMSprop does not support sparse gradients')

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta_1, beta_2 = group['beta_1'], group['beta_2']
                state['moving_avg_grad'] = beta_1 * state['moving_avg_grad'] + (1 - beta_1) * grad

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                exp_avg_sq.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Directional Momentum using Gradient Alignment (DM-MAGA)
                alpha_t = torch.nn.functional.cosine_similarity(grad.flatten(), state['moving_avg_grad'].flatten(), dim=0)
                alpha_t = torch.clamp(alpha_t, -1, 1)
                lambda_ = 1.0 # Tunable parameter, can be set in the config.yaml or as an argument to the optimizer
                p.data.addcdiv_(grad, denom, value=-group['lr'] * (1 + lambda_ * alpha_t)) # Removed the Hessian for stability.

        return loss