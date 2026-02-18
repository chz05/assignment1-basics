from collections.abc import Callable
from typing import Optional
import math

import torch

class AdamW(torch.optim.Optimizer):
    """A minimal AdamW optimizer implementation."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        self.defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, self.defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # Adam first/second moment updates
                # m = beta1 * m + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = beta2 * v + (1 - beta2) * grad**2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected learning rate
                # lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Parameter update
                # w = w - lr_t * m / (sqrt(v) + eps)
                p.data.addcdiv_(exp_avg, torch.sqrt(exp_avg_sq) + eps, value=-lr_t)

                # Decoupled weight decay
                if weight_decay != 0:
                    # w = w * (1 - lr * weight_decay)
                    p.data.mul_(1 - lr * weight_decay)

        return loss