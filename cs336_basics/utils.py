import torch
from collections.abc import Callable
from typing import Optional
import math


def cross_entropy(pred_logits:
     torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_logits = pred_logits - pred_logits.max(dim=-1, keepdim=True).values
    denom = torch.logsumexp(pred_logits, dim=-1, keepdim=True)
    log_p = pred_logits - denom
    neg_log_p = -log_p.gather(dim=1, index=targets.unsqueeze(1))

    return torch.mean(neg_log_p)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                grad = p.grad.data
                updated_m = beta1 * m + (1 - beta1) * grad
                updated_v = beta2 * v + (1 - beta2) * grad ** 2
                updated_lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data -= updated_lr * updated_m / (torch.sqrt(updated_v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = updated_m
                state["v"] = updated_v

        return loss


def learning_rate_schedule(t, alpha_max, alpha_min, t_w, t_c):
    if t < t_w:
        return t / t_w * alpha_max
    elif t >= t_w and t <= t_c:
        return alpha_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min


def gradient_clipping():
    pass


