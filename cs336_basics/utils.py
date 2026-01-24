import torch


def cross_entropy(pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_logits = pred_logits - pred_logits.max(dim=-1, keepdim=True).values
    denom = torch.logsumexp(pred_logits, dim=-1, keepdim=True)
    log_p = pred_logits - denom
    neg_log_p = -log_p.gather(dim=1, index=targets.unsqueeze(1))

    return torch.mean(neg_log_p)