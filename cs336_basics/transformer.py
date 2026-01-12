import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  x @ self.W.T
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids] 


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.g = nn.Parameter(torch.empty(d_model, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.g)
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        results = x / rms * self.g
        return results.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
        self.linear1 = Linear(d_ff, d_model)
        self.linear2 = Linear(d_model, d_ff)
        self.linear3 = Linear(d_ff, d_model)

    def _silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x1 = self._silu(self.linear1(x))
        x = x1 * self.linear3(x)
        x = self.linear2(x)
        return x