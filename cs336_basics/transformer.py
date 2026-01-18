import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    ex = torch.exp(x)
    x = ex / torch.sum(ex, dim=dim, keepdim=True)
    return x


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    if len(K.shape) == 3:
        scores = einsum(Q, K, "n s_q d, n s_k d -> n s_q s_k") * d_k ** -0.5
    else:
        scores = einsum(Q, K, "n h s_q d, n h s_k d -> n h s_q s_k") * d_k ** -0.5
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    return attn @ V
    

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
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # inv_freq[j] = theta^(-2j/d)
        j = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        inv_freq = self.theta ** (-2.0 * j / d_k)   # (B,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, token_positions):
        # x: (N,S,d), token_positions: (N,S)
        N, S, d = x.shape
        assert d == self.d_k and S <= self.max_seq_len

        if token_positions.ndim == 1:          # (S,)
            token_positions = token_positions.unsqueeze(0) # -> (1,S)

        x_blk = x.reshape(N, S, d // 2, 2)  # (N,S,B,2) = [even, odd]

        # angles: (N,S,B)
        pos = token_positions.to(self.inv_freq.dtype)      # (N, S)
        angles = rearrange(pos, "n s -> n s 1") * rearrange(self.inv_freq, "b -> 1 1 b")

        c = torch.cos(angles)  # (N,S,B)
        s = torch.sin(angles)  # (N,S,B)

        x0 = x_blk[..., 0]     # (N,S,B)
        x1 = x_blk[..., 1]     # (N,S,B)

        # rotate
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c

        out = rearrange(torch.stack([y0, y1], dim=-1), "n s b r -> n s (b r)")
        return out