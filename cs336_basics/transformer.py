import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat


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
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        results = x / rms * self.g
        return results.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)

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
        j = torch.arange(0, d_k // 2, device=device, dtype=dtype)
        inv_freq = self.theta ** (-2.0 * j / d_k)   # (B,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, token_positions):
        # x: (N,S,d), token_positions: (N,S)
        N, S, d = x.shape
        assert d == self.d_k and S <= self.max_seq_len

        if token_positions.ndim == 1:          # (S,)
            token_positions = rearrange(token_positions, "s -> 1 s") # -> (1,S)

        x_blk = rearrange(x, "n s (d2 two) -> n s d2 two", two=2)  # (N,S,B,2) = [even, odd]

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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d = d_model // num_heads

        self.W_Q = Linear(d_model, d_model, device, dtype)
        self.W_K = Linear(d_model, d_model, device, dtype)
        self.W_V = Linear(d_model, d_model, device, dtype)
        self.W_O = Linear(d_model, d_model, device, dtype)

        self.rope = RotaryPositionalEmbedding(theta, self.d, max_seq_len, device, dtype) if (theta is not None and max_seq_len is not None) else None

    def forward(self, x, token_positions=None):
        n, s, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = rearrange(Q, "n s (h d) -> n h s d", h=self.num_heads)
        K = rearrange(K, "n s (h d) -> n h s d", h=self.num_heads)
        V = rearrange(V, "n s (h d) -> n h s d", h=self.num_heads)

        mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=x.device))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2])
            Qf = rearrange(Q, "n h s d -> (n h) s d")
            Kf = rearrange(K, "n h s d -> (n h) s d")

            if token_positions.ndim == 1:      # (s,)
                token_positions = rearrange(token_positions, 's -> 1 s')
            if token_positions.shape[0] == 1:  # now (1, s)
                token_positions = repeat(token_positions, '1 s -> n s', n=n)
            pos = repeat(token_positions, "n s -> (n h) s", h=self.num_heads)  # (n*h, s)

            Q = rearrange(self.rope(Qf, pos), "(n h) s d -> n h s d", n=n, h=self.num_heads)
            K = rearrange(self.rope(Kf, pos), "(n h) s d -> n h s d", n=n, h=self.num_heads)

        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "n h s d -> n s (h d)")
        return self.W_O(out)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model, eps, device, dtype)
        self.rms_norm2 = RMSNorm(d_model, eps, device, dtype)
        self.mha = MultiHeadAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ff = SwiGLU(d_model, d_ff, device, dtype)
    
    def forward(self, x, token_positions=None):
        mha_out = x + self.mha(self.rms_norm1(x), token_positions)
        ff_out = mha_out + self.ff(self.rms_norm2(mha_out))
        return ff_out


class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            context_length,
            num_layers,
            d_model,
            num_heads,
            d_ff,
            theta=None,
            eps=1e-5,
            device=None,
            dtype=None
        ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList(
            [
                Block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta, max_seq_len=context_length, eps=eps, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.final_linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        x = self.final_norm(x)
        x = self.final_linear(x)
        return x