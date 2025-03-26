import torch as pt
from torch import nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, query_dim, value_dim, num_heads, dropout=0.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.qk = nn.Linear(embedding_dim, 2 * query_dim * num_heads)
        self.v = nn.Linear(embedding_dim, value_dim * num_heads)
        self.out_proj = nn.Linear(value_dim * num_heads, embedding_dim)
        self.dropout = dropout

    def forward(self, x, attn_mask=None):
        batch_shape = x.shape[:-2]
        L = x.shape[-2]
        q, k = self.qk(x).view(*batch_shape, L, self.num_heads, 2 * self.query_dim).transpose(-2, -3).chunk(2, dim=-1)
        v = self.v(x).view(*batch_shape, L, self.num_heads, self.value_dim).transpose(-2, -3)

        if attn_mask is not None and attn_mask.dim() < x.dim(): # handle same mask for all heads
            attn_mask = attn_mask.unsqueeze(-3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=(self.dropout if self.training else 0.))

        return self.out_proj(out.transpose(-2, -3).flatten(-2))


class T5MHSA(MultiheadSelfAttention):
    def __init__(self, bins, max_distance, embedding_dim, query_dim, value_dim, num_heads, dropout=0., causal=False):
        super().__init__(embedding_dim, query_dim, value_dim, num_heads, dropout)
        self.bins = bins
        self.max_distance = max_distance
        self.causal = causal

        self.bias = nn.Parameter(pt.zeros(bins if causal else 2 * bins - 1, num_heads))

        n = pt.arange(max_distance + 1)
        m = bins // 2
        indices = pt.clamp(n, max=m) + (pt.log(pt.clamp(n, min=m) - m + 1) * (bins - 1 - m) / pt.tensor(max_distance - m + 1).log()).long()
        self.register_buffer("indices", indices, persistent=False)

    def forward(self, x, attn_mask=None):
        L = x.shape[-2]

        with pt.no_grad():
            shifts = pt.arange(L, device=x.device).unsqueeze(1) - pt.arange(L, device=x.device).unsqueeze(0)
            bins_idx = self.indices[pt.clamp(pt.abs(shifts), max=self.max_distance)] + (0 if self.causal else (self.bins - 1) * pt.signbit(shifts))
        
        bias = self.bias[bins_idx].permute(2, 0, 1)
        if attn_mask is not None:
            assert attn_mask.dtype == pt.bool
            bias.masked_fill_(~attn_mask.unsqueeze(-3), float("-inf"))
        
        if self.causal:
            bias.masked_fill_(pt.signbit(shifts), float("-inf"))

        return super().forward(x, bias)


class RoPEMHSA(MultiheadSelfAttention):
    pass
    # TODO
