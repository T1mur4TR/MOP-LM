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
        """
        Args:
            x (torch.Tensor): input tensor with shape [..., L, E]
            attn_mask (Optional[torch.Tensor]): boolean (float) attention mask (bias) with shape [..., L, L] or [..., H, L, L]

        Returns:
            torch.Tensor: output tensor with shape [..., L, E]
        """
        batch_shape = x.shape[:-2]
        L = x.shape[-2]
        q, k = self.qk(x).view(*batch_shape, L, self.num_heads, 2 * self.query_dim).transpose(-2, -3).chunk(2, dim=-1)
        v = self.v(x).view(*batch_shape, L, self.num_heads, self.value_dim).transpose(-2, -3)

        if attn_mask is not None and attn_mask.dim() <= x.dim(): # handle same mask for all heads
            attn_mask = attn_mask.unsqueeze(-3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=(self.dropout if self.training else 0.))

        return self.out_proj(out.transpose(-2, -3).flatten(-2))
    

class MultiheadCrossAttention(nn.Module):
    def __init__(self, embedding_dim, query_dim, value_dim, num_heads, context_embedding_dim=None, dropout=0.):
        super().__init__()
        if context_embedding_dim is None:
            context_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.context_embedding_dim = context_embedding_dim
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.q = nn.Linear(embedding_dim, query_dim * num_heads)
        self.k = nn.Linear(context_embedding_dim, query_dim * num_heads)
        self.v = nn.Linear(context_embedding_dim, value_dim * num_heads)
        self.out_proj = nn.Linear(value_dim * num_heads, embedding_dim)
        self.dropout = dropout

    def forward(self, x, y, attn_mask=None):
        """
        Args:
            x (torch.Tensor): input tensor with shape [..., L, E]
            y (torch.Tensor): context tensor with shape [..., S, Ec]
            attn_mask (Optional[torch.Tensor]): boolean (float) attention mask (bias) with shape [..., L, S] or [..., H, L, S]

        Returns:
            torch.Tensor: output tensor with shape [..., L, E]
        """
        batch_shape = x.shape[:-2]
        L, S = x.shape[-2], y.shape[-2]
        q = self.q(x).view(*batch_shape, L, self.num_heads, self.query_dim).transpose(-2, -3)
        k = self.k(y).view(*batch_shape, S, self.num_heads, self.query_dim).transpose(-2, -3)
        v = self.v(y).view(*batch_shape, S, self.num_heads, self.value_dim).transpose(-2, -3)

        if attn_mask is not None and attn_mask.dim() <= x.dim(): # handle same mask for all heads
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
        """
        Args:
            x (torch.Tensor): input tensor with shape [..., L, E]
            attn_mask (Optional[torch.Tensor]): boolean attention mask with shape [..., L, L]

        Returns:
            torch.Tensor: output tensor with shape [..., L, E]
        """
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


class RoPE(nn.Module):
    def __init__(self, dim, cache_size=512, base=10000):
        super().__init__()

        assert dim % 2 == 0

        self.dim = dim
        self.cache_size = cache_size
        self.base = base

        theta = self.base ** -(pt.arange(0, self.dim, 2) / self.dim)
        self.register_buffer("theta", theta, persistent=False)

        cache = self._compute_rotations(0, cache_size)
        self.register_buffer("cache", cache, persistent=False)

    def _compute_rotations(self, start_idx, end_idx):
        angles = pt.arange(start_idx, end_idx, device=self.theta.device, dtype=self.theta.dtype).outer(self.theta)
        return pt.stack([pt.cos(angles), pt.sin(angles)], dim=-1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor with shape [..., L, E]

        Returns:
            torch.Tensor: output tensor with shape [..., L, E]
        """
        L = x.shape[-2]

        if L <= self.cache_size:
            rotations = self.cache[:L]
        else:
            rotations = pt.cat([self.cache, self._compute_rotations(self.cache_size, L)])
        
        xshaped = x.reshape(*x.shape[:-1], -1, 2)

        return pt.stack([
            xshaped[..., 0] * rotations[..., 0] - xshaped[..., 1] * rotations[..., 1], 
            xshaped[..., 1] * rotations[..., 0] + xshaped[..., 0] * rotations[..., 1]
        ], dim=-1).flatten(-2)


class RoPEMHSA(MultiheadSelfAttention):
    def __init__(self, embedding_dim, query_dim, value_dim, num_heads, dropout=0, cache_size=512, base=10000):
        super().__init__(embedding_dim, query_dim, value_dim, num_heads, dropout)

        self.RoPE = RoPE(query_dim, cache_size, base)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x (torch.Tensor): input tensor with shape [..., L, E]
            attn_mask (Optional[torch.Tensor]): boolean (float) attention mask (bias) with shape [..., L, L] or [..., H, L, L]

        Returns:
            torch.Tensor: output tensor with shape [..., L, E]
        """
        batch_shape = x.shape[:-2]
        L = x.shape[-2]
        qk = self.qk(x).view(*batch_shape, L, 2, self.num_heads, self.query_dim).transpose(-3, -4).contiguous()
        q, k = self.RoPE(qk).view(*batch_shape, 2 * self.num_heads, L, self.query_dim).chunk(2, dim=-3)
        v = self.v(x).view(*batch_shape, L, self.num_heads, self.value_dim).transpose(-2, -3)

        if attn_mask is not None and attn_mask.dim() <= x.dim(): # handle same mask for all heads
            attn_mask = attn_mask.unsqueeze(-3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=(self.dropout if self.training else 0.))

        return self.out_proj(out.transpose(-2, -3).flatten(-2))
    

class LOREMHSA(MultiheadSelfAttention):
    def __init__(self, embedding_dim, query_dim, value_dim, num_heads, dropout=0, cache_size=512, base=10000):
        super().__init__(embedding_dim, query_dim, value_dim, num_heads, dropout)

        self.RoPE = RoPE(query_dim, cache_size, base)

        self.lore_q = nn.Linear(query_dim, query_dim)
        self.lore_k = nn.Linear(query_dim, query_dim)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x (torch.Tensor): input tensor with shape [..., L, E]
            attn_mask (Optional[torch.Tensor]): boolean (float) attention mask (bias) with shape [..., L, L] or [..., H, L, L]

        Returns:
            torch.Tensor: output tensor with shape [..., L, E]
        """
        batch_shape = x.shape[:-2]
        L = x.shape[-2]
        qk = self.qk(x).view(*batch_shape, L, 2, self.num_heads, self.query_dim).transpose(-3, -4).contiguous()
        q, k = self.RoPE(qk).view(*batch_shape, 2 * self.num_heads, L, self.query_dim).chunk(2, dim=-3)
        q, k = self.lore_q(q), self.lore_k(k)
        v = self.v(x).view(*batch_shape, L, self.num_heads, self.value_dim).transpose(-2, -3)

        if attn_mask is not None and attn_mask.dim() <= x.dim(): # handle same mask for all heads
            attn_mask = attn_mask.unsqueeze(-3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=(self.dropout if self.training else 0.))

        return self.out_proj(out.transpose(-2, -3).flatten(-2))


class FFN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0., act=F.silu):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x))))
    

class GLUFFN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0., act=F.silu):
        super().__init__()
        self.w13 = nn.Linear(embedding_dim, 2 * hidden_dim)
        self.w2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

    def forward(self, x):
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(self.dropout(self.act(x1)) * x3)
    

class Encoder(nn.Module):
    def __init__(self, n_layers, MHSA_factory, FFN_factory, norm_factory):
        super().__init__()

        self.n_layers = n_layers
        self.MHSAs = nn.ModuleList([MHSA_factory() for _ in range(n_layers)])
        self.FFNs = nn.ModuleList([FFN_factory() for _ in range(n_layers)])
        self.norms = nn.ModuleList([norm_factory() for _ in range(2 * n_layers)])

    def forward(self, x, attn_mask=None):
        for i in range(self.n_layers):
            x = x + self.MHSAs[i](self.norms[2 * i](x), attn_mask)
            x = x + self.FFNs[i](self.norms[2 * i + 1](x))
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, MHSA_factory, MHCA_factory, FFN_factory, norm_factory):
        super().__init__()

        self.n_layers = n_layers
        self.MHSAs = nn.ModuleList([MHSA_factory() for _ in range(n_layers)])
        self.MHCAs = nn.ModuleList([MHCA_factory() for _ in range(n_layers)])
        self.FFNs = nn.ModuleList([FFN_factory() for _ in range(n_layers)])
        self.norms = nn.ModuleList([norm_factory() for _ in range(3 * n_layers)])

    def forward(self, x, y, attn_mask=None, cross_attn_mask=None):
        for i in range(self.n_layers):
            x = x + self.MHSAs[i](self.norms[3 * i](x), attn_mask)
            x = x + self.MHCAs[i](self.norms[3 * i + 1](x), y, cross_attn_mask)
            x = x + self.FFNs[i](self.norms[3 * i + 2](x))
        return x
