import torch as pt
from torch import nn
import torch.nn.functional as F

from typing import Optional

from common import Encoder, RoPEMHSA, GLUFFN

class DependencyParser(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int=6, final_n_layers: int=3, n_heads: int=16, ffn_hidden_dim: Optional[int]=None, query_dim: Optional[int]=None, value_dim: Optional[int]=None, dropout: float=0.):
        super().__init__()

        if ffn_hidden_dim is None:
            ffn_hidden_dim = embedding_dim * 4
        if query_dim is None:
            assert embedding_dim % n_heads == 0
            query_dim = embedding_dim // n_heads
        if value_dim is None:
            assert embedding_dim % n_heads == 0
            value_dim = embedding_dim // n_heads

        self.embedding_dim = embedding_dim
        self.root_embedding = nn.Parameter(pt.zeros(embedding_dim))
        self.encoder = Encoder(n_layers,
            lambda: RoPEMHSA(embedding_dim, query_dim, value_dim, n_heads, dropout),
            lambda: GLUFFN(embedding_dim, ffn_hidden_dim, dropout),
            lambda: nn.RMSNorm(embedding_dim)
        )

        self.final_q = Encoder(final_n_layers,
            lambda: RoPEMHSA(embedding_dim, query_dim, value_dim, n_heads, dropout),
            lambda: GLUFFN(embedding_dim, ffn_hidden_dim, dropout),
            lambda: nn.RMSNorm(embedding_dim)
        )
        self.final_q_norm = nn.RMSNorm(embedding_dim)
        self.final_k = Encoder(final_n_layers,
            lambda: RoPEMHSA(embedding_dim, query_dim, value_dim, n_heads, dropout),
            lambda: GLUFFN(embedding_dim, ffn_hidden_dim, dropout),
            lambda: nn.RMSNorm(embedding_dim)
        )
        self.final_k_norm = nn.RMSNorm(embedding_dim)
        self.final_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.zeros_(self.final_linear.weight)

    def forward(self, embeddings, padding_mask=None):
        """
        Args:
            embeddings (torch.Tensor): float embeddings tensor with shape [..., L, E]
            padding_mask (torch.Tensor): boolean padding mask with shape [..., L]

        Returns:
            torch.Tensor: dependency logits tensor with shape [..., L, L + 1]
        """
        if padding_mask is not None:
            padding_mask = F.pad(padding_mask, (1, 0), value=True)
            attn_mask = padding_mask.unsqueeze(-2) & padding_mask.unsqueeze(-1)
        else:
            attn_mask = None
        
        batch_shape = embeddings.shape[:-2]
        embeddings = pt.cat([self.root_embedding.view(*(1,)*len(batch_shape), 1, self.embedding_dim).expand(*batch_shape, -1, -1), embeddings], dim=-2)
        embeddings = self.encoder(embeddings, attn_mask)

        q_embeddings = self.final_linear(self.final_q_norm(self.final_q(embeddings, attn_mask)[..., 1:, :]))
        k_embeddings = self.final_k_norm(self.final_k(embeddings, attn_mask))

        return pt.matmul(q_embeddings, k_embeddings.transpose(-1, -2))
    
    @pt.inference_mode()
    def inference(self, embeddings):
        """
        Args:
            embeddings (torch.Tensor): float embeddings tensor with shape [L, E]

        Returns:
            torch.Tensor: dependency index tensor with shape [L]
        """
        self.eval()
        L, _ = embeddings.shape
        logits = self.forward(embeddings)
        probs = F.softmax(logits, dim=-1)

        heads = [None] * L
        found = 0
        while found < L:
            idx = pt.argmax(probs).item()
            from_idx, to_idx = idx // (L + 1), idx % (L + 1)
            probs[from_idx][to_idx] = 0.
            if heads[from_idx] is not None:
                continue
            cur = to_idx - 1
            while cur != -1 and heads[cur] is not None:
                cur = heads[cur] - 1
            if cur == from_idx:
                continue
            heads[from_idx] = to_idx
            found += 1
        
        return pt.tensor(heads, dtype=pt.long, device=embeddings.device)
