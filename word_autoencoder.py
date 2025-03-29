import torch as pt
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Optional

from common import Encoder, RoPEMHSA, FFN



class WordEncoder(nn.Module):
    def __init__(self, vocab_size: Tuple[int, int, int], pad_id: int, embedding_dim: int, spm_dim: Optional[int]=None, suffix_dim: Optional[int]=None, prefix_dim: Optional[int]=None, spm_heads : int=8, spm_rope_cache=64, dropout=0.):
        super().__init__()

        if spm_dim is None:
            spm_dim = embedding_dim
        if suffix_dim is None:
            suffix_dim = embedding_dim
        if prefix_dim is None:
            prefix_dim = embedding_dim

        self.vocab_size = vocab_size
        self.pad_id = pad_id

        assert spm_dim % spm_heads == 0

        self.spm_embedding = nn.Embedding(vocab_size[1], spm_dim, padding_idx=pad_id)
        self.spm_encoder = Encoder(3, 
                                   lambda: RoPEMHSA(spm_dim, spm_dim // spm_heads, spm_dim // spm_heads, spm_heads, dropout, spm_rope_cache), 
                                   lambda: FFN(spm_dim, spm_dim * 2, dropout), 
                                   lambda: nn.RMSNorm(spm_dim))
        self.spm_final_norm = nn.RMSNorm(spm_dim)
        self.spm_final_q = nn.Parameter(pt.zeros(spm_dim))
        self.spm_final_k = nn.Linear(spm_dim, spm_dim)
        self.spm_final_v = nn.Linear(spm_dim, embedding_dim)

        self.suffix_embedding = nn.Embedding(vocab_size[2], suffix_dim, padding_idx=pad_id)
        self.suffix_linear = nn.Linear(suffix_dim, embedding_dim * 2)

        self.prefix_embedding = nn.Embedding(vocab_size[0], prefix_dim, padding_idx=pad_id)
        self.prefix_linear = nn.Linear(prefix_dim, embedding_dim * 2)

        self.word_up = nn.Linear(embedding_dim, embedding_dim * 2)
        self.word_down = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, prefix_ids, spm_ids, suffix_ids):
        """
        Args:
            prefix_ids (torch.Tensor): integer tensor with shape [..., Lp]
            spm_ids (torch.Tensor): integer tensor with shape [..., Lm]
            suffix_ids (torch.Tensor): integer tensor with shape [..., Ls]

        Returns:
            torch.Tensor: output tensor with shape [..., E]
        """
        spm_pad_mask = spm_ids != self.pad_id
        spm_attn_mask = spm_pad_mask.unsqueeze(-1) & spm_pad_mask.unsqueeze(-2) 
        spm_emb = self.spm_final_norm(self.spm_encoder(self.spm_embedding(spm_ids), spm_attn_mask))
        spm_k = self.spm_final_k(spm_emb)
        logits = spm_k @ self.spm_final_q
        logits.masked_fill_(~spm_pad_mask, float("-inf"))
        weights = pt.softmax(logits, dim=-1).nan_to_num()
        word_emb = pt.sum(self.spm_final_v(spm_emb) * weights.unsqueeze(-1), dim=-2)

        suffix_emb = self.suffix_embedding(suffix_ids)
        for i in range(suffix_ids.shape[-1]):
            word_emb = word_emb + self.word_down(self.relu(self.word_up(word_emb)) * self.suffix_linear(suffix_emb[..., i, :]))
        
        prefix_emb = self.prefix_embedding(prefix_ids)
        for i in range(prefix_ids.shape[-1]):
            word_emb = word_emb + self.word_down(self.relu(self.word_up(word_emb)) * self.prefix_linear(prefix_emb[..., i, :]))

        return word_emb
