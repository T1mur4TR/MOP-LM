import torch as pt
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Optional

from common import Encoder, Decoder, RoPEMHSA, MultiheadCrossAttention, FFN



class WordEncoder(nn.Module):
    def __init__(self, vocab_size: Tuple[int, int, int], pad_id: int, embedding_dim: int, spm_dim: Optional[int]=None, suffix_dim: Optional[int]=None, prefix_dim: Optional[int]=None, spm_layers: int=3, spm_heads : int=8, spm_rope_cache: int=64, dropout: float=0.):
        super().__init__()

        if spm_dim is None:
            spm_dim = embedding_dim
        if suffix_dim is None:
            suffix_dim = spm_dim
        if prefix_dim is None:
            prefix_dim = spm_dim

        self.vocab_size = vocab_size
        self.pad_id = pad_id

        assert spm_dim % spm_heads == 0

        self.spm_embedding = nn.Embedding(vocab_size[1], spm_dim, padding_idx=pad_id)
        self.spm_encoder = Encoder(spm_layers, 
            lambda: RoPEMHSA(spm_dim, spm_dim // spm_heads, spm_dim // spm_heads, spm_heads, dropout, spm_rope_cache), 
            lambda: FFN(spm_dim, spm_dim * 2, dropout), 
            lambda: nn.RMSNorm(spm_dim)
        )
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
        logits = spm_k.inner(self.spm_final_q)
        logits.masked_fill_(~spm_pad_mask, float("-inf"))
        weights = pt.softmax(logits, dim=-1).nan_to_num()
        word_emb = self.spm_final_v(pt.sum(spm_emb * weights.unsqueeze(-1), dim=-2))

        suffix_emb = self.suffix_embedding(suffix_ids)
        for i in range(suffix_ids.shape[-1]):
            word_emb = word_emb + self.word_down(self.relu(self.word_up(word_emb)) * self.suffix_linear(suffix_emb[..., i, :]))
        
        prefix_emb = self.prefix_embedding(prefix_ids)
        for i in range(prefix_ids.shape[-1]):
            word_emb = word_emb + self.word_down(self.relu(self.word_up(word_emb)) * self.prefix_linear(prefix_emb[..., i, :]))

        return word_emb



class WordDecoder(nn.Module):
    def __init__(self, vocab_size: Tuple[int, int, int], pad_id:int, bos_id: int, eos_id: int, embedding_dim: int, spm_dim: Optional[int]=None, suffix_dim: Optional[int]=None, prefix_dim: Optional[int]=None, num_layers: int=3, num_heads: int=8, rope_cache: int=64, dropout: float=0.):
        super().__init__()

        if spm_dim is None:
            spm_dim = embedding_dim
        if suffix_dim is None:
            suffix_dim = spm_dim
        if prefix_dim is None:
            prefix_dim = spm_dim

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        assert prefix_dim % num_heads == 0

        self.prefix_embedding = nn.Embedding(vocab_size[0], prefix_dim)
        self.prefix_decoder = Decoder(num_layers,
            lambda: RoPEMHSA(prefix_dim, prefix_dim // num_heads, prefix_dim // num_heads, num_heads, dropout, rope_cache),
            lambda: MultiheadCrossAttention(prefix_dim, prefix_dim // num_heads, prefix_dim // num_heads, num_heads, embedding_dim, dropout),
            lambda: FFN(prefix_dim, prefix_dim * 2, dropout),
            lambda: nn.RMSNorm(prefix_dim)
        )
        self.prefix_norm = nn.RMSNorm(prefix_dim)
        self.prefix_linear = lambda x: F.linear(x, self.prefix_embedding.weight)

        assert spm_dim % num_heads == 0

        self.spm_embedding = nn.Embedding(vocab_size[1], spm_dim)
        self.spm_decoder = Decoder(num_layers,
            lambda: RoPEMHSA(spm_dim, spm_dim // num_heads, spm_dim // num_heads, num_heads, dropout, rope_cache),
            lambda: MultiheadCrossAttention(spm_dim, spm_dim // num_heads, spm_dim // num_heads, num_heads, embedding_dim, dropout),
            lambda: FFN(spm_dim, spm_dim * 2, dropout),
            lambda: nn.RMSNorm(spm_dim)
        )
        self.spm_norm = nn.RMSNorm(spm_dim)
        self.spm_linear = lambda x: F.linear(x, self.spm_embedding.weight)       

        assert suffix_dim % num_heads == 0

        self.suffix_embedding = nn.Embedding(vocab_size[2], suffix_dim)
        self.suffix_decoder = Decoder(num_layers,
            lambda: RoPEMHSA(suffix_dim, suffix_dim // num_heads, suffix_dim // num_heads, num_heads, dropout, rope_cache),
            lambda: MultiheadCrossAttention(suffix_dim, suffix_dim // num_heads, suffix_dim // num_heads, num_heads, embedding_dim, dropout),
            lambda: FFN(suffix_dim, suffix_dim * 2, dropout),
            lambda: nn.RMSNorm(suffix_dim)
        )
        self.suffix_norm = nn.RMSNorm(suffix_dim)
        self.suffix_linear = lambda x: F.linear(x, self.suffix_embedding.weight)

    def forward(self, prefix_ids, spm_ids, suffix_ids, embedding):
        """
        Args:
            prefix_ids (torch.Tensor): integer tensor with shape [..., Lp]
            spm_ids (torch.Tensor): integer tensor with shape [..., Lm]
            suffix_ids (torch.Tensor): integer tensor with shape [..., Ls]
            embedding (torch.Tensor): float embedding tensor with shape [..., E]

        Returns:
            torch.Tensor: prefix logits tensor with shape [..., Lp, Vp]
            torch.Tensor: spm logits tensor with shape [..., Lm, Vm]
            torch.Tensor: suffix logits tensor with shape [..., Ls, Vs]
        """
        embedding = embedding.unsqueeze(-2)

        prefix_pad_mask = prefix_ids != self.pad_id
        prefix_attn_mask = pt.tril(prefix_pad_mask.unsqueeze(-1) & prefix_pad_mask.unsqueeze(-2))
        prefix_logits = self.prefix_linear(self.prefix_norm(self.prefix_decoder(self.prefix_embedding(prefix_ids), embedding, prefix_attn_mask, prefix_pad_mask.unsqueeze(-1))))

        spm_pad_mask = spm_ids != self.pad_id
        spm_attn_mask = pt.tril(spm_pad_mask.unsqueeze(-1) & spm_pad_mask.unsqueeze(-2))
        spm_logits = self.spm_linear(self.spm_norm(self.spm_decoder(self.spm_embedding(spm_ids), embedding, spm_attn_mask, spm_pad_mask.unsqueeze(-1))))

        suffix_pad_mask = suffix_ids != self.pad_id
        suffix_attn_mask = pt.tril(suffix_pad_mask.unsqueeze(-1) & suffix_pad_mask.unsqueeze(-2))
        suffix_logits = self.suffix_linear(self.suffix_norm(self.suffix_decoder(self.suffix_embedding(suffix_ids), embedding, suffix_attn_mask, suffix_pad_mask.unsqueeze(-1))))

        return prefix_logits, spm_logits, suffix_logits
    
    @pt.inference_mode()
    def inference(self, embedding, max_len=64):
        self.eval()
        if isinstance(max_len, int):
            max_len = (max_len, max_len, max_len)

        embedding = embedding.unsqueeze(-2)
        device = embedding.device

        prefix_ids = [self.bos_id]
        for _ in range(max_len[0]):
            next_token = self.prefix_linear(self.prefix_norm(self.prefix_decoder(self.prefix_embedding(pt.tensor(prefix_ids, dtype=pt.long, device=device)), embedding)[-1])).argmax(-1).item()
            prefix_ids.append(next_token)
            if next_token == self.eos_id:
                break
        
        spm_ids = [self.bos_id]
        for _ in range(max_len[1]):
            next_token = self.spm_linear(self.spm_norm(self.spm_decoder(self.spm_embedding(pt.tensor(spm_ids, dtype=pt.long, device=device)), embedding)[-1])).argmax(-1).item()
            spm_ids.append(next_token)
            if next_token == self.eos_id:
                break

        suffix_ids = [self.bos_id]
        for _ in range(max_len[2]):
            next_token = self.suffix_linear(self.suffix_norm(self.suffix_decoder(self.suffix_embedding(pt.tensor(suffix_ids, dtype=pt.long, device=device)), embedding)[-1])).argmax(-1).item()
            suffix_ids.append(next_token)
            if next_token == self.eos_id:
                break
        
        return prefix_ids, spm_ids, suffix_ids

