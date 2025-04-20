import torch as pt
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Optional

from common import Encoder, Decoder, RoPEMHSA, MultiheadCrossAttention, FFN



class WordEncoder(nn.Module):
    def __init__(self, vocab_size: Tuple[int, int, int], pad_id: int, embedding_dim: int, spm_dim: Optional[int]=None, suffix_dim: Optional[int]=None, prefix_dim: Optional[int]=None, spm_layers: int=3, spm_heads : int=8, spm_rope_cache: int=64, ffn_hidden_dim: Optional[int]=None, expansion_factor: int=2, dropout: float=0.):
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
            lambda: FFN(spm_dim, spm_dim * expansion_factor if ffn_hidden_dim is None else ffn_hidden_dim, dropout), 
            lambda: nn.RMSNorm(spm_dim)
        )
        self.spm_final_norm = nn.RMSNorm(spm_dim)
        self.spm_final_q = nn.Parameter(pt.zeros(spm_heads, spm_dim // spm_heads))
        self.spm_final_k = nn.Linear(spm_dim, spm_dim)
        self.spm_final_v = nn.Linear(spm_dim, spm_dim)
        self.spm_final_proj = nn.Linear(spm_dim, embedding_dim)

        self.suffix_embedding = nn.Embedding(vocab_size[2], suffix_dim, padding_idx=pad_id)
        self.suffix_linear = nn.Linear(suffix_dim, embedding_dim * expansion_factor)

        self.prefix_embedding = nn.Embedding(vocab_size[0], prefix_dim, padding_idx=pad_id)
        self.prefix_linear = nn.Linear(prefix_dim, embedding_dim * expansion_factor)

        self.word_up = nn.Linear(embedding_dim, embedding_dim * expansion_factor)
        self.word_down = nn.Linear(embedding_dim * expansion_factor, embedding_dim)
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

        batch_shape = spm_emb.shape[:-2]
        L = spm_emb.shape[-2]
        spm_k = self.spm_final_k(spm_emb).view(*batch_shape, L, *self.spm_final_q.shape).transpose(-2, -3)
        spm_v = self.spm_final_v(spm_emb).view(*batch_shape, L, *self.spm_final_q.shape).transpose(-2, -3)
        word_emb = self.spm_final_proj(F.scaled_dot_product_attention(self.spm_final_q.view(*(1,)*len(batch_shape), self.spm_final_q.shape[0], 1, self.spm_final_q.shape[1]), spm_k, spm_v, spm_pad_mask.view(*batch_shape, 1, 1, L)).flatten(-3))

        suffix_emb = self.suffix_embedding(suffix_ids)
        for i in range(suffix_ids.shape[-1]):
            word_emb = word_emb + self.word_down(self.relu(self.word_up(word_emb)) * self.suffix_linear(suffix_emb[..., i, :]))
        
        prefix_emb = self.prefix_embedding(prefix_ids)
        for i in range(prefix_ids.shape[-1]):
            word_emb = word_emb + self.word_down(self.relu(self.word_up(word_emb)) * self.prefix_linear(prefix_emb[..., i, :]))

        return word_emb



class WordDecoder(nn.Module):
    def __init__(self, vocab_size: Tuple[int, int, int], pad_id:int, bos_id: int, eos_id: int, embedding_dim: int, spm_dim: Optional[int]=None, suffix_dim: Optional[int]=None, prefix_dim: Optional[int]=None, num_layers: int=3, num_heads: int=8, rope_cache: int=64, expansion_factor: int=2, dropout: float=0.):
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

        self.prefix_proj = nn.Linear(embedding_dim, prefix_dim)
        self.prefix_embedding = nn.Embedding(vocab_size[0], prefix_dim, padding_idx=pad_id)
        self.prefix_decoder = Encoder(num_layers,
            lambda: RoPEMHSA(prefix_dim, prefix_dim // num_heads, prefix_dim // num_heads, num_heads, dropout, rope_cache),
            lambda: FFN(prefix_dim, prefix_dim * expansion_factor, dropout),
            lambda: nn.RMSNorm(prefix_dim)
        )
        self.prefix_norm = nn.RMSNorm(prefix_dim)
        self.prefix_linear = lambda x: F.linear(x, self.prefix_embedding.weight)

        assert spm_dim % num_heads == 0

        self.spm_proj = nn.Linear(embedding_dim, spm_dim)
        self.spm_embedding = nn.Embedding(vocab_size[1], spm_dim, padding_idx=pad_id)
        self.spm_decoder = Encoder(num_layers,
            lambda: RoPEMHSA(spm_dim, spm_dim // num_heads, spm_dim // num_heads, num_heads, dropout, rope_cache),
            lambda: FFN(spm_dim, spm_dim * expansion_factor, dropout),
            lambda: nn.RMSNorm(spm_dim)
        )
        self.spm_norm = nn.RMSNorm(spm_dim)
        self.spm_linear = lambda x: F.linear(x, self.spm_embedding.weight)       

        assert suffix_dim % num_heads == 0

        self.suffix_proj = nn.Linear(embedding_dim, suffix_dim)
        self.suffix_embedding = nn.Embedding(vocab_size[2], suffix_dim, padding_idx=pad_id)
        self.suffix_decoder = Encoder(num_layers,
            lambda: RoPEMHSA(suffix_dim, suffix_dim // num_heads, suffix_dim // num_heads, num_heads, dropout, rope_cache),
            lambda: FFN(suffix_dim, suffix_dim * expansion_factor, dropout),
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

        prefix_pad_mask = F.pad(prefix_ids != self.pad_id, (1, 0), value=True)
        prefix_attn_mask = pt.tril(prefix_pad_mask.unsqueeze(-1) & prefix_pad_mask.unsqueeze(-2))
        prefix_emb = pt.concat([self.prefix_proj(embedding), self.prefix_embedding(prefix_ids)], dim=-2)
        prefix_logits = self.prefix_linear(self.prefix_norm(self.prefix_decoder(prefix_emb, prefix_attn_mask)[..., 1:, :]))

        spm_pad_mask = F.pad(spm_ids != self.pad_id, (1, 0), value=True)
        spm_attn_mask = pt.tril(spm_pad_mask.unsqueeze(-1) & spm_pad_mask.unsqueeze(-2))
        spm_emb = pt.concat([self.spm_proj(embedding), self.spm_embedding(spm_ids)], dim=-2)
        spm_logits = self.spm_linear(self.spm_norm(self.spm_decoder(spm_emb, spm_attn_mask)[..., 1:, :]))

        suffix_pad_mask = F.pad(suffix_ids != self.pad_id, (1, 0), value=True)
        suffix_attn_mask = pt.tril(suffix_pad_mask.unsqueeze(-1) & suffix_pad_mask.unsqueeze(-2))
        suffix_emb = pt.concat([self.suffix_proj(embedding), self.suffix_embedding(suffix_ids)], dim=-2)
        suffix_logits = self.suffix_linear(self.suffix_norm(self.suffix_decoder(suffix_emb, suffix_attn_mask)[..., 1:, :]))

        return prefix_logits, spm_logits, suffix_logits
    
    @pt.inference_mode()
    def inference(self, embedding, max_len=64):
        self.eval()
        if isinstance(max_len, int):
            max_len = (max_len, max_len, max_len)

        embedding = embedding.unsqueeze(-2)
        device = embedding.device

        prefix_ids = [self.bos_id]
        prefix_emb = pt.concat([self.prefix_proj(embedding), self.prefix_embedding(pt.tensor(prefix_ids, dtype=pt.long, device=device))], dim=-2)
        for _ in range(max_len[0]):
            next_token = self.prefix_linear(self.prefix_norm(self.prefix_decoder(prefix_emb, pt.tril(pt.ones((len(prefix_ids) + 1,)*2, dtype=pt.bool, device=device)))[-1])).argmax(-1)
            prefix_emb = pt.concat([prefix_emb, self.prefix_embedding(next_token).unsqueeze(-2)], dim=-2)
            next_token = next_token.item()
            prefix_ids.append(next_token)
            if next_token == self.eos_id:
                break
        
        spm_ids = [self.bos_id]
        spm_emb = pt.concat([self.spm_proj(embedding), self.spm_embedding(pt.tensor(spm_ids, dtype=pt.long, device=device))], dim=-2)
        for _ in range(max_len[1]):
            next_token = self.spm_linear(self.spm_norm(self.spm_decoder(spm_emb, pt.tril(pt.ones((len(spm_ids) + 1,)*2, dtype=pt.bool, device=device)))[-1])).argmax(-1)
            spm_emb = pt.concat([spm_emb, self.spm_embedding(next_token).unsqueeze(-2)], dim=-2)
            next_token = next_token.item()
            spm_ids.append(next_token)
            if next_token == self.eos_id:
                break

        suffix_ids = [self.bos_id]
        suffix_emb = pt.concat([self.suffix_proj(embedding), self.suffix_embedding(pt.tensor(suffix_ids, dtype=pt.long, device=device))], dim=-2)
        for _ in range(max_len[2]):
            next_token = self.suffix_linear(self.suffix_norm(self.suffix_decoder(suffix_emb, pt.tril(pt.ones((len(suffix_ids) + 1,)*2, dtype=pt.bool, device=device)))[-1])).argmax(-1)
            suffix_emb = pt.concat([suffix_emb, self.suffix_embedding(next_token).unsqueeze(-2)], dim=-2)
            next_token = next_token.item()
            suffix_ids.append(next_token)
            if next_token == self.eos_id:
                break

        return prefix_ids, spm_ids, suffix_ids
