import torch as pt
from torch import nn
import torch.nn.functional as F

from typing import Optional

from common import Encoder, Decoder, MultiheadCrossAttention, RoPEMHSA, GLUFFN

from word_autoencoder import WordEncoder, WordDecoder


class Transformer(nn.Module):
    def __init__(self, embedding_dim: int, n_encoder_layers: int=6, n_decoder_layers: int=6, n_heads: int=8, ffn_hidden_dim: Optional[int]=None, query_dim: Optional[int]=None, value_dim: Optional[int]=None, dropout: float=0.):
        super().__init__()

        self.embedding_dim = embedding_dim

        if ffn_hidden_dim is None:
            ffn_hidden_dim = embedding_dim * 4
        if query_dim is None:
            assert embedding_dim % n_heads == 0
            query_dim = embedding_dim // n_heads
        if value_dim is None:
            assert embedding_dim % n_heads == 0
            value_dim = embedding_dim // n_heads

        self.encoder = Encoder(n_encoder_layers,
            lambda: RoPEMHSA(embedding_dim, query_dim, value_dim, n_heads, dropout),
            lambda: GLUFFN(embedding_dim, ffn_hidden_dim, dropout),
            lambda: nn.RMSNorm(embedding_dim)
        )
        self.encoder_norm = nn.RMSNorm(embedding_dim)

        self.decoder = Decoder(n_decoder_layers,
            lambda: RoPEMHSA(embedding_dim, query_dim, value_dim, n_heads, dropout),
            lambda: MultiheadCrossAttention(embedding_dim, query_dim, value_dim, n_heads, dropout=dropout),
            lambda: GLUFFN(embedding_dim, ffn_hidden_dim, dropout),
            lambda: nn.RMSNorm(embedding_dim)
        )
        self.decoder_norm = nn.RMSNorm(embedding_dim)

    def encode(self, embeddings):
        """
        Args:
            embeddings (torch.Tensor): float embeddings tensor with shape [L, E]

        Returns:
            torch.Tensor: float embeddings tensor with shape [L, E]
        """
        return self.encoder_norm(self.encoder(embeddings))
    
    def decode(self, embeddings, context_embeddings, attn_mask):
        """
        Args:
            embeddings (torch.Tensor): float embeddings tensor with shape [L, E]
            context_embeddings (torch.Tensor): float embeddings tensor with shape [S, E]
            attn_mask (torch.Tensor): boolean attention mask tensor with shape [L, L]

        Returns:
            torch.Tensor: float embeddings tensor with shape [L, E]
        """
        return self.decoder_norm(self.decoder(embeddings, context_embeddings, attn_mask))


class MOPLM(nn.Module):
    def __init__(self, word_encoder: WordEncoder, word_decoder: WordDecoder, transformer: Transformer):
        super().__init__()

        self.word_encoder = word_encoder
        self.word_decoder = word_decoder
        self.transformer = transformer

    def forward(self, prefix_ids, spm_ids, suffix_ids, context_prefix_ids, context_spm_ids, context_suffix_ids, output_prefix_ids, output_spm_ids, output_suffix_ids, attn_mask):
        """
        Args:
            prefix_ids (torch.Tensor): integer tensor with shape [L, Lp]
            spm_ids (torch.Tensor): integer tensor with shape [L, Lm]
            suffix_ids (torch.Tensor): integer tensor with shape [L, Ls]
            context_prefix_ids (torch.Tensor): integer tensor with shape [S, Sp]
            context_spm_ids (torch.Tensor): integer tensor with shape [S, Sm]
            context_suffix_ids (torch.Tensor): integer tensor with shape [S, Ss]
            output_prefix_ids (torch.Tensor): integer tensor with shape [L, Op]
            output_spm_ids (torch.Tensor): integer tensor with shape [L, Om]
            output_suffix_ids (torch.Tensor): integer tensor with shape [L, Os]
            attn_mask (torch.Tensor): boolean attention mask tensor with shape [L, L]

        Returns:
            torch.Tensor: logits tensor with shape [L, Op, Vp]
            torch.Tensor: logits tensor with shape [L, Om, Vm]
            torch.Tensor: logits tensor with shape [L, Os, Vs]
        """

        context_embeddings = self.transformer.encode(self.word_encoder(context_prefix_ids, context_spm_ids, context_suffix_ids))
        decoder_embeddings = self.transformer.decode(self.word_encoder(prefix_ids, spm_ids, suffix_ids), context_embeddings, attn_mask)
        return self.word_decoder(output_prefix_ids, output_spm_ids, output_suffix_ids, decoder_embeddings)
    
    @pt.inference_mode()
    def inference(self, context_prefix_ids, context_spm_ids, context_suffix_ids, add_tags=True, beam_search=True, inference_kwargs=None, max_iters=10):
        """
        Args:
            context_prefix_ids (list): integer list of lists
            context_spm_ids (list): integer list of lists
            context_suffix_ids (list): integer list of lists
        
        Returns:
            output_prefix_ids (list): integer list of lists
            output_spm_ids (list): integer list of lists
            output_suffix_ids (list): integer list of lists
        """
        if inference_kwargs is None:
            inference_kwargs = {}

        device = self.transformer.encoder_norm.weight.device

        bos_id, eos_id = self.word_decoder.bos_id, self.word_decoder.eos_id
        empty_tensor = pt.tensor([bos_id, eos_id], dtype=pt.long, device=device)
        empty_embedding = self.word_encoder(empty_tensor, empty_tensor, empty_tensor).unsqueeze(0)

        context_embeddings = []
        for prefix_ids, spm_ids, suffix_ids in zip(context_prefix_ids, context_spm_ids, context_suffix_ids):
            if add_tags:
                prefix_ids, spm_ids, suffix_ids = map(lambda x: [bos_id] + x + [eos_id], (prefix_ids, spm_ids, suffix_ids))
            context_embeddings.append(self.word_encoder(pt.tensor(prefix_ids, dtype=pt.long, device=device), pt.tensor(spm_ids, dtype=pt.long, device=device), pt.tensor(suffix_ids, dtype=pt.long, device=device)).unsqueeze(0))
        context_embeddings = self.transformer.encode(pt.cat(context_embeddings, dim=0))

        creation_iters = []
        input_embeddings = []
        output_prefix_ids = []
        output_spm_ids = []
        output_suffix_ids = []
        cur_iter = 0
        for _ in range(max_iters):
            padded_creation_iters = [cur_iter if i % 2 == 0 else creation_iters[i // 2] for i in range(2 * len(creation_iters) + 1)]
            padded_input_embeddings = [empty_embedding if i % 2 == 0 else input_embeddings[i // 2] for i in range(2 * len(input_embeddings) + 1)]
            mask = []
            for i in padded_creation_iters:
                mask_row = []
                for j in padded_creation_iters:
                    mask_row.append(i >= j)
                mask.append(mask_row)
            attn_mask = pt.tensor(mask, dtype=pt.bool, device=device)

            output_embeddings = self.transformer.decode(pt.cat(padded_input_embeddings, dim=0), context_embeddings, attn_mask)

            new_creation_iters = []
            new_input_embeddings = []
            new_prefix_ids = []
            new_spm_ids = []
            new_suffix_ids = []
            stop = True
            for i, output_embedding in enumerate(output_embeddings[::2]):
                if beam_search:
                    prefix_ids, spm_ids, suffix_ids = self.word_decoder.beam_search(output_embedding, **inference_kwargs)
                else:
                    prefix_ids, spm_ids, suffix_ids = self.word_decoder.inference(output_embedding, **inference_kwargs)
                if prefix_ids[-1] != eos_id:
                    prefix_ids.append(eos_id)
                if spm_ids[-1] != eos_id:
                    spm_ids.append(eos_id)
                if suffix_ids[-1] != eos_id:
                    suffix_ids.append(eos_id)
                if prefix_ids != [bos_id, eos_id] or spm_ids != [bos_id, eos_id] or suffix_ids != [bos_id, eos_id]:
                    stop = False
                    new_creation_iters.append(cur_iter)
                    new_input_embeddings.append(self.word_encoder(pt.tensor(prefix_ids, dtype=pt.long, device=device), pt.tensor(spm_ids, dtype=pt.long, device=device), pt.tensor(suffix_ids, dtype=pt.long, device=device)).unsqueeze(0))
                    new_prefix_ids.append(prefix_ids)
                    new_spm_ids.append(spm_ids)
                    new_suffix_ids.append(suffix_ids)
                if i < len(creation_iters):
                    new_creation_iters.append(creation_iters[i])
                    new_input_embeddings.append(input_embeddings[i])
                    new_prefix_ids.append(output_prefix_ids[i])
                    new_spm_ids.append(output_spm_ids[i])
                    new_suffix_ids.append(output_suffix_ids[i])
            
            creation_iters = new_creation_iters
            input_embeddings = new_input_embeddings
            output_prefix_ids = new_prefix_ids
            output_spm_ids = new_spm_ids
            output_suffix_ids = new_suffix_ids 
            cur_iter += 1

            if stop:
                break 

        return output_prefix_ids, output_spm_ids, output_suffix_ids
