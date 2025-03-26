import torch as pt
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Optional



class WordEncoder(nn.Module):
    def __init__(self, vocab_size: Tuple[int, int, int], pad_id: int, embedding_dim: int, suffix_dim: Optional[int]=None, prefix_dim: Optional[int]=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.spm_embedding = nn.Embedding(vocab_size[1], embedding_dim, padding_idx=pad_id)
        # TODO