import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int,
                 max_seq_len: int,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        rows = torch.arange(max_seq_len).unsqueeze(1)
        even_cols = torch.arange(0, d_model, 2).unsqueeze(0)
        odd_cols = torch.arange(1, d_model, 2).unsqueeze(0)

        positional_embeddings = torch.zeros(max_seq_len, d_model)
        positional_embeddings[:,0::2] = torch.sin(rows/(torch.pow(10000, even_cols/d_model)))
        positional_embeddings[:,1::2] = torch.cos(rows/(torch.pow(10000, (odd_cols-1)/d_model)))
        self.register_buffer('positional_embeddings', positional_embeddings, persistent=False)


    def forward(self, token_positions: Tensor) -> Tensor:
        return self.positional_embeddings[token_positions]