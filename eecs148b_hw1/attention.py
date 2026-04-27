import torch
import torch.nn as nn
from torch import Tensor
from .linear import Linear
# from tests.adapters import run_scaled_dot_product_attention

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention."""
    scores = Q @ K.transpose(-1, -2)
    scale = torch.sqrt(torch.tensor(Q.shape[-1], dtype=scores.dtype, device=scores.device))
    normalized_scores = scores / scale
    if mask is not None:
        normalized_scores = normalized_scores.masked_fill(~mask, float("-inf"))
    attn_weights = torch.softmax(normalized_scores, dim=-1)
    return attn_weights @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                 device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
    

    def forward(self, in_features):
        # (batch, seq_len, d_model)
        Q = self.w_q(in_features)
        K = self.w_k(in_features)
        V = self.w_v(in_features)
        # O = self.w_o(in_features)

        # (batch, num_heads, seq_len, d_k)
        Q = Q.view(*Q.shape[:-1], self.num_heads, self.d_k)
        Q = Q.transpose(-2,-3)
        K = K.view(*K.shape[:-1], self.num_heads, self.d_k)
        K = K.transpose(-2, -3)
        V = V.view(*V.shape[:-1], self.num_heads, self.d_k)
        V = V.transpose(-2, -3) 

        # construct bool mask matrix (0 for upper triangle above diagonal)
        seq_len = in_features.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
        
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        multihead = scaled_dot_product_attention(Q, K, V, mask).transpose(-2, -3)
        multihead = multihead.contiguous()
        multihead = multihead.view(*multihead.shape[:-2], self.d_model)

        return self.w_o(multihead)
        