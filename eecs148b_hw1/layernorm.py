import torch
import torch.nn as nn
from torch import Tensor

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean = x.mean(dim=-1, keepdim=True)               # (..., 1)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True) # (..., 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # (..., d_model)
        result = x_norm * self.g + self.b                   # learned affine

        return result.to(in_dtype)