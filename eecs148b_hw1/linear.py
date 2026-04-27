import torch
import torch.nn as nn
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.W)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W.T
        