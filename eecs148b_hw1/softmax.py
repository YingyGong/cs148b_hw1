import torch
import torch.nn as nn
from torch import Tensor

class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_exp = torch.exp(x - torch.max(x, dim=self.dim, keepdim=True).values)
        return x_exp / torch.sum(x_exp, dim=self.dim, keepdim=True)