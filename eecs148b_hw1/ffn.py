import torch
import torch.nn as nn
from torch import Tensor
from eecs148b_hw1.linear import Linear

class FFN(nn.Module):
        def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
            super().__init__()
            self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
            self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        
        def forward(self, x):
            return self.w2(torch.maximum(torch.tensor(0.0), self.w1(x)))
        
