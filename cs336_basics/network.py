import torch
import torch.nn.functional as F
from torch import nn

from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device = device)
        self.w2 = Linear(d_ff, d_model, device = device)
        self.w3 = Linear(d_model, d_ff, device = device)
        self.device = device
        
        for linear in [self.w1, self.w2, self.w3]:
            nn.init.trunc_normal_(linear.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.device:
        #     x = x.to(self.device)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN_SiLU(x) = W2(SiLU(W1(x)))
        return self.w2(F.silu(self.w1(x)))