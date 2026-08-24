import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device = device)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device = device)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device = device)
        self.device = device
        
        for linear in [self.w1, self.w2, self.w3]:
            nn.init.trunc_normal_(linear.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.device:
            x = x.to(self.device)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))