import math

import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        weight_tensor = torch.empty((out_features, in_features), **factory_kwargs)
        
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            weight_tensor,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )

        self.weight = torch.nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")