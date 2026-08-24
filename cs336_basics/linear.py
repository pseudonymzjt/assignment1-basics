import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        '''
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        tensor = torch.empty((out_features, in_features), device = self.device, dtype = self.dtype)
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.device:
            x = x.to(self.device)
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")