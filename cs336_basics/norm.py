import torch
from einops import einsum, rearrange


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        '''
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int  Hidden dimension of the model
        eps: float = 1e-5  Epsilon value for numerical stability
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        tensor = torch.empty(d_model, device = self.device, dtype = self.dtype)
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        '''
        if self.device:
            x = x.to(self.device)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True, dtype = self.dtype) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)
