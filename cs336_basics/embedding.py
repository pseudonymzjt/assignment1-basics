import torch


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        tensor = torch.empty((num_embeddings, embedding_dim), device=self.device, dtype=self.dtype)
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(tensor))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        '''Lookup the embedding vectors for the given token IDs.)'''
        if self.device:
            token_ids = token_ids.to(self.device)
        if self.dtype:
            token_ids = token_ids.to(self.dtype)
        return self.weight[token_ids]

class RotaryPositionalEmbedding(torch.nn.Module):
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        '''
        Construct the RoPE module and create buffers if needed.
        theta: float  Θ value for the RoPE
        d_k: int  dimension of query and key vectors
        max_seq_len: int  Maximum sequence length that will be input
        device: torch.device | None = None  Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        # 向量化计算 theta_matrix
        i = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        k = torch.arange(d_k // 2)  # (d_k // 2,)
        freqs = 1.0 / (theta ** (2 * k / d_k))  # (d_k // 2,)
        theta_matrix = i * freqs  # (max_seq_len, d_k // 2)
        
        # 计算 cos 和 sin
        cos_cache = torch.cos(theta_matrix)
        sin_cache = torch.sin(theta_matrix)
        
        # 注册为 buffer（如果指定了 device，移到该设备上）
        if device is not None:
            cos_cache = cos_cache.to(device)
            sin_cache = sin_cache.to(device)
        
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        
        # 确保输入在正确的设备上（使用缓存的设备）
        if x.device != self.cos_cache.device:
            x = x.to(self.cos_cache.device)
        if token_positions.device != self.cos_cache.device:
            token_positions = token_positions.to(self.cos_cache.device)

        cos = self.cos_cache[token_positions]  # (..., seq_len, d_k // 2)
        sin = self.sin_cache[token_positions]  # (..., seq_len, d_k // 2)
        
        *batch_dims, seq_len, d_k = x.shape
        
        x_pairs = x.reshape(*batch_dims, seq_len, d_k // 2, 2)
        x_even = x_pairs[..., 0]  # (..., seq_len, d_k // 2)
        x_odd = x_pairs[..., 1]   # (..., seq_len, d_k // 2)
        
        x_even_ro = x_even * cos - x_odd * sin
        x_odd_ro = x_even * sin + x_odd * cos
        
        x_ro_pairs = torch.stack([x_even_ro, x_odd_ro], dim=-1)
        x_ro = x_ro_pairs.reshape(*batch_dims, seq_len, d_k)
        
        return x_ro
