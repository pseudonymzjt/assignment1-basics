import torch
from torch import nn

from cs336_basics.attention import MultiHeadAttention, softmax
from cs336_basics.embedding import Embedding
from cs336_basics.network import SwiGLU
from cs336_basics.norm import RMSNorm
from cs336_basics.linear import Linear


class Block(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int, 
        theta: float,
        norm_type: str = "rmsnorm",       # [插入] 'rmsnorm' 或 'none'
        norm_position: str = "pre",       # [插入] 'pre' 或 'post'
        use_rope: bool = True,          # [插入] 新增参数，默认使用 RoPE
    ):
        super().__init__()
        self.norm_position = norm_position
        
        # [插入] 如果为 none 则使用恒等映射 nn.Identity()
        if norm_type == "none":
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)

        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, use_rope=use_rope)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [插入] 兼容 Post-Norm 与 Pre-Norm
        if self.norm_position == "post":
            # post-norm: y = Norm(x + Sublayer(x))
            y = self.ln1(x + self.attn(x))
            y = self.ln2(y + self.ffn(y))
        else:
            # pre-norm: y = x + Sublayer(Norm(x))
            y = x + self.attn(self.ln1(x))
            y = y + self.ffn(self.ln2(y))
        return y


class Transformer(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        d_ff, 
        theta,
        weights, 
        vocab_size, 
        context_length, 
        num_layers,
        norm_type: str = "rmsnorm",       # [插入] 新增参数，默认保持原有行为
        norm_position: str = "pre",       # [插入] 新增参数，默认 pre-norm
        use_rope: bool = True,          # [插入] 新增参数，默认使用 RoPE
    ):
        super().__init__()
        
        # 保存配置
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.norm_position = norm_position

        # 初始化所有层（随机初始化）
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            Block(
                d_model, 
                num_heads, 
                d_ff, 
                context_length, 
                theta,
                norm_type=norm_type,
                norm_position=norm_position,
                use_rope=use_rope,
            )
            for _ in range(num_layers)
        ])
        
        # [插入] 最终 Norm：如果为 none 则用 Identity
        if norm_type == "none":
            self.ln_final = nn.Identity()
        else:
            self.ln_final = RMSNorm(d_model)
            
        self.lm_head = Linear(d_model, vocab_size)
        
        # 如果提供了权重，加载它们
        if weights is not None:
            self._load_weights(weights)
    
    def _load_weights(self, weights):
        """从字典加载预训练权重"""
        self.token_embeddings.weight = nn.Parameter(weights["token_embeddings.weight"])
        
        for i, block in enumerate(self.blocks):
            block.attn.W_Q = nn.Parameter(weights[f"layers.{i}.attn.q_proj.weight"])
            block.attn.W_K = nn.Parameter(weights[f"layers.{i}.attn.k_proj.weight"])
            block.attn.W_V = nn.Parameter(weights[f"layers.{i}.attn.v_proj.weight"])
            block.attn.W_O = nn.Parameter(weights[f"layers.{i}.attn.output_proj.weight"])
            # [插入] 仅当包含 weight 时才赋值（避免 Identity 报错）
            if hasattr(block.ln1, "weight"):
                block.ln1.weight = nn.Parameter(weights[f"layers.{i}.ln1.weight"])
            if hasattr(block.ln2, "weight"):
                block.ln2.weight = nn.Parameter(weights[f"layers.{i}.ln2.weight"])
            block.ffn.w1.weight = nn.Parameter(weights[f"layers.{i}.ffn.w1.weight"])
            block.ffn.w2.weight = nn.Parameter(weights[f"layers.{i}.ffn.w2.weight"])
            block.ffn.w3.weight = nn.Parameter(weights[f"layers.{i}.ffn.w3.weight"])
        
        if hasattr(self.ln_final, "weight"):
            self.ln_final.weight = nn.Parameter(weights["ln_final.weight"])
        self.lm_head.weight = nn.Parameter(weights["lm_head.weight"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch, seq_len, d_model)
        out = self.token_embeddings(x)

        for block in self.blocks:
            out = block(out)

        out = self.ln_final(out)

        # (batch, seq_len, vocab_size)
        return self.lm_head(out)

