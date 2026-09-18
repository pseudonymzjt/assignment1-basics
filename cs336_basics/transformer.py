import torch
from torch import nn

from cs336_basics.attention import MultiHeadAttention, softmax
from cs336_basics.embedding import Embedding
from cs336_basics.network import SwiGLU
from cs336_basics.norm import RMSNorm
from cs336_basics.linear import Linear


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm + residual
        y = x + self.attn(self.ln1(x))
        y = y + self.ffn(self.ln2(y))
        return y

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta,
                 weights, vocab_size, context_length, num_layers):
        super().__init__()
        
        # 保存配置
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        # 初始化所有层（随机初始化）
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layers)
        ])
        
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
            block.ln1.weight = nn.Parameter(weights[f"layers.{i}.ln1.weight"])
            block.ln2.weight = nn.Parameter(weights[f"layers.{i}.ln2.weight"])
            block.ffn.w1.weight = nn.Parameter(weights[f"layers.{i}.ffn.w1.weight"])
            block.ffn.w2.weight = nn.Parameter(weights[f"layers.{i}.ffn.w2.weight"])
            block.ffn.w3.weight = nn.Parameter(weights[f"layers.{i}.ffn.w3.weight"])
        
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

