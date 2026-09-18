import torch
from einops import einsum, rearrange
from torch import nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算 cross-entropy loss，带数值稳定性优化
    
        max_val = max(logits)
        ℓ = -logits[target] + log(sum(exp(logits - max_val))) + max_val
          = -(logits[target] - max_val) + log(sum(exp(logits - max_val)))
    
    Args:
        logits: shape (..., vocab_size) 的未归一化 logits
        targets: shape (...,) 的目标类别索引（long类型）
    
    Returns:
        scalar tensor，所有样本的平均 cross-entropy loss
    """
    # logits: (..., vocab_size)
    # targets: (...)
    
    vocab_size = logits.shape[-1]
    batch_shape = targets.shape
    
    logits_flat = logits.reshape(-1, vocab_size)  # (N, vocab_size)
    targets_flat = targets.reshape(-1)  # (N,)
    
    # 减去最大值
    max_logits = logits_flat.max(dim=-1, keepdim=True).values  # (N, 1)
    logits_shifted = logits_flat - max_logits  # (N, vocab_size)
    
    # log(sum(exp(logits_shifted)))
    log_sum_exp = torch.log(torch.exp(logits_shifted).sum(dim=-1))  # (N,)
    
    # logits_shifted[i, targets_flat[i]]
    target_logits_shifted = logits_shifted[torch.arange(targets_flat.size(0)), targets_flat]  # (N,)

    loss = -target_logits_shifted + log_sum_exp  # (N,)
    
    # 5. 返回平均值
    return loss.mean()


def scaled_dot_product_attention(Query: torch.Tensor, Key: torch.Tensor, Values: torch.Tensor, Mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Query.shape[-1]

    scores = einsum(Query, Key, "... queries d_k, ... keys d_k -> ... queries keys") / d_k ** 0.5

    # masked
    if Mask is not None:
        scores = scores.masked_fill(~Mask, float('-inf')) 
    # PyTorch 2.x 的 Inductor 编译器在对符号图做边界推导（Value Ranges / SymPy）时，遇到 Mask == False 会生成布尔比较节点，
    # 进而触发了 SymPy 的已知类型比对 Bug（TypeError: A Boolean argument can only be used in Eq and Ne）
    # 关闭torch.compile后可使用 Mask = False

    return einsum(softmax(scores, dim=-1), Values, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        W_Q: torch.Tensor | None = None,  # (h*d_k, d_model)
        W_K: torch.Tensor | None = None,
        W_V: torch.Tensor | None = None,
        W_O: torch.Tensor | None = None,  # (d_model, h*d_v)
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 权重形状和 state_dict 里 nn.Linear 一致：(out, in)
        self.W_Q = nn.Parameter(W_Q if W_Q is not None else torch.randn(num_heads * self.d_k, d_model))
        self.W_K = nn.Parameter(W_K if W_K is not None else torch.randn(num_heads * self.d_k, d_model))
        self.W_V = nn.Parameter(W_V if W_V is not None else torch.randn(num_heads * self.d_v, d_model))
        self.W_O = nn.Parameter(W_O if W_O is not None else torch.randn(d_model, num_heads * self.d_v))

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions

        # 在初始化时创建 RoPE（如果需要）
        self.rope = None
        if max_seq_len is not None and theta is not None:
            from cs336_basics.embedding import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # x @ W.T: (batch, seq_len, d_model) @ (d_model, h*d_k) -> (batch, seq_len, h*d_k)
        Q = rearrange(x @ self.W_Q.T, "b s (h d_k) -> b h s d_k", h=self.h)
        K = rearrange(x @ self.W_K.T, "b s (h d_k) -> b h s d_k", h=self.h)
        V = rearrange(x @ self.W_V.T, "b s (h d_v) -> b h s d_v", h=self.h)
        
        # RoPE
        if self.rope is not None:
            token_positions = torch.arange(0, seq_len, device=x.device) if self.token_positions is None else self.token_positions
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask)

        # concat
        out = rearrange(out, "b h s d_v -> b s (h d_v)")

        # x @ W_O.T: (batch, seq_len, h*d_v) @ (h*d_v, d_model)
        return out @ self.W_O.T
