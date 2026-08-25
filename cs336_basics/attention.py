import torch
from einops import einsum, rearrange
from torch import nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim = dim, keepdim = True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim = dim, keepdim = True)

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
        scores = scores.masked_fill(Mask == False, float('-inf'))

    return einsum(softmax(scores, dim = -1), Values, "... queries keys, ... keys d_v -> ... queries d_v")

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
        if self.max_seq_len is not None and self.theta is not None:
            from cs336_basics.embedding import RotaryPositionalEmbedding
            RPE = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len)
            token_positions = torch.arange(0, seq_len) if self.token_positions is None else self.token_positions
            Q = RPE.forward(Q, token_positions)
            K = RPE.forward(K, token_positions)

        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask)

        # concat
        out = rearrange(out, "b h s d_v -> b s (h d_v)")

        # x @ W_O.T: (batch, seq_len, h*d_v) @ (h*d_v, d_model)
        return out @ self.W_O.T

