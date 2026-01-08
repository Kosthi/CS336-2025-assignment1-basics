import torch
import math
from .softmax import softmax


def scaled_dot_product_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None = None
):
    """
    参数：
    queries: (batch_size, ..., seq_len_q, d_k)
    keys: (batch_size, ..., seq_len_k, d_k)
    values: (batch_size, ..., seq_len_k, d_v)
    mask: 可选的布尔掩码，形状 (seq_len_q, seq_len_k)

    返回：
    output: (batch_size, ..., d_v)
    """
    # 计算 (q, d) (k, d) Q*K^T (q, k)
    d_k = queries.size(-1)
    scores = torch.einsum("...qd, ...kd -> ...qk", queries, keys)
    # 缩放
    scores = scores / math.sqrt(d_k)

    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 计算注意力权重
    weights = softmax(scores, dim=-1)

    return torch.einsum("...qk, ...kv -> ...qv", weights, values)
