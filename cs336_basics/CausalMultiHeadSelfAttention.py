import torch
import torch.nn as nn
from .Linear import Linear
from .attention import scaled_dot_product_attention
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
    ):
        """
        构建线性变换模块。
        参数：
            d_model: int：输入的维度
            num_heads: int：多头自注意力的头数
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads

        # 查询参数矩阵
        self.W_Q = Linear(in_features=d_model, out_features=d_model)
        self.W_K = Linear(in_features=d_model, out_features=d_model)
        self.W_V = Linear(in_features=d_model, out_features=d_model)
        self.W_O = Linear(in_features=d_model, out_features=d_model)

        # 是否使用 rope
        if theta is not None:
            if max_seq_len is None:
                raise ValueError("使用 RoPE 时必须提供 max_seq_len")
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """将线性变换应用于输入。"""
        batch_size, seq_len, _ = x.shape

        # d_model 64
        # num_heads 4
        # x: shape(4, 12, 64)

        # 1. 线性变换
        # (batch, seq_len, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # 2. 分头
        # (batch, seq_len, num_heads, d_k)
        # (4, 12, 4, 16)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        # 3. 转置为: (batch, num_heads, seq_len, d_k)
        # (4, 4, 12, 16)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 可选，应用 RoPE 到 Q 和 K
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 4.创建因果掩码，False 表示不可关注
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0).bool()

        # (batch, num_heads, seq_len, d_k)
        # (batch_size, ..., d_v)
        # 5.计算注意力
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # 6. 合并多头（Concat）
        # 转置回: [batch, seq_len, num_heads, d_k]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 重塑为: [batch, seq_len, d_model]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        return self.W_O(attn_output)
