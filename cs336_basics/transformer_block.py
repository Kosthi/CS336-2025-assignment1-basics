import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .CausalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from .SwiGLU import SiLUFFN, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        *,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        """
        Transtormer 块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的 Transformer 块。
        Args:
            d_model: int: 输入的维度,也就是d_model
            num_heads: int: 头的数量
            d_ff: int: 前馈神经网络的维度
            max_seq_len: int: 最大序列长度
            theta: float: 底数超参数
        """
        super().__init__()
        self.norm_style = norm_style
        self.ffn_type = ffn_type

        if use_rmsnorm:
            self.rms_norm1 = RMSNorm(d_model=d_model, device=device)
            self.rms_norm2 = RMSNorm(d_model=d_model, device=device)
        else:
            self.rms_norm1 = nn.Identity()
            self.rms_norm2 = nn.Identity()

        attn_theta = theta if use_rope else None
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, attn_theta, max_seq_len, device=device)

        if ffn_type == "swiglu":
            self.swi_glu = SwiGLU(d_model, d_ff)
        elif ffn_type == "silu":
            self.silu_ffn = SiLUFFN(d_model, d_ff)
        else:
            raise ValueError(f"未知 ffn_type: {ffn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (..., seq_len, d_model)

        Returns:
            输出张量，形状为 (..., seq_len, d_model)
        """
        token_positions = torch.arange(x.shape[-2], device=x.device)
        ffn = self.swi_glu if self.ffn_type == "swiglu" else self.silu_ffn
        if self.norm_style == "pre":
            y = x + self.attn(self.rms_norm1(x), token_positions)
            z = y + ffn(self.rms_norm2(y))
            return z
        if self.norm_style == "post":
            y = self.rms_norm1(x + self.attn(x, token_positions))
            z = self.rms_norm2(y + ffn(y))
            return z
        raise ValueError(f"未知 norm_style: {self.norm_style}")
