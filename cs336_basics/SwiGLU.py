import torch
import torch.nn as nn
from .Linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, multiple_of: int = 64):
        """
        初始化 SwiGLU 前馈网络

        Args:
            d_model: 模型维度（输入/输出维度）
            d_ff: 内部前馈层维度，如果为 None 则自动计算为 8/3 * d_model
            multiple_of: 确保 d_ff 是该值的倍数（默认为 64）
        """
        super().__init__()
        self.d_model = d_model

        # 计算 d_ff
        if d_ff is None:
            # 公式：d_ff = (8/3) * d_model
            d_ff = int(8 / 3 * d_model)
            # 确保是 multiple_of 的倍数
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.d_ff = d_ff

        # 权重参数 W，形状为 (out_features, in_features)
        self.W1 = Linear(in_features=d_model, out_features=d_ff)
        self.W2 = Linear(in_features=d_ff, out_features=d_model)
        self.W3 = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (..., d_model)

        Returns:
            输出张量，形状为 (..., d_model)
        """
        w1_x = self.W1(x)
        # 应用 SiLU 激活函数
        silu = w1_x * torch.sigmoid(w1_x)
        # 逐元素相乘（门控）
        glu = silu * self.W3(x)
        return self.W2(glu)


class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, multiple_of: int = 64):
        super().__init__()
        if d_ff % multiple_of != 0:
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.W1 = Linear(in_features=d_model, out_features=d_ff)
        self.W2 = Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W1(x)
        x = x * torch.sigmoid(x)
        return self.W2(x)
