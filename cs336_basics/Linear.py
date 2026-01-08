import torch
import torch.nn as nn
from torch.nn import init
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        构建线性变换模块。
        参数：
            in_features: int：输入的最终维度
            out_features: int：输出的最终维度
            device: torch.device | None = None：参数存储设备
            dtype: torch.dtype | None = None：参数数据类型
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 创建权重参数 W，形状为 (out_features, in_features)
        # 使用指定的设备和数据类型
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        # 使用截断正态分布初始化权重
        # 均值为0，标准差为0.02，截断范围通常为[-2*std, 2*std]
        init.trunc_normal_(W, mean=0.0, std=0.02, a=-0.04, b=0.04)
        # 将权重包装为 nn.Parameter
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将线性变换应用于输入。"""
        return torch.einsum("...i, oi -> ...o", x, self.W)
        # return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


if __name__ == "__main__":
    D = torch.randn(32, 10, 512)  # 32个样本，每个样本10个token，512维特征
    A = torch.randn(256, 512)  # 权重矩阵
    Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
    print(A)
