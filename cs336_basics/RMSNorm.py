import torch
import torch.nn as nn
from torch.nn import init


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        构建RMSNorm模块，均方根层归一化
        参数：
            d_model: int：模型隐藏维度
            eps: float = 1e-5：数值稳定性参数
            device: torch.device | None = None：参数存储设备
            dtype: torch.dtype | None = None：参数数据类型
        """
        super().__init__()
        # 创建权重参数 W，形状为 (d_model,)
        # 使用指定的设备和数据类型
        W = torch.empty(d_model, device=device, dtype=dtype)
        # 使用截断正态分布初始化权重
        # 均值为0，标准差为0.02，截断范围通常为[-2*std, 2*std]
        init.trunc_normal_(W, mean=0.0, std=0.02, a=-0.04, b=0.04)
        # 将权重包装为 nn.Parameter
        self.W = nn.Parameter(W)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """处理形状为(batch_size, sequence_length, d_model)的输入张量，返回相同形状的张量。"""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 公式：x * weight / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.W
        return result.to(in_dtype)


if __name__ == "__main__":
    eps = 1e-6
    x_test = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # 形状: (1, 2, 3)
    # 步骤 1: 平方
    x_squared = x_test.pow(2)
    print(f"1. x.pow(2) 形状: {x_squared.shape}")
    print(f"   平方结果:\n{x_squared}")

    # 步骤 2: 计算最后一个维度的均方值
    mean_squared = torch.mean(x_squared, dim=-1, keepdim=True)
    print(f"2. torch.mean(..., dim=-1, keepdim=True) 形状: {mean_squared.shape}")
    print(f"   均值结果:\n{mean_squared}")

    # 步骤 3: 计算 RMS
    rms = torch.sqrt(mean_squared + eps)
    print(f"3. RMS = sqrt(mean + eps) 形状: {rms.shape}")
    print(f"   RMS 结果:\n{rms}")

    # 步骤 4: 归一化
    normalized = x_test / rms
    print(f"4. x / RMS 形状: {normalized.shape}")
    print(f"   归一化 结果:\n{normalized}")
