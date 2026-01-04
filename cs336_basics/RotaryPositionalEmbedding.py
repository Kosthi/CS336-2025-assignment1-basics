import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        构建RoPE模块，创建必要的缓冲区。
        参数：
            theta: float：RoPE的Θ值
            d_k: int：查询和键向量的维度
            max_seq_len: int：输入的最大序列长度
            device: torch.device | None = None：缓冲区存储设备
        """
        super().__init__()
        # 验证d_k必须是偶数
        if d_k % 2 != 0:
            raise ValueError(f"d_k必须是偶数，当前d_k={d_k}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 创建位置索引（0到max_seq_len-1）
        position = torch.arange(max_seq_len, device=device).float()
        # 创建维度索引（0到d_k/2-1）
        i = torch.arange(0, d_k, 2, device=device).float() / d_k

        # 计算频率矩阵：shape = (max_seq_len, d_k/2)
        # 公式: freqs = 1.0 / (theta ** (2 * i / d_k))
        freqs = 1.0 / (theta**i)

        # 计算位置和频率的乘积：shape = (max_seq_len, d_k/2)
        # 每个位置m，每个维度i：m * theta^(-2i/d_k)
        # 高效计算一维张量的外积
        angles = torch.outer(position, freqs)
        # 计算sin和cos值
        # 最终shape: (max_seq_len, d_k/2)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        # 将计算好的cos和sin值注册为缓冲区
        # 这些是模型的不可训练参数
        self.register_buffer("cos_cached", cos_vals, persistent=False)
        self.register_buffer("sin_cached", sin_vals, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        处理形状为(..., seq_len, d_k)的输入张量，返回相同形状的张量。
        支持输入包含任意数量的批处理维度。
        token_positions是形状为(..., seq_len)的张量，表示x在序列维度上的令牌位置。
        """
        # 根据token_positions获取对应的cos和sin值
        # 使用高级索引获取指定位置的cos和sin值
        cos_selected = self.cos_cached[token_positions]  # shape: (batch_size, seq_len, d_k)
        sin_selected = self.sin_cached[token_positions]

        # 比如 x: shape(4, 12, 64)，4句话，每句话有12个token，每个token用64维向量编码
        # token_positions(0, 1, 2, ..., 11)
        # cos_selected: shape(12, 64)，12个位置，64个维度每2个维度旋转的角度不一样
        # sin_selected: shape(12, 64)

        x1 = x[..., 0::2]  # x 偶数位置 0, 2, 4
        x2 = x[..., 1::2]  # x 奇数位置 1, 3, 5
        x1_rotated = x1 * cos_selected - x2 * sin_selected
        x2_rotated = x2 * cos_selected + x1 * sin_selected

        # 堆叠张量，增加最后一维 2，使奇偶位置拼在一起 (0, 1), (2, 3), (4, 5)...
        # x_rotated: shape(4, 12, 32, 2)
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        # 最后两个维度展平，按顺序连接 (0, 1, 2, 3, 4, 5, ...)
        # x_rotated: shape(4, 12, 64)
        x_rotated = torch.flatten(x_rotated, start_dim=-2, end_dim=-1)

        return x_rotated


if __name__ == "__main__":
    # 假设我们有3个位置和4个频率分量
    seq_len = 16
    d = 32  # 嵌入维度
    theta = 10000

    # 位置索引
    # position = torch.arange(seq_len, dtype=torch.float32)  # tensor([0., 1., 2.])
    position = torch.arange(seq_len).float()
    # 频率值（通常是theta^{-2i/d}形式）
    # freqs = torch.tensor([0.1, 0.2], dtype=torch.float32)  # d/2 = 2个频率
    i = torch.arange(0, d, 2).float() / d
    freqs = 1.0 / (theta**i)
    # print(position)
    # print(i)
    # print(freqs)
    # 广播乘法
    angles = position[:, None] * freqs[None, :]
    # print("position[:, None] 形状:", position[:, None].shape)
    # print(position[:, None])
    # print("freqs[None, :] 形状:", freqs[None, :].shape)
    # print(freqs[None, :])
    # print("angles 形状:", angles.shape)
    # print("angles 值:\n", angles)

    cos_vals = torch.cos(angles).repeat_interleave(2, dim=1)
    sin_vals = torch.sin(angles).repeat_interleave(2, dim=1)
    print("cos", cos_vals)
    print("sin", sin_vals.shape)
