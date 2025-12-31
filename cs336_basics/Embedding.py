import torch
import torch.nn as nn
from torch.nn import init


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        构建嵌入模块。
        参数：
            num_embeddings: int：词汇表大小
            embedding_dim: int：嵌入向量维度（即d_model）
            device: torch.device | None = None：参数存储设备
            dtype: torch.dtype | None = None：参数数据类型
        """
        super().__init__()
        # 创建权重参数 W，形状为 (out_features, in_features)
        # 使用指定的设备和数据类型
        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        # 使用截断正态分布初始化权重
        # 均值为0，标准差为0.02，截断范围通常为[-2*std, 2*std]
        init.trunc_normal_(W, mean=0.0, std=0.02, a=-0.04, b=0.04)
        # 将权重包装为 nn.Parameter
        self.W = nn.Parameter(W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """为给定令牌ID查找嵌入向量。"""
        return self.W[token_ids]


if __name__ == "__main__":
    # 创建 Embedding 层
    vocab_size = 3
    embedding_dim = 4
    embedding = nn.Embedding(vocab_size, embedding_dim)

    # 查看权重矩阵 W
    W = embedding.weight
    print(f"W 形状: {W.shape}")  # (3, 4)
    print(W)

    # 创建 token_ids
    token_ids = torch.tensor([[0, 1, 2], [2, 0, 1]])
    print(f"token_ids 形状: {token_ids.shape}")  # (2, 3)

    # Embedding 查找
    result = embedding(token_ids)
    # 支持使用张量作为索引，并行索引查找
    result1 = W[token_ids]  # 结果是一个三维矩阵，python 输出是从上往下看
    print(f"result 形状: {result.shape}")  # (2, 3, 4)
    print("Embedding查找", result)
    print("并行索引查找", result1)
    print(result == result1)
