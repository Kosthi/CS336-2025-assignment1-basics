import torch


def softmax(x: torch.Tensor, dim: int):
    """
    Args:
        x: 输入张量，形状为 (...,)

    Returns:
        输出张量，形状为 (...,)
    """
    weights = torch.exp(x - torch.max(x))
    t = torch.sum(weights, dim=dim, keepdim=True)
    return weights / t
