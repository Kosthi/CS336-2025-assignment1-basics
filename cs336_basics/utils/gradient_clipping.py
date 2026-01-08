import torch
from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    全局裁剪：所有梯度一起考虑
    计算总范数，如果超过阈值，按相同比例缩放所有梯度
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))

    # 如果需要裁剪
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        # 使用 foreach 进行向量化缩放，高性能
        torch._foreach_mul_(grads, clip_coef)
