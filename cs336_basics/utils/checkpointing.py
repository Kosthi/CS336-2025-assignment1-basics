import torch
import os
import typing


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    config: dict[str, typing.Any] | None = None,
):
    """
    保存模型、优化器和训练迭代次数的检查点。

    参数:
        model: PyTorch模型
        optimizer: PyTorch优化器
        iteration: 当前训练迭代次数
        out: 输出路径或类文件对象
    """
    base_model = _unwrap_model(model)
    checkpoint = {
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device | None = None,
):
    """
    从检查点加载模型、优化器和训练迭代次数。

    参数:
        src: 检查点路径或类文件对象
        model: PyTorch模型（将加载状态）
        optimizer: PyTorch优化器（将加载状态）

    返回:
        iteration: 保存的训练迭代次数
    """
    checkpoint = torch.load(src, map_location=map_location)
    base_model = _unwrap_model(model)
    model_state_dict = checkpoint["model_state_dict"]
    if isinstance(model_state_dict, dict):
        model_state_dict = _strip_prefix(model_state_dict, "_orig_mod.")
    base_model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
