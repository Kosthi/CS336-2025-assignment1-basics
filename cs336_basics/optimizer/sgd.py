from collections.abc import Callable
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"无效的学习率：{lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与参数p相关的状态
                t = state.get("t", 0)  # 从状态中获取迭代次数，若无则初始化为0
                grad = p.grad.data  # 获取损失相对于p的梯度
                # 原地更新权重张量
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # 递增迭代次数
        return loss


if __name__ == "__main__":
    lr = 1e3
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    print(f"learing rate: {lr}")
    for t in range(10):
        opt.zero_grad()  # 重置所有可学习参数的梯度
        loss = (weights**2).mean()  # 计算标量损失值
        print(f"step{t + 1:<2}, loss: {loss.cpu().item()}")
        loss.backward()  # 执行反向传播，计算梯度
        opt.step()  # 执行优化器步骤
