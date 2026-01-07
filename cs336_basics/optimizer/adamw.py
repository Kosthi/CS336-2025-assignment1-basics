import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"无效的学习率：{lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # 获取学习率等参数
            alpha = group["lr"]
            beta1, beat2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取与参数 p 相关的状态
                state = self.state[p]

                # 从状态中获取迭代次数，若无则初始化为 1
                t = state.get("t", 1)
                # 一阶矩估计
                m = state.get("m", torch.zeros_like(p))
                # 二阶矩估计
                v = state.get("v", torch.zeros_like(p))

                # 获取损失相对于p的梯度
                g = p.grad.data

                # 更新一、二阶矩估计
                m = beta1 * m + (1 - beta1) * g
                v = beat2 * v + (1 - beat2) * g * g

                # 计算当前迭代的调整后学习率 αt
                alpha_t = alpha * math.sqrt(1 - beat2**t) / (1 - beta1**t)
                # 更新参数
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                # 应用权重衰减
                p.data -= alpha * weight_decay * p.data

                # 递增迭代次数
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
