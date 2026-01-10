import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"无效的学习率：{lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # 获取学习率等参数
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取与参数 p 相关的状态
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # 递增迭代次数
                state["t"] += 1

                # 引用非拷贝
                # 从状态中获取迭代次数，若无则初始化为 1
                t = state["t"]
                # 一阶矩估计
                m = state["m"]
                # 二阶矩估计
                v = state["v"]

                # 获取损失相对于p的梯度
                g = p.grad.data

                # 更新一、二阶矩估计，原地更新
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # 应用权重衰减
                if group["weight_decay"] != 0:
                    p.mul_(1 - alpha * weight_decay)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                # 计算分母和步长，使用原论文里的计算方法
                denom = (v / bias_correction2).sqrt().add_(eps)
                step_size = alpha / bias_correction1

                # 更新参数
                p.addcdiv_(m, denom, value=-step_size)

        return loss
