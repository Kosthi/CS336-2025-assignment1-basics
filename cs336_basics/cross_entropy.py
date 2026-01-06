import torch
from .softmax import log_softmax


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # inputs: (batch_size, vocab_size)
    # targets: (batch_size, )
    # (batch_size, vocab_size)
    D = inputs.shape[0]

    # 1. 计算softmax概率
    probs = log_softmax(inputs, dim=-1)

    # 2. 提取目标位置的概率
    p = probs[torch.arange(D), targets]

    # 3. 计算平均损失（除以batch_size）
    return -torch.mean(p)
