import torch
import numpy as np
from typing import Tuple, Union
import random


class DataLoader:
    """
    数据加载器类，用于管理数据集和批量生成。
    """

    def __init__(
        self,
        dataset: Union[str, np.ndarray],  # 修改：支持直接传入numpy数组
        batch_size: int,
        context_length: int,
        device: str = "cpu",
        mmap_mode: str = "r",
        dtype: np.dtype = np.uint16,
    ):
        """
        初始化数据加载器。

        参数:
            data: 可以是numpy数组或.npy文件路径
            batch_size: 批量大小
            context_length: 上下文长度
            device: 目标设备
            mmap_mode: 内存映射模式，'r'表示只读（仅当data是文件路径时有效）
            dtype: 数据存储的数据类型
        """
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        # 根据data类型决定如何加载数据
        if isinstance(dataset, str):
            # 如果是文件路径，使用内存映射加载
            # print(f"从文件加载数据: {dataset}")
            self.data = np.load(dataset, mmap_mode=mmap_mode).astype(dtype)
            self.data_source = "file"
        elif isinstance(dataset, np.ndarray):
            # 如果直接是numpy数组，直接使用
            # print(f"直接使用numpy数组")
            self.data = dataset.astype(dtype)
            self.data_source = "array"
        else:
            raise TypeError(f"data必须是字符串（文件路径）或numpy数组，但得到 {type(data)}")

        # 验证数据
        self._validate_data()

        # print(f"数据集大小: {len(self.data):,} tokens")
        # print(f"数据类型: {self.data.dtype}")
        # print(f"数据范围: [{self.data.min()}, {self.data.max()}]")

    def _validate_data(self):
        """验证加载的数据。"""
        # 验证输入数据
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data必须是numpy数组，但得到 {type(self.data)}")

        if len(self.data) < self.context_length + 1:
            raise ValueError(f"数据长度({len(self.data)})必须至少为context_length+1({self.context_length + 1})")

        # 验证token ID范围（假设词汇表大小合理）
        if self.data.dtype not in [np.int32, np.int64, np.uint32, np.uint64, np.int16, np.uint16]:
            print(f"警告: 数据类型{self.data.dtype}可能不是整数类型")

    def get_batch(self, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从token序列中随机采样一个批量。

        参数:
            data: token ID的numpy数组，可以是内存映射数组
            batch_size: 批量大小
            context_length: 上下文长度
            device: 目标设备 ('cpu', 'cuda:0', 'mps')
            seed: 随机种子（用于测试）

        返回:
            inputs: 输入序列张量，形状为(batch_size, context_length)
            targets: 目标序列张量，形状为(batch_size, context_length)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 计算有效起始位置：从0到len(data)-context_length
        # 100 - 7 = 93
        max_start_idx = len(self.data) - self.context_length

        # 随机采样批量大小的起始位置
        # randint 右边界取开
        # 0~92 1~93
        start_indices = np.random.randint(0, max_start_idx, size=self.batch_size)

        # 预分配批量数组
        inputs = np.empty((self.batch_size, self.context_length), dtype=self.data.dtype)
        targets = np.empty((self.batch_size, self.context_length), dtype=self.data.dtype)

        # 填充批量
        for i, start_idx in enumerate(start_indices):
            end_idx = start_idx + self.context_length
            inputs[i] = self.data[start_idx:end_idx]
            targets[i] = self.data[start_idx + 1 : end_idx + 1]

        # 转换为 PyTorch 张量并移动到指定设备
        inputs_tensor = torch.from_numpy(inputs).long().to(self.device)
        targets_tensor = torch.from_numpy(targets).long().to(self.device)

        return inputs_tensor, targets_tensor
