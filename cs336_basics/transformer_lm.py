import torch
import torch.nn as nn
from .Embedding import Embedding
from .transformer_block import TransformerBlock
from .RMSNorm import RMSNorm
from .Linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        *,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        """
        TransformerLM 是整个训练过程的封装，它把包含 Embedding、TransformerBlock、RMSNorm、Linear
        Args:
            vocab_size (int): 词表大小
            context_length (int): 上下文长度
            d_model (int): 输入的维度，也就是d_model
            num_Layers (int): 层数
            num_heads (int): 头的数量
            d_ff (int): 前馈神经网络的维度
            rope_theta (float): 底数超参数
            weights (dict [str, torch. Tensor]): ti
            input:
                in_indices (torch.Tensor): 输入的索引
            output:
                out_linear (torch.Tensor): 输出的线性
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device)

        # 创建 Transformer Block 列表
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    use_rmsnorm=use_rmsnorm,
                    norm_style=norm_style,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.rms_norm = RMSNorm(d_model, device=device) if use_rmsnorm else nn.Identity()
        self.linear = Linear(in_features=d_model, out_features=vocab_size, device=device)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
        """
        # token embedding
        embedding = self.embedding(in_indices)
        # block
        for block in self.blocks:
            embedding = block(embedding)
        # norm
        embedding = self.rms_norm(embedding)
        # output embedding
        embedding = self.linear(embedding)
        return embedding
