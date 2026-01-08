#!/usr/bin/env python3
"""
Transformer语言模型资源核算脚本
用于计算GPT-2系列模型的参数量、内存占用和FLOPs
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置类"""

    name: str
    vocab_size: int = 50257  # GPT-2的词汇表大小
    context_length: int = 1024
    num_layers: int = 48
    d_model: int = 1600
    num_heads: int = 25
    d_ff: int = 6400

    def __post_init__(self):
        """计算派生参数"""
        self.d_k = self.d_model // self.num_heads  # 每个注意力头的维度
        self.d_v = self.d_k  # 通常d_k = d_v


class ResourceAccountant:
    """资源核算器"""

    @staticmethod
    def calculate_parameters(config: ModelConfig) -> dict[str, int]:
        """计算模型各部分参数数量"""
        params = {}

        # 1. 词嵌入层参数
        params["token_embedding"] = config.vocab_size * config.d_model

        # 2. Transformer层参数（每层）
        layer_params = {}

        # 注意力模块参数
        layer_params["attn_qkv"] = 3 * config.d_model * config.d_model  # Q, K, V投影
        layer_params["attn_output"] = config.d_model * config.d_model  # 输出投影

        # 前馈网络参数
        layer_params["ffn_first"] = config.d_ff * config.d_model  # 第一层（升维）
        layer_params["ffn_second"] = config.d_model * config.d_ff  # 第二层（降维）
        layer_params["ffn_third"] = config.d_ff * config.d_model  # 第一层（升维）

        # 层归一化参数（每个层归一化有 d_model 个参数：g）
        layer_params["ln1"] = config.d_model  # 注意力前的层归一化
        layer_params["ln2"] = config.d_model  # 前馈网络前的层归一化

        # 总层参数（乘以层数）
        for key, value in layer_params.items():
            params[f"layers_{key}"] = value * config.num_layers

        # 3. 归一化参数
        params["ln_final"] = config.d_model

        # 4. 输出层参数（语言模型头）
        params["lm_head"] = config.vocab_size * config.d_model

        # 5. 计算总参数
        params["total"] = sum(params.values())

        return params

    @staticmethod
    def calculate_flops(config: ModelConfig, include_embeddings: bool = False) -> dict[str, float]:
        """
        计算前向传播的FLOPs
        矩阵乘法FLOPs公式: 2 × m × n × p
        (m, n) * (n, p) = (m, p) (2n)*(m*p) = 2mnp
        """
        flops = {}
        seq_len = config.context_length  # 序列长度
        d_model = config.d_model
        d_ff = config.d_ff
        num_layers = config.num_layers
        d_k = config.d_k
        d_v = d_k
        vocab_size = config.vocab_size

        # 1. 词嵌入投影（通常只是查表，无矩阵乘法）
        flops["token_embedding"] = 0

        # 2. 单个Transformer层的FLOPs
        # 多头注意力模块
        # Q/K/V投影: (seq_len, d_model) × (d_model, d_model) -> (seq_len, d_model)，共3次
        flops_attn_qkv = 3 * (2 * seq_len * d_model * d_model)

        # 缩放点积注意力分数计算: (seq_len, d_model) × (d_model, seq_len) -> (seq_len, seq_len) 但需要分头计算
        # 实际计算: 每个头 (seq_len, d_k) × (d_k, seq_len) -> (seq_len, seq_len)
        # 注意: 这里使用简化计算 2 × S^2 × d (因为d = H × d_k)
        flops_attn_scores = 2 * seq_len * d_k * seq_len
        # 点积注意力值加权求和
        flops_attn_wv = 2 * seq_len * seq_len * d_v  # (seq_q, seq_k) * (seq_k, d_v) -> (seq_q, d_v)
        # flops_attn_scores = 2 * seq_len * seq_len * d_model  # 等价于 2 × S² × d

        # 多头注意力输出投影: (seq_len, d_model) × (d_model, d_model) -> (seq_len, d_model)
        flops_attn_output = 2 * seq_len * d_model * d_model

        # 前馈网络
        # 第一层（升维）: (seq_len, d_model) × (d_model, d_ff) -> (seq_len, d_ff)
        flops_ffn_first = 2 * seq_len * d_model * d_ff

        # 第二层（降维）: (seq_len, d_ff) × (d_ff, d_model) -> (seq_len, d_model)
        flops_ffn_second = 2 * seq_len * d_ff * d_model

        # 第三层（升维）: (seq_len, d_model) × (d_model, d_ff) -> (seq_len, d_ff)
        flops_ffn_third = 2 * seq_len * d_model * d_ff

        # 层归一化和激活函数的FLOPs（近似估计）
        # 层归一化: ~4 × S × d FLOPs（计算均值、方差、归一化、缩放平移）
        # 点积操作，忽略
        # flops_layer_norm = 4 * seq_len * d_model

        # 单个层的总FLOPs
        single_layer_flops = {
            "attn_qkv": flops_attn_qkv,
            "attn_scores": flops_attn_scores,
            "attn_wv": flops_attn_wv,
            "attn_output": flops_attn_output,
            "ffn_first": flops_ffn_first,
            "ffn_second": flops_ffn_second,
            "ffn_third": flops_ffn_third,
        }

        # 所有层的FLOPs
        for key, value in single_layer_flops.items():
            flops[f"layers_{key}"] = value * num_layers

        # 3. 输出层（语言模型头）
        # (sel_len, d_model) × (d_model, vocab_size) -> (seq_len, vocab_size)
        flops["lm_head"] = 2 * seq_len * d_model * vocab_size

        # 4. 总FLOPs（排除嵌入层，因为它们不涉及矩阵乘法）
        flops["total_without_embeddings"] = sum(
            [v for k, v in flops.items() if k != "token_embedding" and k != "position_embedding"]
        )

        # 如果需要包括嵌入层（虽然它们主要是查表操作）
        if include_embeddings:
            # 词嵌入查表操作（近似）: S × d FLOPs
            flops["token_embedding_approx"] = seq_len * d_model
            # 位置嵌入加法: S × d FLOPs
            flops["position_embedding_approx"] = seq_len * d_model
            flops["total"] = flops["total_without_embeddings"] + seq_len * d_model * 2
        else:
            flops["total"] = flops["total_without_embeddings"]

        return flops

    @staticmethod
    def calculate_memory_usage(params: dict[str, int], precision_bits: int = 32) -> dict[str, float]:
        """计算内存使用情况（以GB为单位）"""
        # 参数内存（以字节为单位）
        bytes_per_param = precision_bits / 8

        memory = {}
        for key, value in params.items():
            memory[key] = value * bytes_per_param / (1024**3)  # 转换为GB

        # 激活内存估计（近似）
        # 注意：这是一个简化估计，实际激活内存取决于实现和优化
        memory["estimated_activations"] = params.get("total", 0) * bytes_per_param / (1024**3) * 0.5

        return memory

    @staticmethod
    def analyze_flops_distribution(flops: dict[str, float]) -> dict[str, float]:
        """分析FLOPs分布"""
        total = flops.get("total", 0)
        if total == 0:
            return {}

        distribution = {}
        for key, value in flops.items():
            if key not in ["total", "total_without_embeddings"] and value > 0:
                distribution[key] = value / total * 100

        return distribution

    @staticmethod
    def print_model_report(config: ModelConfig):
        """打印模型资源报告"""
        print(f"\n{'=' * 60}")
        print(f"模型: {config.name}")
        print(f"{'=' * 60}")
        print("配置:")
        print(f"  层数: {config.num_layers}")
        print(f"  模型维度: {config.d_model}")
        print(f"  注意力头数: {config.num_heads}")
        print(f"  前馈网络维度: {config.d_ff}")
        print(f"  序列长度: {config.context_length}")
        print(f"  词汇表大小: {config.vocab_size}")

        # 计算参数
        params = ResourceAccountant.calculate_parameters(config)

        print("\n参数数量:")
        print(f"  总参数: {params['total']:,}")
        print(f"  词嵌入: {params['token_embedding']:,} ({params['token_embedding'] / params['total'] * 100:.2f}%)")
        print(
            f"  注意力Q/K/V投影: {params.get('layers_attn_qkv', 0):,} ({params.get('layers_attn_qkv', 0) / params['total'] * 100:.2f}%)"
        )
        print(
            f"  注意力输出投影: {params.get('layers_attn_output', 0):,} ({params.get('layers_attn_output', 0) / params['total'] * 100:.2f}%)"
        )
        print(
            f"  前馈网络: {params.get('layers_ffn_first', 0) + params.get('layers_ffn_second', 0):,} ({(params.get('layers_ffn_first', 0) + params.get('layers_ffn_second', 0)) / params['total'] * 100:.2f}%)"
        )
        print(
            f"  层归一化: {params.get('layers_ln1', 0) + params.get('layers_ln2', 0):,} ({(params.get('layers_ln1', 0) + params.get('layers_ln2', 0)) / params['total'] * 100:.2f}%)"
        )
        print(f"  归一化: {params['ln_final']:,} ({params['ln_final'] / params['total'] * 100:.2f}%)")
        print(f"  输出层: {params['lm_head']:,} ({params['lm_head'] / params['total'] * 100:.2f}%)")

        # 计算内存使用
        memory = ResourceAccountant.calculate_memory_usage(params)
        print(f"\n内存占用 (FP{32}):")
        print(f"  总参数内存: {memory['total']:.2f} GB")
        print(f"  估计激活内存: {memory['estimated_activations']:.2f} GB")
        print(f"  总计 (参数+激活): {memory['total'] + memory['estimated_activations']:.2f} GB")

        # 计算FLOPs
        flops = ResourceAccountant.calculate_flops(config, include_embeddings=False)

        print(f"\n前向传播FLOPs (序列长度={config.context_length}):")
        FFN_flops = flops["layers_ffn_first"] + flops["layers_ffn_second"] + flops["layers_ffn_third"]
        QKV_flops = flops["layers_attn_qkv"]
        lm_head_flops = flops["lm_head"]
        print(f"  FNNs: {FFN_flops:.2e}")
        print(f"  attn_QKVs: {QKV_flops:.2e}")
        print(f"  lm_head: {lm_head_flops:.2e}")
        total_flops = flops["total"]
        if total_flops > 1e12:
            print(f"  总FLOPs: {total_flops:.2e}(TFLOPs)")
        elif total_flops > 1e9:
            print(f"  总FLOPs: {total_flops:.2e}(GFLOPs)")
        else:
            print(f"  总FLOPs: {total_flops:.2e}(MFLOPs)")

        # 分析FLOPs分布
        distribution = ResourceAccountant.analyze_flops_distribution(flops)

        print("\nFLOPs分布:")
        print(f"  注意力Q/K/V投影: {distribution.get('layers_attn_qkv', 0):.2f}%")
        print(f"  点积注意力分数计算: {distribution.get('layers_attn_scores', 0):.2f}%")
        print(f"  点积注意力权重值计算: {distribution.get('layers_attn_wv', 0):.2f}%")
        print(f"  注意力输出投影: {distribution.get('layers_attn_output', 0):.2f}%")
        print(f"  前馈网络第一层: {distribution.get('layers_ffn_first', 0):.2f}%")
        print(f"  前馈网络第二层: {distribution.get('layers_ffn_second', 0):.2f}%")
        print(f"  前馈网络第三层: {distribution.get('layers_ffn_third', 0):.2f}%")
        # print(f"  层归一化: {distribution.get('layers_layer_norm', 0):.2f}%")
        # print(f"  激活函数: {distribution.get('layers_activation', 0):.2f}%")
        print(f"  输出层: {distribution.get('lm_head', 0):.2f}%")

        # 计算前馈网络总占比
        ffn_total = (
            distribution.get("layers_ffn_first", 0)
            + distribution.get("layers_ffn_second", 0)
            + distribution.get("layers_ffn_third", 0)
        )
        print(f"  前馈网络总计: {ffn_total:.2f}%")

        # 计算注意力总占比（不包括层归一化和激活）
        attn_total = (
            distribution.get("layers_attn_qkv", 0)
            + distribution.get("layers_attn_scores", 0)
            + distribution.get("layers_attn_wv", 0)
            + distribution.get("layers_attn_output", 0)
        )
        print(f"  注意力总计: {attn_total:.2f}%")

        return params, flops, memory


def compare_gpt2_models():
    """比较不同GPT-2模型的资源使用"""
    # GPT-2模型配置 (基于公开信息)
    gpt2_configs = [
        ModelConfig(name="GPT-2 Small", num_layers=12, d_model=768, num_heads=12, d_ff=3072),
        ModelConfig(name="GPT-2 Medium", num_layers=24, d_model=1024, num_heads=16, d_ff=4096),
        ModelConfig(name="GPT-2 Large", num_layers=36, d_model=1280, num_heads=20, d_ff=5120),
        ModelConfig(name="GPT-2 XL", num_layers=48, d_model=1600, num_heads=25, d_ff=6400),
    ]

    results = []

    for config in gpt2_configs:
        params, flops, memory = ResourceAccountant.print_model_report(config)

        # 存储结果用于比较
        results.append(
            {
                "name": config.name,
                "params": params["total"],
                "flops": flops["total"],
                "memory_gb": memory["total"],
                "ffn_percentage": (
                    flops.get("layers_ffn_first", 0)
                    + flops.get("layers_ffn_second", 0)
                    + flops.get("layers_ffn_third", 0)
                )
                / flops["total"]
                * 100,
                "attn_qkv": flops.get("layers_attn_qkv", 0) / flops["total"] * 100,
                "attn_wv": flops.get("layers_attn_wv", 0) / flops["total"] * 100,
                "attn_output": flops.get("layers_attn_output", 0) / flops["total"] * 100,
                "attn_percentage": (
                    flops.get("layers_attn_qkv", 0)
                    + flops.get("layers_attn_scores", 0)
                    + flops.get("layers_attn_wv", 0)
                    + flops.get("layers_attn_output", 0)
                )
                / flops["total"]
                * 100,
                "attn_scores": flops.get("layers_attn_scores", 0) / flops["total"] * 100,
                "lm_head": flops.get("lm_head", 0) / flops["total"] * 100,
            }
        )

    # 打印比较表格
    print(f"\n{'=' * 60}")
    print("GPT-2模型比较")
    print(f"{'=' * 60}")
    print(
        f"{'模型':<10} {'参数':<10} {'FLOPs':<10} {'内存(GB)':<10}"
        f"{'多头注意力QKV占比':<10} {'点积注意力分数占比':<10} {'点积注意力加权求和':<10} {'多头注意力输出投影':<10} {'FFN占比':<10} {'输出层':<10}"
    )
    print(f"{'-' * 100}")

    for result in results:
        print(
            f"{result['name']:<10} {result['params'] / 1e6:>8.2f}M {result['flops'] / 1e12:>10.2f}T {result['memory_gb']:>8.2f} "
            f"{result['attn_qkv']:>12.2f}% {result['attn_scores']:>15.2f}% {result['attn_wv']:>17.2f}% {result['attn_output']:>17.2f}%"
            f"{result['ffn_percentage']:>17.2f}% {result['lm_head']:>15.2f}%"
        )

    return results


def analyze_context_length_impact():
    """分析上下文长度对FLOPs的影响"""
    print(f"\n{'=' * 60}")
    print("上下文长度对GPT-2 XL的影响")
    print(f"{'=' * 60}")

    config = ModelConfig(name="GPT-2 XL", num_layers=48, d_model=1600, num_heads=25, d_ff=6400)

    context_lengths = [1024, 2048, 4096, 8192, 16384]

    print(
        f"{'序列长度':<12} {'总FLOPs':<15} {'FFN占比':<10} {'注意力占比':<12} {'输出层占比':<12} {'注意力分数占比':<15} {'注意力分数增长率':<18}"
    )
    print(f"{'-' * 90}")

    prev_attn_scores = None
    for seq_len in context_lengths:
        config.context_length = seq_len
        flops = ResourceAccountant.calculate_flops(config, include_embeddings=False)

        total_flops = flops["total"]
        ffn_percentage = (
            (flops.get("layers_ffn_first", 0) + flops.get("layers_ffn_second", 0) + flops.get("layers_ffn_third", 0))
            / total_flops
            * 100
        )
        attn_percentage = (
            (
                flops.get("layers_attn_qkv", 0)
                + flops.get("layers_attn_scores", 0)
                + flops.get("layers_attn_wv", 0)
                + flops.get("layers_attn_output", 0)
            )
            / total_flops
            * 100
        )
        lm_head = (flops.get("lm_head", 0)) / total_flops * 100
        attn_scores_percentage = flops.get("layers_attn_scores", 0) / total_flops * 100

        # 计算注意力分数FLOPs的增长率
        current_attn_scores = flops.get("layers_attn_scores", 0)
        if prev_attn_scores is not None:
            growth_rate = current_attn_scores / prev_attn_scores
        else:
            growth_rate = 1.0

        print(
            f"{seq_len:<12} {total_flops / 1e12:>10.2f}T {ffn_percentage:>15.2f}% {attn_percentage:>12.2f}% "
            f"{lm_head:>15.2f}% {attn_scores_percentage:>15.2f}% {growth_rate:>20.2f}x"
        )

        prev_attn_scores = current_attn_scores


def main():
    """主函数"""
    print("Transformer语言模型资源核算脚本")
    print("=" * 60)

    # 1. 分析GPT-2 XL
    print("\n1. GPT-2 XL详细分析:")
    config_xl = ModelConfig(name="GPT-2 XL")
    ResourceAccountant.print_model_report(config_xl)

    # 2. 比较所有GPT-2模型
    print("\n2. GPT-2系列模型比较:")
    compare_gpt2_models()

    # 3. 分析上下文长度影响
    print("\n3. 上下文长度影响分析:")
    analyze_context_length_impact()

    # 4. 额外分析：不同精度的影响
    print(f"\n{'=' * 60}")
    print("不同精度对内存的影响")
    print(f"{'=' * 60}")

    config = ModelConfig(name="GPT-2 XL")
    params = ResourceAccountant.calculate_parameters(config)

    precisions = [
        ("FP32", 32),
        ("FP16/BF16", 16),
        ("INT8", 8),
        ("INT4", 4),
    ]

    print(f"{'精度':<12} {'参数内存(GB)':<15} {'激活内存(GB)':<15} {'总计(GB)':<12}")
    print(f"{'-' * 60}")

    for precision_name, bits in precisions:
        memory = ResourceAccountant.calculate_memory_usage(params, precision_bits=bits)
        param_memory = memory["total"]
        activation_memory = memory["estimated_activations"]
        total_memory = param_memory + activation_memory

        print(f"{precision_name:<12} {param_memory:>10.2f} {activation_memory:>15.2f} {total_memory:>20.2f}")


if __name__ == "__main__":
    main()
