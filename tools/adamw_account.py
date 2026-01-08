import math


def calculate_memory_breakdown():
    """
    (a) Calculate peak memory breakdown for AdamW training
    """
    print("=" * 80)
    print("(a) 峰值内存分解（以字节为单位）")
    print("=" * 80)

    # 通用表达式
    print("\n1. 参数内存 M_params:")
    print("   M_params = 4 * P")
    print("   其中 P = 2*V*d + L*(16d² + 2d) + d")
    print()

    print("2. 梯度内存 M_grads:")
    print("   M_grads = 4 * P")
    print()

    print("3. 优化器状态内存 M_opt (AdamW):")
    print("   AdamW需要为每个参数存储两个状态张量（一阶矩和二阶矩）")
    print("   每个状态张量占4字节，共8字节/参数")
    print("   M_opt = 8 * P")
    print()

    print("4. 激活内存 M_act:")
    print("   激活内存基于前向传播中的中间变量：")
    print("   M_act = 4 * [L*(16*B*T*d + 2*B*h*T²) + B*T*d + 2*B*T*V]")
    print()

    print("5. 总峰值内存 M_total:")
    print("   M_total = M_params + M_grads + M_opt + M_act")
    print("           = 16P + 4[L*(16BTd + 2BhT²) + BTd + 2BTV]")
    print("   其中 P = 2*V*d + L*(16d² + 2d) + d")

    return True


def calculate_gpt2xl_instance(batch_size):
    """
    (b) Instantiate for GPT-2 XL and find max batch size for 80GB
    """
    print("\n" + "=" * 80)
    print("(b) GPT-2 XL 实例化")
    print("=" * 80)

    # GPT-2 XL 超参数
    V = 50257  # vocab_size
    T = 1024  # context_length
    L = 48  # num_layers
    d = 1600  # d_model
    h = 25  # num_heads (d = 1600, d/h = 64)

    print("GPT-2 XL 超参数:")
    print(f"  词表大小 V = {V:,}")
    print(f"  上下文长度 T = {T}")
    print(f"  层数 L = {L}")
    print(f"  模型维度 d = {d}")
    print(f"  注意力头数 h = {h}")

    # 计算参数数量 P
    # Embedding: V*d
    # Transformer layers: L*(16d² + 2d) (QKV投影: 3d², 输出投影: d², 注意力参数: 忽略偏置)
    #   实际更精确：每层: QKV(3d²) + 输出(d²) + FFN(第一个线性4d² + 第二个线性4d² + 第三个线性4d²) = 16d²
    #   加上两个RMSNorm参数: 2d (每层有两个RMSNorm)
    # Final layer norm: d
    embedding_params = V * d
    layer_params = L * (16 * d * d + 2 * d)  # 12d² + 2d
    final_norm_params = d

    P = 2 * embedding_params + layer_params + final_norm_params

    print(f"\n总参数数量 P = {P:,}")
    print(f"  词嵌入: {embedding_params:,}")
    print(f"  Transformer层: {layer_params:,}")
    print(f"  最终归一化: {final_norm_params:,}")

    # 计算固定内存（与batch size无关）
    fixed_memory_bytes = 16 * P  # 参数+梯度+优化器状态

    # 计算batch相关的内存（激活）
    # L*(16*B*T*d + 2*B*h*T²) + B*T*d + 2*B*T*V
    BTd = batch_size * T * d
    BhT2 = batch_size * h * T * T
    BTV = batch_size * T * V

    activation_memory_elements = L * (16 * BTd + 2 * BhT2) + BTd + 2 * BTV
    activation_memory_bytes = 4 * activation_memory_elements

    # 总内存
    total_memory_bytes = fixed_memory_bytes + activation_memory_bytes

    # 转换为GB
    fixed_memory_gb = fixed_memory_bytes / (1024**3)
    activation_memory_gb = activation_memory_bytes / (1024**3)
    total_memory_gb = total_memory_bytes / (1024**3)

    print(f"\n批次大小 B = {batch_size}:")
    print(f"  固定内存 (参数+梯度+优化器状态): {fixed_memory_gb:.2f} GB")
    print(f"  激活内存: {activation_memory_gb:.2f} GB")
    print(f"  总内存: {total_memory_gb:.2f} GB")

    # 计算最大批次大小（80GB内存限制）
    # 内存公式: M_total = fixed_memory_gb + a * batch_size
    # 其中 a = activation_memory_gb / batch_size
    a = activation_memory_gb / batch_size
    b = fixed_memory_gb

    print(f"\n内存模型: M_total ≈ {a:.2f} × B + {b:.2f} GB")

    max_memory_gb = 80
    max_batch_size = math.floor((max_memory_gb - b) / a)

    print(f"\n在 {max_memory_gb} GB 内存限制下:")
    print(f"  最大批次大小 B = {max_batch_size}")

    # 验证
    verification_memory = b + a * max_batch_size
    print(f"  验证内存使用: {verification_memory:.2f} GB")

    return a, b, max_batch_size


def calculate_adamw_flops():
    """
    (c) Calculate FLOPs for one step of AdamW
    """
    print("\n" + "=" * 80)
    print("(c) 一步 AdamW 的 FLOPs 计算")
    print("=" * 80)

    # GPT-2 XL 参数数量（重复使用前面的计算）
    V = 50257
    d = 1600
    L = 48

    embedding_params = V * d
    layer_params = L * (16 * d * d + 2 * d)
    final_norm_params = d
    P = 2 * embedding_params + layer_params + final_norm_params

    print(f"GPT-2 XL 参数总数 P = {P:,}")
    print("\nAdamW 更新每个参数约需 10 次浮点操作:")
    print("  1. 计算一阶矩估计: 2 FLOPs (乘法和加法)")
    print("  2. 计算二阶矩估计: 2 FLOPs (乘法和加法)")
    print("  3. 偏差修正: 2 FLOPs (幂运算和除法)")
    print("  4. 参数更新: 4 FLOPs (除法、平方根、乘法、减法)")
    print("  总计: ~10 FLOPs/参数")

    flops_adamw = 10 * P

    print("\n一步 AdamW 的总 FLOPs = 10 × P")
    print(f"                       = {flops_adamw:.2e}")

    return flops_adamw


def calculate_training_time(mfu_percent=50, steps=400000, batch_size=1024):
    """
    (d) Calculate training time for GPT-2 XL on a single A100
    """
    print("\n" + "=" * 80)
    print("(d) 单 A100 训练时间估算")
    print("=" * 80)

    # GPT-2 XL 参数数量
    V = 50257
    d = 1600
    L = 48
    T = 1024

    embedding_params = V * d
    layer_params = L * (16 * d * d + 2 * d)
    final_norm_params = d
    P = 2 * embedding_params + layer_params + final_norm_params

    # 每步的 FLOPs
    # 前向传播: 每个参数一次乘加操作 = 2 FLOPs
    # 后向传播: 约是前向的 2 倍
    # 总 FLOPs/步 ≈ 6 × B × T × P
    flops_per_step = 6 * batch_size * T * P

    print("每训练步的 FLOPs:")
    print("  前向传播: ~2 × B × T × P")
    print("  后向传播: ~4 × B × T × P (假设是前向的2倍)")
    print("  总计: ~6 × B × T × P")
    print("\n代入:")
    print(f"  B = {batch_size}, T = {T}, P = {P:.2e}")
    print(f"  FLOPs/步 = {flops_per_step:.2e}")

    # 总 FLOPs
    total_flops = flops_per_step * steps

    print(f"\n总训练步骤: {steps:,}")
    print(f"总 FLOPs = {total_flops:.2e}")

    # A100 理论峰值性能
    a100_peak_tflops = 19.5  # TFLOPS for float32
    a100_peak_flops = a100_peak_tflops * 1e12  # FLOPs/sec

    print("\nNVIDIA A100 理论峰值:")
    print(f"  {a100_peak_tflops} TFLOPS = {a100_peak_flops:.1e} FLOPs/sec")

    # 实际吞吐量
    mfu = mfu_percent / 100
    actual_throughput = a100_peak_flops * mfu

    print(f"\n模型 FLOPs 利用率 (MFU): {mfu_percent}%")
    print(f"实际吞吐量 = {actual_throughput:.1e} FLOPs/sec")

    # 训练时间
    training_time_seconds = total_flops / actual_throughput
    training_time_days = training_time_seconds / (24 * 3600)

    print("\n训练时间:")
    print(f"  {training_time_seconds:.1e} 秒")
    print(f"  {training_time_days:.1f} 天")
    print(f"  {training_time_days / 365:.1f} 年")

    return training_time_days


def main():
    """
    主函数：执行所有计算
    """
    print("AdamW 内存和计算资源分析")
    print("=" * 80)

    # (a) 内存分解
    calculate_memory_breakdown()

    # (b) GPT-2 XL 实例化（使用示例批次大小）
    sample_batch = 1
    a, b, max_batch = calculate_gpt2xl_instance(sample_batch)

    # 使用最大批次大小重新计算
    print(f"\n{'=' * 80}")
    print("使用最大批次大小重新计算内存使用:")
    calculate_gpt2xl_instance(max_batch)

    # (c) AdamW FLOPs
    flops_adamw = calculate_adamw_flops()

    # (d) 训练时间估算
    print(f"\n{'=' * 80}")
    print("训练配置:")
    print("  步骤数: 400,000")
    print("  批次大小: 1024")
    print("  上下文长度: 1024")
    print("  MFU: 50%")
    training_days = calculate_training_time()

    print(f"\n{'=' * 80}")
    print("总结:")
    print(f"  1. 峰值内存 ≈ {b:.2f} + {a:.2f}×B GB")
    print(f"  2. 80GB内存下最大批次大小: {max_batch}")
    print(f"  3. 一步 AdamW FLOPs: {flops_adamw:.2e}")
    print(f"  4. 单A100训练时间: {training_days:.1f} 天 ≈ {training_days / 365:.1f} 年")

    # 额外分析：不同MFU下的训练时间
    print(f"\n{'=' * 80}")
    print("不同MFU下的训练时间:")
    for mfu in [20, 30, 40, 50, 60]:
        days = calculate_training_time(mfu, 400000, 1024)
        print(f"  MFU={mfu}%: {days:.0f} 天")


if __name__ == "__main__":
    main()
