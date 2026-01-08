import json
from collections import defaultdict
import os


def analyze_vocab(vocab_file):
    """分析词汇表中最长的令牌"""
    # 1. 加载词汇表
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"词汇表总大小: {len(vocab)} 个令牌")

    # 2. 分析词汇表结构
    # 取第一个键值对看看结构
    first_key, first_value = next(iter(vocab.items()))
    print(f"词汇表结构示例 - 键: {repr(first_key)}, 值: {repr(first_value)}")

    # 判断是 id->token 还是 token->id
    is_id_to_token = first_key.isdigit() or (isinstance(first_key, str) and first_key.isdigit())

    if is_id_to_token:
        print("检测到词汇表格式: id -> token")
        # 交换键值对，变成 token -> id 方便分析
        token_to_id = {v: int(k) for k, v in vocab.items()}
        vocab = token_to_id
    else:
        print("检测到词汇表格式: token -> id")
        # 确保所有 id 都是整数
        vocab = {k: int(v) if isinstance(v, str) else v for k, v in vocab.items()}

    # 3. 找出最长的令牌（按字节长度）
    max_length = 0
    longest_tokens = []
    token_lengths = defaultdict(list)

    for token_str, token_id in vocab.items():
        # 将字符串转换为字节（假设是UTF-8编码）
        try:
            token_bytes = token_str.encode("utf-8", errors="replace")
        except Exception:
            # 如果 token_str 不是字符串，跳过
            continue

        token_len = len(token_bytes)

        # 统计长度分布
        token_lengths[token_len].append(token_str)

        # 更新最长令牌
        if token_len > max_length:
            max_length = token_len
            longest_tokens = [(token_str, token_len, token_id)]
        elif token_len == max_length:
            longest_tokens.append((token_str, token_len, token_id))

    # 4. 打印结果
    print(f"\n最长的令牌长度: {max_length} 字节")
    print(f"最长的令牌数量: {len(longest_tokens)} 个")

    if longest_tokens:
        print("\n最长的令牌列表 (前20个):")
        for token_str, token_len, token_id in sorted(longest_tokens)[:20]:
            try:
                # 尝试解码显示
                decoded = token_str.encode("utf-8").decode("utf-8", errors="replace")
                print(f"  ID: {token_id:6d} | 字节长度: {token_len:3d} | 内容: {repr(decoded)}")
            except Exception:
                # 如果 token_id 不是整数，使用字符串显示
                print(f"  ID: {str(token_id):6s} | 字节长度: {token_len:3d} | 内容: [无法解码]")
    else:
        print("没有找到任何令牌")

    # 5. 分析长度分布
    print("\n令牌长度分布:")
    for length in sorted(token_lengths.keys()):
        count = len(token_lengths[length])
        percentage = (count / len(vocab)) * 100 if vocab else 0

        # 只显示有令牌的长度
        if count > 0:
            sample_tokens = token_lengths[length][:3]  # 显示每个长度的前3个样本

            # 处理显示
            sample_display = []
            for token in sample_tokens:
                try:
                    decoded = token.encode("utf-8").decode("utf-8", errors="replace")
                    if len(decoded) > 20:
                        decoded = decoded[:20] + "..."
                    sample_display.append(repr(decoded))
                except Exception:
                    sample_display.append("[binary]")

            print(f"  长度 {length:2d} 字节: {count:6d} 个 ({percentage:5.1f}%) | 样本: {', '.join(sample_display)}")

    # 6. 详细分析最长令牌
    if max_length <= 10:  # 如果最大长度很小，分析所有长度的令牌
        print("\n详细分析所有令牌:")

        # 按长度分组
        length_groups = {}
        for length, tokens in token_lengths.items():
            length_groups[length] = tokens[:5]  # 取每个长度的前5个

        for length in sorted(length_groups.keys()):
            tokens = length_groups[length]
            print(f"\n长度 {length} 字节的令牌样本:")
            for i, token in enumerate(tokens):
                try:
                    decoded = token.encode("utf-8").decode("utf-8", errors="replace")
                    print(f"  {i + 1}. {repr(decoded)}")
                except Exception:
                    print(f"  {i + 1}. [二进制数据]")

                # 显示原始字节
                bytes_repr = token.encode("utf-8")
                hex_repr = bytes_repr.hex()
                print(f"     十六进制: {hex_repr}")

    return max_length, longest_tokens, token_lengths


# 运行分析
vocab_file = "owt_test_results/owt_vocab.json"
if os.path.exists(vocab_file):
    print(f"分析词汇表: {vocab_file}")
    max_len, longest_tokens, length_dist = analyze_vocab(vocab_file)

    # 输出总结
    print("\n" + "=" * 80)
    print("分析总结:")
    print(f"  1. 词汇表大小: {len(length_dist)} 种不同长度的令牌")
    print(f"  2. 最长令牌: {max_len} 字节")
    print(f"  3. 最长令牌示例: {longest_tokens[0][0] if longest_tokens else '无'}")
    print("  4. 令牌长度分布:")

    # 计算统计信息
    all_lengths = []
    for length, tokens in length_dist.items():
        all_lengths.extend([length] * len(tokens))

    if all_lengths:
        avg_length = sum(all_lengths) / len(all_lengths)
        median_length = sorted(all_lengths)[len(all_lengths) // 2]
        print(f"     平均长度: {avg_length:.2f} 字节")
        print(f"     中位数长度: {median_length} 字节")

        # 检查是否有超过10字节的令牌
        long_tokens = [length_val for length_val in all_lengths if length_val > 10]
        if long_tokens:
            print(f"     警告: 有 {len(long_tokens)} 个令牌长度超过10字节")
        else:
            print("     所有令牌长度 ≤ 10 字节")

    print("=" * 80)
else:
    print(f"文件 {vocab_file} 不存在")

    # 查找可能的文件
    print("\n当前目录下的JSON文件:")
    for file in os.listdir("."):
        if file.endswith(".json"):
            print(f"  - {file}")
