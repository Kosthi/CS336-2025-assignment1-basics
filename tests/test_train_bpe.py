import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode, DATA_PATH, HOST_DATA_PATH
import json
import os
import time


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )


def save_bpe_results(vocab, merges, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    vocab_file = os.path.join(output_dir, f"{name}_vocab.json")
    merges_file = os.path.join(output_dir, f"{name}_merges.txt")

    # Prepare vocab for JSON (keys must be strings, values are strings for readability)
    # The vocab is dict[int, bytes]
    vocab_export = {}
    for idx, token in vocab.items():
        if isinstance(token, bytes):
            try:
                token_str = token.decode("utf-8")
            except UnicodeDecodeError:
                token_str = token.hex()  # Fallback for non-utf8 bytes
        else:
            token_str = str(token)
        vocab_export[str(idx)] = token_str

    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_export, f, indent=2, ensure_ascii=False)

    # Save merges
    with open(merges_file, "w", encoding="utf-8") as f:
        for t1, t2 in merges:

            def to_str_for_file(t):
                if isinstance(t, bytes):
                    s = t.decode("utf-8", errors="replace")
                else:
                    s = str(t)
                return json.dumps(s, ensure_ascii=False)

            f.write(f"{to_str_for_file(t1)}\t{to_str_for_file(t2)}\n")

    return vocab_file, merges_file


def test_train_bpe_tinystories(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    # return
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"

    # input_path = HOST_DATA_PATH / "owt_train.txt"

    print("=" * 60)
    print("🧪 开始 BPE 训练测试")
    print(f"📁 输入文件: {input_path}")
    print(f"📏 目标词汇表大小: 10000")
    print(f"🏷️  特殊标记: ['<|endoftext|>']")
    print("=" * 60)

    # 记录开始时间
    start_time = time.time()

    # 执行 BPE 训练
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # 记录结束时间
    training_time = time.time() - start_time
    print(f"⏱️  BPE 训练完成，耗时: {training_time:.2f}秒")

    # 1. 打印基本信息
    print(f"\n📊 训练结果统计:")
    print(f"   词汇表大小: {len(vocab)}")
    print(f"   合并规则数: {len(merges)}")

    # 2. 显示前10个高频 token
    print(f"\n🏆 前10个高频 token:")
    # 将 vocab 转换为按 ID 排序的列表
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
    for idx, token in sorted_vocab[:10]:
        if isinstance(token, bytes):
            try:
                token_str = token.decode("utf-8")
            except UnicodeDecodeError:
                token_str = f"bytes[{len(token)}]"
        else:
            token_str = str(token)
        print(f"   [{idx:4d}] {repr(token_str)}")

    # 3. 显示前10个合并规则
    print(f"\n🔗 前10个合并规则:")
    for i, (token1, token2) in enumerate(merges[:10], 1):
        # 转换 token 为可读字符串
        def token_to_str(t):
            if isinstance(t, bytes):
                try:
                    return t.decode("utf-8")
                except UnicodeDecodeError:
                    return t.hex()  # 十六进制表示
            return str(t)

        token1_str = token_to_str(token1)
        token2_str = token_to_str(token2)
        print(f"   {i:2d}. {repr(token1_str)} + {repr(token2_str)}")

    # 4. 检查特殊标记是否在词汇表中
    print(f"\n✅ 特殊标记检查:")
    special_token = "<|endoftext|>"
    special_token_id = None
    for idx, token in vocab.items():
        if token == special_token or (
            isinstance(token, bytes) and token.decode("utf-8", errors="ignore") == special_token
        ):
            special_token_id = idx
            break

    if special_token_id is not None:
        print(f"   特殊标记 '{special_token}' 在词汇表中，ID: {special_token_id}")
    else:
        print(f"   ⚠️  特殊标记 '{special_token}' 未在词汇表中找到！")

    # 5. 保存结果到文本文件
    print(f"\n💾 保存结果到文件...")
    output_dir = "test_results"
    vocab_file, merges_file = save_bpe_results(vocab, merges, output_dir, "tinystories")

    # 6. 创建汇总报告
    report_file = os.path.join(output_dir, "training_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BPE 训练结果报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练耗时: {training_time:.2f}秒\n\n")

        f.write(f"输入文件: {input_path}\n")
        f.write(f"目标词汇表大小: 10000\n")
        f.write(f"实际词汇表大小: {len(vocab)}\n")
        f.write(f"合并规则数: {len(merges)}\n\n")

        f.write("特殊标记检查:\n")
        if special_token_id is not None:
            f.write(f"  ✅ '{special_token}' -> ID: {special_token_id}\n")
        else:
            f.write(f"  ❌ '{special_token}' 未找到\n")

        f.write("\n词汇表示例 (前20个):\n")
        for idx, token in sorted_vocab[:20]:
            if isinstance(token, bytes):
                try:
                    token_str = token.decode("utf-8")
                except UnicodeDecodeError:
                    token_str = f"0x{token.hex()}"
            else:
                token_str = str(token)
            f.write(f"  [{idx:4d}] {repr(token_str)}\n")

        f.write("\n合并规则示例 (前20个):\n")
        for i, (token1, token2) in enumerate(merges[:20], 1):

            def safe_repr(t):
                if isinstance(t, bytes):
                    try:
                        return t.decode("utf-8")
                    except UnicodeDecodeError:
                        return f"0x{t.hex()}"
                return str(t)

            f.write(f"  {i:3d}. {repr(safe_repr(token1))} + {repr(safe_repr(token2))}\n")

        f.write(f"\n文件保存位置:\n")
        f.write(f"  词汇表 (JSON): {vocab_file}\n")
        f.write(f"  合并规则: {merges_file}\n")
        f.write(f"  完整报告: {report_file}\n")

    print(f"\n📄 训练报告已保存: {report_file}")
    print("=" * 60)

    # 7. 返回结果用于快照测试
    return vocab, merges
