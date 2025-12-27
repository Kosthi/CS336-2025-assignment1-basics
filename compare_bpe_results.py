import json
from pathlib import Path


def load_vocab(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_merges(path: Path) -> list[str]:
    # merges 文件里的每一行是 "token1 token2"
    # 这里直接按行字符串对比，避免因为空格等细节自己重新解析出错
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def compare_dicts(name: str, d1: dict, d2: dict) -> None:
    print(f"\n==== Comparing {name} ====")
    if d1 == d2:
        print(f"{name} OK: 完全一致")
        return

    print(f"{name} 不一致：")
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    only1 = keys1 - keys2
    only2 = keys2 - keys1
    both = keys1 & keys2

    if only1:
        print(f"  仅在 result 中出现的 key 数量: {len(only1)}，示例: {list(sorted(only1))[:5]}")
    if only2:
        print(f"  仅在 result_opt 中出现的 key 数量: {len(only2)}，示例: {list(sorted(only2))[:5]}")

    diff_values = [k for k in both if d1[k] != d2[k]]
    print(f"  key 相同但 value 不同的数量: {len(diff_values)}")
    for k in diff_values[:5]:
        print(f"    key={k!r}: result={d1[k]!r}, result_opt={d2[k]!r}")


def compare_lists(name: str, l1: list[str], l2: list[str]) -> None:
    print(f"\n==== Comparing {name} ====")
    if l1 == l2:
        print(f"{name} OK: 顺序和内容完全一致")
        return

    print(f"{name} 不一致：")
    print(f"  result 长度:     {len(l1)}")
    print(f"  result_opt 长度: {len(l2)}")

    min_len = min(len(l1), len(l2))
    first_diff = None
    for i in range(min_len):
        if l1[i] != l2[i]:
            first_diff = i
            break

    if first_diff is not None:
        print(f"  第一个不一致的位置: {first_diff}")
        print(f"    result    : {l1[first_diff]!r}")
        print(f"    result_opt: {l2[first_diff]!r}")
    else:
        print("  前 min_len 个元素都相同，只是长度不同")


def main() -> None:
    base = Path(".")
    dir_a = base / "test_results"
    dir_b = base / "test_results_opt"

    vocab_a = load_vocab(dir_a / "tinystories_vocab.json")
    vocab_b = load_vocab(dir_b / "tinystories_vocab.json")

    merges_a = load_merges(dir_a / "tinystories_merges.txt")
    merges_b = load_merges(dir_b / "tinystories_merges.txt")

    compare_dicts("vocab", vocab_a, vocab_b)
    compare_lists("merges", merges_a, merges_b)


if __name__ == "__main__":
    main()
