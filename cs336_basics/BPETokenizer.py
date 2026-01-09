import json
from collections.abc import Iterable, Iterator
import importlib
from pathlib import Path
import sys
import regex as re


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        从词汇表、合并序列和（可选）特殊令牌列表构建分词器。

        参数:
            vocab: dict[int, bytes] - 词汇表，映射ID到字节序列
            merges: list[tuple[bytes, bytes]] - BPE合并操作列表
            special_tokens: list[str] | None = None - 特殊令牌列表
        """
        # 存储词汇表
        self.id_to_token = vocab  # ID -> bytes
        self.token_to_id = {token: idx for idx, token in vocab.items()}  # bytes -> ID

        # 存储合并操作
        self.merges = merges

        # 处理特殊令牌
        self.special_tokens = special_tokens if special_tokens else []
        self.special_token_to_id = {}
        self.special_id_to_token = {}

        print("merge", len(merges), "vocab", len(vocab.keys()))

        self._cpp_merger = None
        self._cpp_id_encoder = None
        self._cpp_pretoken_encoder = None
        try:
            project_root = Path(__file__).resolve().parents[1]
            cpp_build_dir = project_root / "cpp" / "build"
            if cpp_build_dir.exists():
                build_dir_str = str(cpp_build_dir)
                if build_dir_str not in sys.path:
                    sys.path.append(build_dir_str)
            bpe = importlib.import_module("bpe")
            pretoken_encoder_cls = getattr(bpe, "BPEPreTokenEncoder", None)
            if pretoken_encoder_cls is not None:
                self._cpp_pretoken_encoder = pretoken_encoder_cls(self.id_to_token, self.merges, self.special_tokens)
            id_encoder_cls = getattr(bpe, "BPEIdEncoder", None)
            if id_encoder_cls is not None:
                self._cpp_id_encoder = id_encoder_cls(self.id_to_token, self.merges)
            merger_cls = getattr(bpe, "BPEMerger", None)
            if merger_cls is not None:
                self._cpp_merger = merger_cls(self.merges)
        except Exception:
            self._cpp_merger = None
            self._cpp_id_encoder = None
            self._cpp_pretoken_encoder = None

        if self._cpp_merger is None and self._cpp_id_encoder is None and self._cpp_pretoken_encoder is None:
            # 使用py版本归并，构建合并操作索引，用于快速查找
            self.merge_rank = {}
            for rank, (a, b) in enumerate(merges):
                self.merge_rank[(a, b)] = rank

        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 构建特殊令牌的正则表达式模式，用于识别特殊令牌
        if self.special_tokens:
            # 按长度降序排序，确保更长的特殊令牌优先匹配
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # 创建正则模式，匹配任意特殊标记
            self.special_token_pattern = re.compile("(" + "|".join(map(re.escape, sorted_special_tokens)) + ")")
        else:
            self.special_token_pattern = None

    def _pre_tokenize(self, text: str) -> list[str]:
        """
        预分词函数：将文本分割成单词/子词
        """
        if not text:
            return []

        res = []
        pos = 0  # 记录当前位置

        # 如果有特殊令牌模式，先匹配特殊令牌
        if self.special_token_pattern:
            for match in self.special_token_pattern.finditer(text):
                # 添加特殊令牌之前的普通文本
                start, end = match.span()
                if start > pos:
                    # 处理普通文本
                    for sub_match in self.gpt2_pat.finditer(text[pos:start]):
                        res.append(sub_match.group())

                # 添加特殊令牌
                res.append(match.group())

                # 跳过特殊令牌（不添加到结果）
                pos = end

            # 处理最后一段普通文本
            if pos < len(text):
                for match in self.gpt2_pat.finditer(text[pos:]):
                    res.append(match.group())
        else:
            # 没有特殊令牌，直接处理整个文本
            for match in self.gpt2_pat.finditer(text):
                res.append(match.group())

        return res

    def _apply_merges_to_token(self, token_bytes: bytes) -> list[bytes]:
        """
        对一个token的字节序列应用BPE合并操作。

        参数:
            token_bytes: bytes - token的字节表示

        返回:
            list[bytes] - 合并后的字节token列表
        """
        # 如果没有合并操作，直接返回单个token
        if not self.merges:
            return [token_bytes]

        if self._cpp_merger is not None:
            return list(self._cpp_merger.apply(token_bytes))

        # 将字节序列拆分为单个字节的列表
        tokens = [bytes([b]) for b in token_bytes]

        # 持续应用合并操作，直到无法合并
        changed = True
        while changed and len(tokens) > 1:
            changed = False

            # 查找最佳合并（具有最小rank的合并）
            best_pair = None
            best_rank = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_rank:
                    rank = self.merge_rank[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = (i, pair)

            # 如果找到最佳合并，应用它
            if best_pair is not None:
                i, (a, b) = best_pair
                # 合并两个token，a、b是字节连接操作
                merged = a + b
                # 替换原来的两个token为合并后的token
                tokens = tokens[:i] + [merged] + tokens[i + 2 :]
                changed = True

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为令牌ID序列。

        参数:
            text: str - 输入文本

        返回:
            list[int] - 令牌ID序列
        """
        # 预分词
        pre_tokens = self._pre_tokenize(text)
        # print("pre_tokens:", pre_tokens)

        if self._cpp_pretoken_encoder is not None:
            # pybind11 把 list[str] 转成 std::vector<std::string> 时，
            # 会用 CPython 的 UTF-8 表示把每个 str 转成 std::string，不需要手动 encode("utf-8")
            return list(self._cpp_pretoken_encoder.encode_pretokens(pre_tokens))

        token_ids = []

        for token_str in pre_tokens:
            # token：转换为字节并应用BPE合并
            token_bytes = token_str.encode("utf-8")

            # 检查是否是特殊令牌
            if token_str in self.special_tokens:
                # 特殊令牌，直接获取ID
                token_ids.append(self.token_to_id[token_bytes])
                continue

            if self._cpp_id_encoder is not None:
                token_ids.extend(self._cpp_id_encoder.encode_token(token_bytes))
                continue

            # 应用BPE合并
            merged_tokens = self._apply_merges_to_token(token_bytes)

            # 将字节token转换为ID
            for token in merged_tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    # 处理未知token：可以使用特殊UNK令牌或拆分为字节
                    # 这里拆分为字节并查找每个字节的ID
                    for i in range(len(token)):
                        byte_token = token[i : i + 1]
                        if byte_token in self.token_to_id:
                            token_ids.append(self.token_to_id[byte_token])
                        else:
                            # 如果字节也不在词汇表中，跳过或使用默认ID
                            # 这里我们跳过未知字节
                            pass

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定字符串可迭代对象（如Python文件句柄），返回惰性生成令牌ID的生成器。
        用于高效编码无法直接加载到内存的大型文件。

        参数:
            iterable: Iterable[str] - 字符串可迭代对象

        返回:
            Iterator[int] - 生成令牌ID的迭代器
        """
        # 用于处理跨块边界的情况
        buffer = ""

        for chunk in iterable:
            # 将新块添加到缓冲区
            buffer += chunk

            # 尝试从缓冲区末尾找到一个完整的分词边界
            # 简单实现：从后往前查找空格或换行符作为边界
            boundary = self._find_last_boundary(buffer)

            # 处理完整的部分
            if boundary > 0:
                complete_text = buffer[:boundary]
                for token_id in self.encode(complete_text):
                    yield token_id

                # 保留未处理的部分
                buffer = buffer[boundary:]

        # 处理剩余的缓冲区内容
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def _find_last_boundary(self, text: str, max_lookback: int = 100) -> int:
        """
        在文本中查找最后一个合适的分词边界。

        参数:
            text: str - 输入文本
            max_lookback: int - 最大回溯长度

        返回:
            int - 边界位置
        """
        # 限制回溯长度以避免性能问题
        lookback_text = text[-max_lookback:] if len(text) > max_lookback else text

        # 查找最后一个空白字符或标点作为边界
        for i in range(len(lookback_text) - 1, -1, -1):
            char = lookback_text[i]
            if char.isspace() or char in ".,!?;:":
                # 计算在原始文本中的位置
                boundary_pos = len(text) - len(lookback_text) + i + 1
                return boundary_pos

        # 如果没有找到边界，回退到处理整个文本
        # 这可能会导致跨块边界的token，但在实际使用中应该很少发生
        return len(text)

    def decode(self, ids: list[int]) -> str:
        """
        将令牌ID序列解码为文本。

        参数:
            ids: list[int] - 令牌ID序列

        返回:
            str - 解码后的文本
        """
        # 将ID转换为字节
        byte_parts = []

        for token_id in ids:
            if token_id in self.id_to_token:
                byte_parts.append(self.id_to_token[token_id])
            # elif token_id in self.special_id_to_token:
            #     byte_parts.append(self.special_id_to_token[token_id])
            else:
                # 未知ID，跳过或使用占位符
                # 这里我们跳过
                pass

        # 合并所有字节并解码为字符串
        if byte_parts:
            # 合并所有字节部分
            all_bytes = b"".join(byte_parts)
            try:
                return all_bytes.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                # 如果UTF-8解码失败，尝试使用错误替换
                return all_bytes.decode("utf-8", errors="ignore")
        else:
            return ""

    def get_vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.id_to_token)

    def save(self, vocab_filepath: str, merges_filepath: str):
        """
        保存词汇表和合并序列到文件。

        参数:
            vocab_filepath: str - 词汇表文件路径
            merges_filepath: str - 合并序列文件路径
        """
        # 保存词汇表
        vocab_dict = {}
        for idx, token_bytes in self.id_to_token.items():
            # 将字节转换为可序列化的格式
            # 使用十六进制表示
            hex_str = token_bytes.hex()
            vocab_dict[str(idx)] = hex_str

        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2)

        # 保存合并序列
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                # 将字节转换为字符串表示
                a_str = a.hex()
                b_str = b.hex()
                f.write(f"{a_str} {b_str}\n")


# 示例使用
if __name__ == "__main__":
    # 示例词汇表和合并序列（使用问题中的例子）
    vocab_example = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
    }

    merges_example = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]

    # 创建tokenizer
    tokenizer = BPETokenizer(vocab_example, merges_example, special_tokens=["<|endoftext|>"])

    # 测试编码
    text = "the cat ate<|endoftext|>the cat ate"
    encoded = tokenizer.encode(text)
    print(f"编码 '{text}': {encoded}")

    # 测试解码
    decoded = tokenizer.decode(encoded)
    print(f"解码 {encoded}: '{decoded}'")

    # 测试迭代编码
    def text_generator():
        yield "the cat "
        yield "ate the "
        yield "cat"

    print("\n迭代编码:")
    for token_id in tokenizer.encode_iterable(text_generator()):
        print(f"  Token ID: {token_id}")

    # 测试特殊令牌
    tokenizer_with_special = BPETokenizer(vocab_example, merges_example, special_tokens=["[CLS]", "[SEP]"])
    text_with_special = "[CLS] the cat ate [SEP]"
    encoded_special = tokenizer_with_special.encode(text_with_special)
    print(f"\n编码带特殊令牌的文本 '{text_with_special}': {encoded_special}")
    print(f"解码: '{tokenizer_with_special.decode(encoded_special)}'")

    # 测试保存和加载
    tokenizer.save("vocab.json", "merges.txt")
    print("\n保存到文件完成")
