import random
import os
import mmap


class LargeFileSampler:
    def __init__(self, seed: int = 42):
        """初始化采样器"""
        random.seed(seed)

    def count_documents_quick(self, filepath: str, delimiter: str = "<|endoftext|>") -> int:
        """
        快速统计文档数量（使用内存映射）
        """
        print(f"快速统计 {filepath} 中的文档数量...")

        doc_count = 0
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                # 使用内存映射处理大文件
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

                # 使用简单计数
                doc_count = mm.read().count(delimiter.encode("utf-8"))

                mm.close()
        except Exception as e:
            print(f"统计文档数量时出错: {e}")
            # 回退到逐行读取
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read(100 * 1024 * 1024)  # 只读取前100MB来估计
                doc_count = content.count(delimiter)

        print(f"估计文档数量: {doc_count}")
        return doc_count

    def sample_documents_mmap(
        self, filepath: str, num_samples: int = 10, delimiter: str = "<|endoftext|>"
    ) -> list[str]:
        """
        使用内存映射从大文件中采样文档
        """
        print(f"使用内存映射采样: {filepath}")

        try:
            with open(filepath, "r+b") as f:
                # 创建内存映射
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

                # 找到所有分隔符的位置
                delimiter_bytes = delimiter.encode("utf-8")
                delimiter_len = len(delimiter_bytes)

                # 查找所有分隔符位置
                positions = []
                pos = mm.find(delimiter_bytes, 0)
                while pos != -1:
                    positions.append(pos)
                    pos = mm.find(delimiter_bytes, pos + delimiter_len)

                print(f"找到 {len(positions)} 个分隔符")

                if len(positions) < 2:
                    print("分隔符数量不足，无法提取文档")
                    return []

                # 采样文档
                sampled_docs = []
                # 确保我们采样不同的文档段
                if len(positions) - 1 <= num_samples:
                    # 文档数量不足，返回所有文档
                    for i in range(len(positions) - 1):
                        start = positions[i] + delimiter_len
                        end = positions[i + 1]
                        doc_bytes = mm[start:end]
                        try:
                            doc_text = doc_bytes.decode("utf-8").strip()
                            if doc_text:
                                sampled_docs.append(doc_text)
                        except UnicodeDecodeError:
                            # 跳过解码失败的文档
                            pass
                else:
                    # 随机采样文档
                    sampled_indices = random.sample(range(len(positions) - 1), num_samples)
                    for idx in sampled_indices:
                        start = positions[idx] + delimiter_len
                        end = positions[idx + 1]
                        doc_bytes = mm[start:end]
                        try:
                            doc_text = doc_bytes.decode("utf-8").strip()
                            if doc_text:
                                sampled_docs.append(doc_text)
                        except UnicodeDecodeError:
                            # 跳过解码失败的文档
                            pass

                mm.close()
                print(f"成功采样 {len(sampled_docs)} 个文档")
                return sampled_docs

        except Exception as e:
            print(f"内存映射采样失败: {e}")
            return []

    def sample_documents_streaming(
        self, filepath: str, num_samples: int = 10, delimiter: str = "<|endoftext|>", chunk_size: int = 1024 * 1024
    ) -> list[str]:
        """
        流式采样：分批读取文件，避免内存问题
        """
        print(f"使用流式采样: {filepath}")

        try:
            # 第一步：统计文档数量
            print("正在统计文档数量...")
            total_docs = 0
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                buffer = ""
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    buffer += chunk
                    total_docs += buffer.count(delimiter)
                    buffer = buffer[-len(delimiter) * 2 :]  # 保留末尾部分以防止分隔符被切断

            print(f"总文档数量: {total_docs}")

            if total_docs == 0:
                print("未找到文档分隔符")
                return []

            # 第二步：使用蓄水池采样算法
            reservoir = []
            current_doc_num = 0

            with open(filepath, encoding="utf-8", errors="ignore") as f:
                current_doc = []

                for line in f:
                    # 检查行中是否包含分隔符
                    if delimiter in line:
                        parts = line.split(delimiter)

                        for i, part in enumerate(parts):
                            if i > 0 and current_doc:
                                # 文档结束
                                doc_text = "".join(current_doc).strip()
                                if doc_text:
                                    current_doc_num += 1

                                    # 蓄水池采样
                                    if len(reservoir) < num_samples:
                                        reservoir.append(doc_text)
                                    else:
                                        r = random.randint(1, current_doc_num)
                                        if r <= num_samples:
                                            reservoir[random.randint(0, num_samples - 1)] = doc_text

                                current_doc = []

                            if part.strip():
                                current_doc.append(part)
                    else:
                        if current_doc:
                            current_doc.append(line)

                # 处理最后一个文档
                if current_doc:
                    doc_text = "".join(current_doc).strip()
                    if doc_text:
                        current_doc_num += 1
                        if len(reservoir) < num_samples:
                            reservoir.append(doc_text)
                        else:
                            r = random.randint(1, current_doc_num)
                            if r <= num_samples:
                                reservoir[random.randint(0, num_samples - 1)] = doc_text

            print(f"流式采样完成: {len(reservoir)} 个文档")
            return reservoir

        except Exception as e:
            print(f"流式采样失败: {e}")
            return []

    def sample_documents_simple(
        self, filepath: str, num_samples: int = 10, delimiter: str = "<|endoftext|>"
    ) -> list[str]:
        """
        简单但有效的采样方法：随机读取文件位置
        """
        print(f"使用简单随机位置采样: {filepath}")

        try:
            file_size = os.path.getsize(filepath)
            print(f"文件大小: {file_size / (1024**3):.2f} GB")

            # 随机选择文件位置
            sample_positions = random.sample(range(file_size), min(100, file_size))
            sample_positions.sort()

            documents = []
            with open(filepath, "rb") as f:
                for pos in sample_positions:
                    # 定位到随机位置
                    f.seek(pos)

                    # 向前查找分隔符
                    f.seek(max(0, pos - 10000))  # 向前搜索最多10KB
                    chunk = f.read(20000)  # 读取20KB的块

                    try:
                        chunk_text = chunk.decode("utf-8", errors="ignore")

                        # 查找最近的分隔符
                        last_delimiter = chunk_text.rfind(delimiter)
                        next_delimiter = chunk_text.find(
                            delimiter, last_delimiter + len(delimiter) if last_delimiter != -1 else 0
                        )

                        if last_delimiter != -1 and next_delimiter != -1:
                            doc_text = chunk_text[last_delimiter + len(delimiter) : next_delimiter].strip()
                            if doc_text and len(doc_text) > 100:  # 确保文档有足够长度
                                documents.append(doc_text)
                                if len(documents) >= num_samples:
                                    break
                    except Exception:
                        continue

            print(f"简单采样完成: {len(documents)} 个文档")
            return documents

        except Exception as e:
            print(f"简单采样失败: {e}")
            return []

    def sample_with_validation(
        self, filepath: str, num_samples: int = 10, delimiter: str = "<|endoftext|>"
    ) -> list[str]:
        """
        带验证的采样：尝试多种方法
        """
        print(f"开始采样 {filepath}...")

        # 方法1: 尝试内存映射
        samples = self.sample_documents_mmap(filepath, num_samples, delimiter)
        if len(samples) >= num_samples:
            print("内存映射采样成功")
            return samples[:num_samples]

        # 方法2: 尝试流式采样
        if len(samples) < num_samples:
            print("内存映射采样不足，尝试流式采样...")
            stream_samples = self.sample_documents_streaming(filepath, num_samples - len(samples), delimiter)
            samples.extend(stream_samples)

        # 方法3: 如果还不够，尝试简单采样
        if len(samples) < num_samples:
            print("采样不足，尝试简单随机采样...")
            simple_samples = self.sample_documents_simple(filepath, num_samples - len(samples), delimiter)
            samples.extend(simple_samples)

        # 如果还是没有采样到，创建一些示例数据
        if len(samples) == 0:
            print("未能从文件中采样到文档，创建示例数据...")
            samples = self.create_sample_documents(num_samples)

        print(f"最终采样到 {len(samples)} 个文档")
        return samples[:num_samples]

    def create_sample_documents(self, num_samples: int) -> list[str]:
        """创建示例文档"""
        sample_docs = [
            "This is a sample document from a large OpenWebText file.",
            "The file contains web-crawled text data from various sources.",
            "OpenWebText is commonly used for training language models.",
            "Each document is separated by the <|endoftext|> delimiter.",
            "This dataset contains millions of documents from the internet.",
            "The text covers a wide range of topics and writing styles.",
            "Documents vary in length from short paragraphs to long articles.",
            "The dataset is used for research in natural language processing.",
            "Training on this data helps models understand diverse language patterns.",
            "This sample was generated because the actual file could not be read.",
        ]
        return sample_docs[:num_samples]

    def analyze_file_format(self, filepath: str, delimiter: str = "<|endoftext|>"):
        """分析文件格式"""
        print(f"\n分析文件格式: {filepath}")

        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                # 读取文件头部
                head = f.read(5000)

                print("文件头部预览 (前5000字符):")
                print("-" * 50)
                print(head[:500])
                print("..." if len(head) > 500 else "")
                print("-" * 50)

                # 统计分隔符出现次数
                delimiter_count = head.count(delimiter)
                print(f"前5000字符中分隔符出现次数: {delimiter_count}")

                # 检查是否有换行符
                newline_count = head.count("\n")
                print(f"前5000字符中换行符出现次数: {newline_count}")

                # 检查分隔符格式
                lines_with_delimiter = [line for line in head.split("\n") if delimiter in line]
                print(f"包含分隔符的行数: {len(lines_with_delimiter)}")

                if lines_with_delimiter:
                    print("分隔符格式示例:")
                    for i, line in enumerate(lines_with_delimiter[:3]):
                        print(f"  行 {i + 1}: {line.strip()}")

                # 读取文件尾部
                file_size = os.path.getsize(filepath)
                with open(filepath, "rb") as f_bin:
                    f_bin.seek(max(0, file_size - 5000))
                    tail_bytes = f_bin.read(5000)
                    try:
                        tail = tail_bytes.decode("utf-8", errors="ignore")
                        print("\n文件尾部预览:")
                        print("-" * 50)
                        print(tail[-500:])
                        print("-" * 50)
                    except Exception:
                        print("无法解码文件尾部")

        except Exception as e:
            print(f"分析文件格式时出错: {e}")


def sample_openwebtext_large(filepath: str, num_samples: int = 10) -> list[str]:
    """
    专为大型OpenWebText文件设计的采样函数
    """
    print(f"开始采样大型OpenWebText文件: {filepath}")

    sampler = LargeFileSampler(seed=42)

    # 首先分析文件格式
    sampler.analyze_file_format(filepath)

    # 尝试采样
    samples = sampler.sample_with_validation(filepath, num_samples)

    return samples


def sample_tinystories_simple(filepath: str, num_samples: int = 10) -> list[str]:
    """
    采样TinyStories文件（通常较小）
    """
    print(f"采样TinyStories文件: {filepath}")

    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # 分割文档
        documents = [doc.strip() for doc in content.split("<|endoftext|>") if doc.strip()]

        print(f"找到 {len(documents)} 个文档")

        if len(documents) <= num_samples:
            print(f"文档数量不足，返回全部 {len(documents)} 个文档")
            return documents

        # 随机采样
        random.seed(42)
        sampled = random.sample(documents, num_samples)
        print(f"成功采样 {len(sampled)} 个文档")

        return sampled

    except Exception as e:
        print(f"采样TinyStories失败: {e}")
        # 返回示例数据
        return [
            "Once upon a time, there was a little cat named Whiskers.",
            "The sun was shining brightly on a beautiful summer day.",
            "Tommy found a shiny red apple under the tree.",
            "Lucy learned how to ride her bike without training wheels.",
            "The little mouse was afraid of the big brown cat.",
            "It started to rain, so the children ran inside.",
            "Grandma baked delicious chocolate chip cookies.",
            "The butterfly had beautiful blue and yellow wings.",
            "The dog wanted to play with the ball in the garden.",
            "A boy and a girl went to the park to fly their kite.",
        ][:num_samples]


def main():
    """
    主函数：采样两个数据集
    """
    # 设置文件路径
    TINYSTORIES_PATH = "./data/TinyStoriesV2-GPT4-train.txt"  # 请修改为实际路径
    OPENWEBTEXT_PATH = "./data/owt_train.txt"  # 11GB的大文件

    print("=" * 70)
    print("文档采样程序")
    print("=" * 70)

    # 检查文件是否存在
    if not os.path.exists(TINYSTORIES_PATH):
        print(f"警告: TinyStories文件不存在: {TINYSTORIES_PATH}")
        TINYSTORIES_PATH = None

    if not os.path.exists(OPENWEBTEXT_PATH):
        print(f"警告: OpenWebText文件不存在: {OPENWEBTEXT_PATH}")
        OPENWEBTEXT_PATH = None

    # 采样TinyStories
    print("\n" + "=" * 30 + " TinyStories " + "=" * 30)
    if TINYSTORIES_PATH:
        tinystories_samples = sample_tinystories_simple(TINYSTORIES_PATH, 10)
    else:
        print("使用示例TinyStories数据")
        tinystories_samples = [
            "Once upon a time, there was a little cat named Whiskers.",
            "The sun was shining brightly on a beautiful summer day.",
            "Tommy found a shiny red apple under the tree.",
            "Lucy learned how to ride her bike without training wheels.",
            "The little mouse was afraid of the big brown cat.",
            "It started to rain, so the children ran inside.",
            "Grandma baked delicious chocolate chip cookies.",
            "The butterfly had beautiful blue and yellow wings.",
            "The dog wanted to play with the ball in the garden.",
            "A boy and a girl went to the park to fly their kite.",
        ]

    # 采样OpenWebText
    print("\n" + "=" * 30 + " OpenWebText " + "=" * 30)
    if OPENWEBTEXT_PATH:
        openwebtext_samples = sample_openwebtext_large(OPENWEBTEXT_PATH, 10)
    else:
        print("使用示例OpenWebText数据")
        openwebtext_samples = [
            "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "The internet has revolutionized the way we communicate and access information.",
            "Climate change is one of the most pressing issues facing humanity today.",
            "The development of renewable energy sources is crucial for sustainable development.",
            "Advances in medicine have significantly increased life expectancy worldwide.",
            "Blockchain technology has applications beyond cryptocurrency in various industries.",
            "The rise of social media has transformed how people interact and share information.",
            "Remote work has become increasingly common due to improvements in technology.",
            "Artificial intelligence is being integrated into various aspects of daily life.",
            "Space exploration continues to expand our understanding of the universe.",
        ]

    # 显示结果
    print("\n" + "=" * 30 + " 采样结果 " + "=" * 30)
    print(f"TinyStories 采样数: {len(tinystories_samples)}")
    print(f"OpenWebText 采样数: {len(openwebtext_samples)}")

    # 显示前2个样本
    print("\nTinyStories 前2个样本:")
    for i, doc in enumerate(tinystories_samples[:2]):
        preview = doc[:150] + "..." if len(doc) > 150 else doc
        print(f"  文档 {i + 1}: {preview}")

    print("\nOpenWebText 前2个样本:")
    for i, doc in enumerate(openwebtext_samples[:2]):
        preview = doc[:150] + "..." if len(doc) > 150 else doc
        print(f"  文档 {i + 1}: {preview}")

    # 保存结果
    print("\n" + "=" * 30 + " 保存结果 " + "=" * 30)

    # 保存带分隔符的版本
    with open("sampled_tinystories.txt", "w", encoding="utf-8") as f:
        for doc in tinystories_samples:
            f.write(f"{doc}<|endoftext|>")
    with open("sampled_owt.txt", "w", encoding="utf-8") as f:
        for doc in openwebtext_samples:
            f.write(f"{doc}<|endoftext|>")

    print("结果已保存到 sampled_tinystories.txt, sampled_owt.txt")

    # 保存纯文本版本（不带分隔符）
    # with open("sampled_documents_clean.txt", "w", encoding="utf-8") as f:
    #     f.write("TINYSTORIES SAMPLES\n" + "="*50 + "\n\n")
    #     for i, doc in enumerate(tinystories_samples, 1):
    #         f.write(f"文档 {i}:\n{doc}\n\n")

    #     f.write("\n" + "="*50 + "\n\n")
    #     f.write("OPENWEBTEXT SAMPLES\n" + "="*50 + "\n\n")
    #     for i, doc in enumerate(openwebtext_samples, 1):
    #         f.write(f"文档 {i}:\n{doc}\n\n")

    # print(f"纯文本版本已保存到 sampled_documents_clean.txt")


def check_file_format():
    """检查文件格式的辅助函数"""
    filepath = "./data/owt_train.txt"

    print(f"检查文件格式: {filepath}")

    # 检查文件大小
    file_size = os.path.getsize(filepath)
    print(f"文件大小: {file_size / (1024**3):.2f} GB")

    # 读取文件开头
    with open(filepath, "rb") as f:
        # 读取前1000字节
        head_bytes = f.read(1000)
        try:
            head_text = head_bytes.decode("utf-8", errors="ignore")
            print("\n文件开头 (前1000字节):")
            print("-" * 50)
            print(head_text)
            print("-" * 50)

            # 检查分隔符
            delimiter = "<|endoftext|>"
            if delimiter in head_text:
                print(f"找到分隔符: {delimiter}")
                print(f"在文件开头出现次数: {head_text.count(delimiter)}")
            else:
                print(f"未找到分隔符: {delimiter}")
                # 尝试查找其他可能的分隔符
                common_delimiters = ["\n\n\n", "\n\n", "---", "===="]
                for delim in common_delimiters:
                    if delim in head_text:
                        print(f"找到可能的分隔符: {repr(delim)}")
        except Exception:
            print("无法解码文件开头")

    # 读取文件结尾
    with open(filepath, "rb") as f:
        f.seek(max(0, file_size - 1000))
        tail_bytes = f.read(1000)
        try:
            tail_text = tail_bytes.decode("utf-8", errors="ignore")
            print("\n文件结尾 (最后1000字节):")
            print("-" * 50)
            print(tail_text)
            print("-" * 50)
        except Exception:
            print("无法解码文件结尾")


if __name__ == "__main__":
    # 首先检查文件格式
    print("检查OpenWebText文件格式...")
    check_file_format()

    # 运行主采样程序
    main()
