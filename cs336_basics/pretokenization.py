import os
from typing import BinaryIO
import multiprocessing
import functools
import regex as re
from collections import defaultdict
import time
import mmap
from functools import lru_cache


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# 在 process_chunk 函数内部顶部添加
@lru_cache(maxsize=65536)
def encode_word(word: str) -> bytes:
    """缓存编码结果，对重复单词极快"""
    return word.encode("utf-8")


def process_chunk(
    start: int,
    end: int,
    input_path: str,
    special_tokens: list[str],
) -> dict[bytes, int]:
    """
    Process a single chunk of the file and return word counts.
    """
    t0 = time.time()
    print(f"[pretok] chunk {start}-{end} enter", flush=True)

    word_counts = defaultdict(int)
    special_tokens_set = set(special_tokens)

    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    t_read_start = time.time()
    with open(input_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk_bytes = mm[start:end]
            text = chunk_bytes.decode("utf-8", errors="ignore")
    t_read_end = time.time()

    # print(
    #     f"[pretok] chunk {start}-{end} after read, bytes={end - start}, "
    #     f"time={t_read_end - t_read_start:.2f}s",
    #     flush=True,
    # )

    t_split_start = time.time()
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(f"({pattern})", text)
    else:
        parts = [text]
    t_split_end = time.time()

    # print(
    #     f"[pretok] chunk {start}-{end} after split, parts={len(parts)}, "
    #     f"time={t_split_end - t_split_start:.2f}s",
    #     flush=True,
    # )

    t_regex_start = time.time()

    for part in parts:
        if part in special_tokens_set:
            continue

        # 用 finditer 减少内存开销
        for match in gpt2_pat.finditer(part):
            word = match.group()
            word_counts[encode_word(word)] += 1  # 使用缓存函数

    t_regex_end = time.time()

    total = t_regex_end - t0
    print(
        f"[pretok] chunk {start}-{end} done, "
        f"words={len(word_counts)}, "
        f"read={t_read_end - t_read_start:.2f}s "
        f"split={t_split_end - t_split_start:.2f}s "
        f"regex={t_regex_end - t_regex_start:.2f}s "
        f"total={total:.2f}s",
        flush=True,
    )
    return word_counts


def get_word_counts_parallel(input_path: str, special_tokens: list[str], num_processes: int = 4) -> dict[bytes, int]:
    """
    Parallelly count words in a file using multiple processes.
    """
    start_time = time.time()

    split_token = (
        b"<|endoftext|>"
        if "<|endoftext|>" in special_tokens
        else special_tokens[0].encode("utf-8")
        if special_tokens
        else b"\n"
    )

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
    print(f"[pretok] boundaries found: {len(boundaries) - 1} chunks", flush=True)

    chunk_args = list(zip(boundaries[:-1], boundaries[1:]))

    chunks_per_batch = 24

    # 分批处理
    total_word_counts = defaultdict(int)
    for i in range(0, len(chunk_args), chunks_per_batch):
        batch_args = chunk_args[i : i + chunks_per_batch]
        print(
            f"[pretok] 处理批次 {i // chunks_per_batch + 1}/{(len(chunk_args) + chunks_per_batch - 1) // chunks_per_batch}: {len(batch_args)}个chunks"
        )

        with multiprocessing.Pool(processes=min(num_processes, len(batch_args))) as pool:
            process_func = functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens)
            results = pool.starmap(process_func, batch_args)

        # 合并结果
        for res in results:
            for word, count in res.items():
                total_word_counts[word] += count

    total_elapsed = time.time() - start_time
    print(f"[pretok] total unique words={len(total_word_counts)}, total time={total_elapsed:.2f}s", flush=True)

    return total_word_counts


if __name__ == "__main__":
    start_time = time.time()

    input_file = "./tinystories_sample_5M.txt"
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        exit(1)

    special_tokens = ["<|endoftext|>"]
    print(f"🔧 Processing {input_file} with 4 processes...")

    # 并行处理
    parallel_start = time.time()
    total_word_counts = get_word_counts_parallel(input_file, special_tokens, num_processes=4)
    parallel_time = time.time() - parallel_start

    # 排序
    sort_start = time.time()
    top5_words = sorted(total_word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    sort_time = time.time() - sort_start

    # 输出结果
    print(f"📊 唯一单词数: {len(total_word_counts):,}")
    print("🏆 频率最高的5个单词:")
    for word, count in top5_words:
        print(f"  - {word}: {count:,}次")

    # 计时结果
    total_time = time.time() - start_time
    print("\n⏱️ 性能统计:")
    print(f"  并行处理: {parallel_time:.3f}秒")
    print(f"  排序处理: {sort_time * 1000:.1f}毫秒")
    print(f"  总时间:   {total_time:.3f}秒")
    print(f"  处理速度: {len(total_word_counts) / total_time:.0f} 单词/秒")
