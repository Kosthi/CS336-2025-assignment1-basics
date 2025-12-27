import os
from typing import BinaryIO
import multiprocessing
import functools
import regex as re
from collections import defaultdict
import time
import mmap


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


def process_chunk(
    start: int,
    end: int,
    input_path: str,
    special_tokens: list[str],
) -> dict[bytes, int]:
    """
    Process a single chunk of the file and return word counts.
    """
    word_counts = defaultdict(int)
    special_tokens_set = set(special_tokens)

    # Compile regex pattern locally for each process
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    with open(input_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk_bytes = mm[start:end]
            text = chunk_bytes.decode("utf-8", errors="ignore")

    # Pre-tokenization
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(f"({pattern})", text)
    else:
        parts = [text]

    for part in parts:
        if part in special_tokens_set:
            continue
        words = gpt2_pat.findall(part)
        for word in words:
            word_counts[word.encode("utf-8")] += 1

    return word_counts


def get_word_counts_parallel(input_path: str, special_tokens: list[str], num_processes: int = 4) -> dict[bytes, int]:
    """
    Parallelly count words in a file using multiple processes.
    """
    # Determine split token
    split_token = b"<|endoftext|>"
    if special_tokens and "<|endoftext|>" in special_tokens:
        split_token = b"<|endoftext|>"
    elif special_tokens:
        split_token = special_tokens[0].encode("utf-8")
    else:
        split_token = b"\n"

    # Find boundaries
    # 二进制模式读取
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    # Process chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens)

        chunk_args = list(zip(boundaries[:-1], boundaries[1:]))
        results = pool.starmap(process_func, chunk_args)

    # Aggregate results
    total_word_counts = defaultdict(int)
    for res in results:
        for word, count in res.items():
            total_word_counts[word] += count

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
