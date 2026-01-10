import argparse
import json
import os
import queue
import sys
import time
import importlib
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import torch

from .BPETokenizer import BPETokenizer
from .cross_entropy import cross_entropy
from .optimizer.adamw import AdamW
from .optimizer.lr_cosine_schedule import lr_cosine_schedule
from .optimizer.sgd import SGD
from .pretokenization import get_word_counts_parallel
from .transformer_lm import TransformerLM
from .utils.checkpointing import load_checkpoint, save_checkpoint
from .utils.gradient_clipping import gradient_clipping


_ENCODE_WORKER_TOKENIZER: BPETokenizer | None = None


def _encode_worker_init(tokenizer: BPETokenizer) -> None:
    global _ENCODE_WORKER_TOKENIZER
    _ENCODE_WORKER_TOKENIZER = tokenizer


def _encode_worker(text: str) -> list[int]:
    if _ENCODE_WORKER_TOKENIZER is None:
        raise RuntimeError("worker tokenizer 未初始化")
    return _ENCODE_WORKER_TOKENIZER.encode(text)


def _encode_shm_worker_loop(
    task_q: Any,
    done_q: Any,
    *,
    tokenizer: BPETokenizer,
    shm_name: str,
    dtype_str: str,
    slot_tokens: int,
    num_slots: int,
    max_value: int | None,
) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        dtype = np.dtype(dtype_str)
        slots = np.ndarray((num_slots, slot_tokens), dtype=dtype, buffer=shm.buf)
        while True:
            item = task_q.get()
            if item is None:
                return
            seq, text, slot_id = item
            try:
                token_ids = tokenizer.encode(text)
                if max_value is not None:
                    bad_tok = None
                    for tok in token_ids:
                        if tok > max_value:
                            bad_tok = tok
                            break
                    if bad_tok is not None:
                        done_q.put(
                            (seq, slot_id, 0, f"token id {bad_tok} 超出 dtype={dtype} 可表示范围（max={max_value}）")
                        )
                        continue

                length = len(token_ids)
                if length > slot_tokens:
                    done_q.put((seq, slot_id, 0, f"片段 token 数 {length} 超过 slot_tokens={slot_tokens}"))
                    continue
                slots[slot_id, :length] = np.asarray(token_ids, dtype=dtype)
                done_q.put((seq, slot_id, length, None))
            except Exception as e:
                done_q.put((seq, slot_id, 0, str(e)))
    finally:
        shm.close()


def _auto_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_numpy_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(name)
    except TypeError as e:
        raise ValueError(f"无效 dtype: {name}") from e


def _open_tokens(path: str | os.PathLike, dtype: np.dtype) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".bin":
        return np.memmap(p, mode="r", dtype=dtype)
    if p.suffix == ".npy":
        arr = np.load(p, mmap_mode="r")
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr
    raise ValueError(f"不支持的数据文件后缀: {p.suffix}（仅支持 .bin / .npy）")


def _get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    starts = rng.integers(0, max_start + 1, size=(batch_size,), dtype=np.int64)
    offsets = np.arange(context_length, dtype=np.int64)[None, :]
    idx = starts[:, None] + offsets
    x_np = dataset[idx]
    y_np = dataset[idx + 1]
    x = torch.from_numpy(np.asarray(x_np)).long().to(device)
    y = torch.from_numpy(np.asarray(y_np)).long().to(device)
    return x, y


def _loss_per_token(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    loss = cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return loss


@torch.no_grad()
def _eval_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
    rng: np.random.Generator,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        x, y = _get_batch(dataset, batch_size, context_length, device, rng)
        logits = model(x)
        loss = _loss_per_token(logits, y)
        losses.append(loss.detach().float().cpu().item())
    return float(np.mean(losses)) if losses else float("nan")


def _load_bpe_tokenizer(vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str]):
    with open(vocab_path, encoding="utf-8") as f:
        vocab_json: dict[str, str] = json.load(f)
    vocab: dict[int, bytes] = {int(i): bytes.fromhex(hex_str) for i, hex_str in vocab_json.items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a_hex, b_hex = line.split()
            merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def _import_cpp_bpe() -> Any:
    project_root = Path(__file__).resolve().parents[1]
    cpp_build_dir = project_root / "cpp" / "build"
    if cpp_build_dir.exists():
        build_dir_str = str(cpp_build_dir)
        if build_dir_str not in sys.path:
            sys.path.append(build_dir_str)
    try:
        return importlib.import_module("bpe")
    except ImportError as e:
        raise RuntimeError(
            "bpe 模块不可用；请先构建 cpp 扩展（cmake -S cpp -B cpp/build && cmake --build cpp/build）"
        ) from e


def _train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> BPETokenizer:
    bpe = _import_cpp_bpe()
    train_fn = getattr(bpe, "train", None)
    if train_fn is None:
        raise RuntimeError("bpe.train 不可用；请先构建 cpp 扩展")

    cpu_cores = min(multiprocessing.cpu_count(), 24)
    total_word_counts = get_word_counts_parallel(str(input_path), special_tokens, cpu_cores)
    sorted_items = sorted(total_word_counts.items(), key=lambda x: (-x[1], x[0]))
    distinct_words = [w for w, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    result = train_fn(distinct_words, counts, int(vocab_size), list(special_tokens))
    return BPETokenizer(vocab=result.vocab, merges=result.merges, special_tokens=special_tokens)


def _iter_text_chunks(path: str | os.PathLike, chunk_bytes: int) -> Iterable[str]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                return
            yield chunk


def _dtype_max_value(dtype: np.dtype) -> int | None:
    if dtype.kind not in {"u", "i"}:
        return None
    info = np.iinfo(dtype)
    return int(info.max)


def _encode_text_to_bin(
    *,
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    dtype: np.dtype,
    overwrite: bool,
    chunk_bytes: int = 1 << 22,
    buffer_tokens: int = 50_000_000,
    encode_workers: int = 8,
    encode_backend: str = "pool",
    encode_slot_tokens: int = 0,
    encode_num_slots: int = 0,
    log_interval_sec: float = 5.0,
) -> None:
    out_path = Path(output_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在: {out_path}（传 --overwrite 以覆盖）")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_value = _dtype_max_value(dtype)
    total_tokens = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0

    def _iter_complete_texts() -> Iterable[str]:
        buffer = ""
        for chunk in _iter_text_chunks(input_path, chunk_bytes=chunk_bytes):
            buffer += chunk
            boundary = tokenizer._find_last_boundary(buffer)
            if boundary > 0:
                yield buffer[:boundary]
                buffer = buffer[boundary:]
        if buffer:
            yield buffer

    with open(out_path, "wb") as out_f:

        def log_progress() -> None:
            nonlocal last_log_time, last_log_tokens
            now = time.time()
            dt = now - last_log_time
            if dt >= log_interval_sec:
                d_tokens = total_tokens - last_log_tokens
                tok_per_s = d_tokens / dt if dt > 0 else float("nan")
                elapsed = now - start_time
                print(
                    f"encode {out_path.name}: tokens={total_tokens} tok/s={tok_per_s:.0f} time={elapsed / 60:.1f}m",
                    flush=True,
                )
                last_log_time = now
                last_log_tokens = total_tokens

        if encode_workers and encode_workers > 1 and encode_backend == "shm":
            ctx = multiprocessing.get_context("fork")
            slot_tokens = encode_slot_tokens if encode_slot_tokens > 0 else max(1_000_000, int(chunk_bytes) * 2)
            num_slots = encode_num_slots if encode_num_slots > 0 else int(encode_workers) * 2
            shm_size = int(num_slots) * int(slot_tokens) * int(dtype.itemsize)
            shm = shared_memory.SharedMemory(create=True, size=shm_size)
            slots = np.ndarray((num_slots, slot_tokens), dtype=dtype, buffer=shm.buf)
            task_q = ctx.Queue(maxsize=num_slots * 2)
            done_q = ctx.Queue(maxsize=num_slots * 2)
            free_q = ctx.Queue()
            for i in range(num_slots):
                free_q.put(i)

            workers: list[multiprocessing.Process] = []
            try:
                for _ in range(int(encode_workers)):
                    p = ctx.Process(
                        target=_encode_shm_worker_loop,
                        args=(task_q, done_q),
                        kwargs={
                            "tokenizer": tokenizer,
                            "shm_name": shm.name,
                            "dtype_str": dtype.str,
                            "slot_tokens": slot_tokens,
                            "num_slots": num_slots,
                            "max_value": max_value,
                        },
                    )
                    p.start()
                    workers.append(p)

                pending: dict[int, tuple[int, int]] = {}
                next_seq = 0
                submitted = 0
                completed = 0

                def process_one_done(*, block: bool) -> None:
                    nonlocal completed, next_seq, total_tokens
                    done_seq, slot_id, length, err = done_q.get(block=block)
                    if err is not None:
                        raise RuntimeError(err)
                    pending[int(done_seq)] = (int(slot_id), int(length))
                    completed += 1
                    while next_seq in pending:
                        slot_id2, length2 = pending.pop(next_seq)
                        out_f.write(slots[slot_id2, :length2].tobytes())
                        total_tokens += length2
                        free_q.put(slot_id2)
                        if (total_tokens & 0x3FFFF) == 0:
                            log_progress()
                        next_seq += 1

                for text in _iter_complete_texts():
                    while True:
                        try:
                            slot_id = free_q.get_nowait()
                            break
                        except queue.Empty:
                            process_one_done(block=True)
                    task_q.put((submitted, text, slot_id))
                    submitted += 1

                    while True:
                        try:
                            process_one_done(block=False)
                        except queue.Empty:
                            break

                for _ in workers:
                    task_q.put(None)

                while completed < submitted:
                    process_one_done(block=True)
            finally:
                for p in workers:
                    p.join(timeout=1)
                for p in workers:
                    if p.is_alive():
                        p.terminate()
                for p in workers:
                    if p.is_alive():
                        p.join(timeout=1)
                shm.close()
                shm.unlink()
        else:
            buf = np.empty((buffer_tokens,), dtype=dtype)
            n = 0

            def write_tok(tok: int) -> None:
                nonlocal n, total_tokens
                if max_value is not None and tok > max_value:
                    raise ValueError(f"token id {tok} 超出 dtype={dtype} 可表示范围（max={max_value}）")
                buf[n] = tok
                n += 1
                total_tokens += 1
                if n == buf.shape[0]:
                    out_f.write(buf.tobytes())
                    n = 0
                if (total_tokens & 0x3FFFF) == 0:
                    log_progress()

            if encode_workers and encode_workers > 1:
                ctx = multiprocessing.get_context("fork")
                with ctx.Pool(
                    processes=int(encode_workers), initializer=_encode_worker_init, initargs=(tokenizer,)
                ) as pool:
                    for token_ids in pool.imap(_encode_worker, _iter_complete_texts(), chunksize=1):
                        for tok in token_ids:
                            write_tok(tok)
            else:
                for tok in tokenizer.encode_iterable(_iter_text_chunks(input_path, chunk_bytes=chunk_bytes)):
                    write_tok(tok)
            if n:
                out_f.write(buf[:n].tobytes())
    elapsed = time.time() - start_time
    tok_per_s = total_tokens / elapsed if elapsed > 0 else float("nan")
    print(
        f"encode done {out_path.name}: tokens={total_tokens} tok/s={tok_per_s:.0f} time={elapsed / 60:.1f}m", flush=True
    )


def _train_from_text(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.txt"
    train_bin = out_dir / "train.bin"
    valid_bin = out_dir / "valid.bin"

    dtype = _parse_numpy_dtype(args.data_dtype)

    print(f"stage=bpe_train vocab_size={args.vocab_size}", flush=True)
    t0 = time.time()
    tokenizer = _train_bpe_tokenizer(
        input_path=args.train_text,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    t1 = time.time()
    print(f"stage=bpe_train_done time={t1 - t0:.1f}s", flush=True)
    if (vocab_path.exists() or merges_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{vocab_path} 或 {merges_path} 已存在（传 --overwrite 以覆盖）")
    tokenizer.save(str(vocab_path), str(merges_path))
    print(f"stage=tokenizer_saved vocab={vocab_path.name} merges={merges_path.name}", flush=True)
    actual_vocab_size = tokenizer.get_vocab_size()

    print(f"stage=encode_train path={Path(args.train_text).name}", flush=True)
    _encode_text_to_bin(
        tokenizer=tokenizer,
        input_path=args.train_text,
        output_path=train_bin,
        dtype=dtype,
        overwrite=args.overwrite,
        encode_workers=args.encode_workers,
        encode_backend=args.encode_backend,
        encode_slot_tokens=args.encode_slot_tokens,
        encode_num_slots=args.encode_num_slots,
    )
    print(f"stage=encode_valid path={Path(args.valid_text).name}", flush=True)
    _encode_text_to_bin(
        tokenizer=tokenizer,
        input_path=args.valid_text,
        output_path=valid_bin,
        dtype=dtype,
        overwrite=args.overwrite,
        encode_workers=args.encode_workers,
        encode_backend=args.encode_backend,
        encode_slot_tokens=args.encode_slot_tokens,
        encode_num_slots=args.encode_num_slots,
    )

    print("stage=train", flush=True)
    train_args = argparse.Namespace(**vars(args))
    train_args.vocab_size = actual_vocab_size
    train_args.train_data = str(train_bin)
    train_args.valid_data = str(valid_bin)
    _train(train_args)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _build_optimizer(
    name: str, params: Iterable[torch.nn.Parameter], args: argparse.Namespace
) -> torch.optim.Optimizer:
    if name == "adamw":
        return AdamW(
            params,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    if name == "sgd":
        return SGD(params, lr=args.learning_rate)
    raise ValueError(f"未知 optimizer: {name}")


def _train(args: argparse.Namespace) -> None:
    device = _auto_device(args.device)
    if device.startswith("cuda") and args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    dtype = _parse_numpy_dtype(args.data_dtype)
    train_tokens = _open_tokens(args.train_data, dtype=dtype)
    valid_tokens = _open_tokens(args.valid_data, dtype=dtype)

    if args.d_ff is None:
        args.d_ff = 4 * args.d_model if args.ffn_type == "silu" else 1344

    ckpt_config = vars(args).copy()
    base_model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        use_rmsnorm=not args.no_rmsnorm,
        norm_style=args.norm_style,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    ).to(device)

    model: torch.nn.Module = base_model
    if args.compile:
        preferred_backend = args.compile_backend
        if preferred_backend is None and device.startswith("mps"):
            preferred_backend = "aot_eager"
        try:
            model = torch.compile(model, backend=preferred_backend) if preferred_backend else torch.compile(model)
        except Exception as e:
            print(f"compile_failed backend={preferred_backend or 'inductor'} err={type(e).__name__}: {e}", flush=True)
            if preferred_backend != "aot_eager":
                try:
                    model = torch.compile(model, backend="aot_eager")
                    print("compile_fallback=success backend=aot_eager", flush=True)
                except Exception as e2:
                    print(f"compile_fallback=failed backend=aot_eager err={type(e2).__name__}: {e2}", flush=True)
                    print("compile_disabled=true", flush=True)
                    model = base_model
            else:
                print("compile_disabled=true", flush=True)
                model = base_model

    optimizer = _build_optimizer(args.optimizer, base_model.parameters(), args)

    it0 = 0
    if args.resume_from:
        it0 = int(load_checkpoint(args.resume_from, model=model, optimizer=optimizer, map_location=device))
        print(f"resume_from={args.resume_from} step={it0}", flush=True)
        if it0 >= args.max_steps:
            print(f"nothing_to_do: checkpoint_step={it0} >= max_steps={args.max_steps}", flush=True)
            return

    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "rope_theta": args.rope_theta,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "min_learning_rate": args.min_learning_rate,
                "warmup_iters": args.warmup_iters,
                "cosine_cycle_iters": args.cosine_cycle_iters,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "eps": args.eps,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_steps": args.max_steps,
                "grad_clip": args.grad_clip,
                "no_rmsnorm": args.no_rmsnorm,
                "norm_style": args.norm_style,
                "no_rope": args.no_rope,
                "ffn_type": args.ffn_type,
                "device": device,
            },
        )

    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0

    for it in range(it0, args.max_steps):
        model.train()
        lr = lr_cosine_schedule(
            it=it,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        _set_lr(optimizer, lr)

        x, y = _get_batch(train_tokens, args.batch_size, args.context_length, device, rng)
        try:
            logits = model(x)
        except Exception as e:
            backend_failed = type(e).__name__ == "BackendCompilerFailed"
            triton_missing = "Cannot find a working triton installation" in str(e)
            if backend_failed or triton_missing:
                print(f"compile_runtime_failed err={type(e).__name__}: {e}", flush=True)
                print("compile_disabled=true", flush=True)
                model = base_model
                logits = model(x)
            else:
                raise
        loss = _loss_per_token(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        wall_time = time.time() - start_time
        tokens_seen = (it + 1) * args.batch_size * args.context_length

        if (it + 1) % args.log_interval == 0:
            now = time.time()
            dt = now - last_log_time
            d_tokens = tokens_seen - last_log_tokens
            tok_per_s = d_tokens / dt if dt > 0 else float("nan")
            last_log_time = now
            last_log_tokens = tokens_seen

            train_loss = float(loss.detach().float().cpu().item())
            record = {
                "split": "train",
                "step": it + 1,
                "loss": train_loss,
                "lr": lr,
                "wall_time_sec": wall_time,
                "tokens_seen": tokens_seen,
                "tokens_per_sec": tok_per_s,
            }
            pct = 100.0 * (it + 1) / args.max_steps
            total_tokens = args.max_steps * args.batch_size * args.context_length
            remaining_tokens = max(total_tokens - tokens_seen, 0)
            eta = remaining_tokens / tok_per_s if np.isfinite(tok_per_s) and tok_per_s > 0 else float("nan")
            print(
                f"step={it + 1}/{args.max_steps} ({pct:.1f}%) loss={train_loss:.4f} lr={lr:.3e} "
                f"tok/s={tok_per_s:.0f} eta={_format_duration(eta)} time={wall_time / 60:.1f}m",
                flush=True,
            )
            _append_jsonl(metrics_path, record)
            if wandb_run:
                wandb_run.log(record, step=it + 1)

        if (it + 1) % args.eval_interval == 0:
            val_loss = _eval_loss(
                model=model,
                dataset=valid_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
                rng=eval_rng,
            )
            record = {
                "split": "valid",
                "step": it + 1,
                "loss": val_loss,
                "lr": lr,
                "wall_time_sec": wall_time,
                "tokens_seen": tokens_seen,
            }
            print(f"valid step={it + 1} loss={val_loss:.4f}", flush=True)
            _append_jsonl(metrics_path, record)
            if wandb_run:
                wandb_run.log(record, step=it + 1)

        if args.ckpt_interval and (it + 1) % args.ckpt_interval == 0:
            ckpt_path = out_dir / f"checkpoint_step_{it + 1}.pt"
            save_checkpoint(model, optimizer, iteration=it + 1, out=ckpt_path, config=ckpt_config)
            print(f"ckpt_saved={ckpt_path.name}", flush=True)

    final_ckpt = out_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, iteration=args.max_steps, out=final_ckpt, config=ckpt_config)
    print(f"ckpt_saved={final_ckpt.name}", flush=True)
    if wandb_run:
        wandb_run.finish()


@torch.no_grad()
def _generate(args: argparse.Namespace) -> None:
    device = _auto_device(args.device)
    if device.startswith("cuda") and args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)

    tokenizer = _load_bpe_tokenizer(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)
    vocab_size_from_tokenizer = tokenizer.get_vocab_size()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = ckpt_config if isinstance(ckpt_config, dict) else {}

    vocab_size = int(config.get("vocab_size", vocab_size_from_tokenizer))
    if "vocab_size" in config and vocab_size_from_tokenizer != vocab_size:
        raise ValueError(
            f"checkpoint vocab_size={vocab_size} 与 tokenizer vocab_size={vocab_size_from_tokenizer} 不一致"
        )

    context_length = int(config.get("context_length", args.context_length))
    d_model = int(config.get("d_model", args.d_model))
    num_layers = int(config.get("num_layers", args.num_layers))
    num_heads = int(config.get("num_heads", args.num_heads))
    rope_theta = float(config.get("rope_theta", args.rope_theta))
    no_rmsnorm = bool(config.get("no_rmsnorm", args.no_rmsnorm))
    norm_style = str(config.get("norm_style", args.norm_style))
    no_rope = bool(config.get("no_rope", args.no_rope))
    ffn_type = str(config.get("ffn_type", args.ffn_type))
    d_ff = config.get("d_ff", args.d_ff)
    if d_ff is None:
        d_ff = 4 * d_model if ffn_type == "silu" else 1344
    d_ff = int(d_ff)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        use_rmsnorm=not no_rmsnorm,
        norm_style=norm_style,
        use_rope=not no_rope,
        ffn_type=ffn_type,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.compile:
        if device.startswith("mps") and args.compile_backend is None:
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model, backend=args.compile_backend) if args.compile_backend else torch.compile(model)
        model.eval()

    eos_id = None
    if args.eos_token:
        eos_bytes = args.eos_token.encode("utf-8")
        eos_id = tokenizer.token_to_id.get(eos_bytes, None)

    ids = tokenizer.encode(args.prompt)
    generated: list[int] = list(ids)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    for _ in range(args.max_new_tokens):
        x = torch.tensor([generated[-context_length:]], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]
        logits = logits / max(args.temperature, 1e-8)

        if args.repetition_penalty and args.repetition_penalty != 1.0 and generated:
            penalty = float(args.repetition_penalty)
            unique_ids = torch.unique(torch.tensor(generated, dtype=torch.long, device=device))
            logits[:, unique_ids] = logits[:, unique_ids] / penalty

        if args.top_k and args.top_k > 0:
            v, _ = torch.topk(logits, k=min(args.top_k, logits.shape[-1]))
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.tensor(float("-inf"), device=device), logits)

        if args.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            to_remove = cumprobs > args.top_p
            to_remove[:, 1:] = to_remove[:, :-1].clone()
            to_remove[:, 0] = False
            remove_mask = to_remove.scatter(1, sorted_idx, to_remove)
            logits = logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        if args.stop_strings:
            text_so_far = tokenizer.decode(generated)
            for s in args.stop_strings:
                if not s:
                    continue
                stop_at = text_so_far.find(s)
                if stop_at != -1:
                    print(text_so_far[:stop_at], flush=True)
                    return

    text = tokenizer.decode(generated)
    print(text, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cs336_basics.main")
    sub = p.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--train-data", required=True)
    train.add_argument("--valid-data", required=True)
    train.add_argument("--data-dtype", default="uint16")
    train.add_argument("--out-dir", required=True)
    train.add_argument("--resume-from", default=None)

    train.add_argument("--vocab-size", type=int, required=True)
    train.add_argument("--context-length", type=int, default=256)
    train.add_argument("--d-model", type=int, default=512)
    train.add_argument("--d-ff", type=int, default=None)
    train.add_argument("--num-layers", type=int, default=4)
    train.add_argument("--num-heads", type=int, default=16)
    train.add_argument("--rope-theta", type=float, default=10000.0)

    train.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--min-learning-rate", type=float, default=3e-5)
    train.add_argument("--warmup-iters", type=int, default=200)
    train.add_argument("--cosine-cycle-iters", type=int, default=5000)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.95)
    train.add_argument("--eps", type=float, default=1e-8)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=1.0)

    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--max-steps", type=int, default=5000)
    train.add_argument("--log-interval", type=int, default=10)
    train.add_argument("--eval-interval", type=int, default=200)
    train.add_argument("--eval-iters", type=int, default=50)
    train.add_argument("--ckpt-interval", type=int, default=500)

    train.add_argument("--no-rmsnorm", action="store_true")
    train.add_argument("--norm-style", choices=["pre", "post"], default="pre")
    train.add_argument("--no-rope", action="store_true")
    train.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--matmul-precision", choices=["high", "medium"], default=None)
    train.add_argument("--compile", action="store_true")
    train.add_argument("--compile-backend", default=None)

    train.add_argument("--wandb-project", default="")
    train.add_argument("--wandb-name", default=None)

    tft = sub.add_parser("train-from-text")
    tft.add_argument("--train-text", required=True)
    tft.add_argument("--valid-text", required=True)
    tft.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    tft.add_argument("--data-dtype", default="uint16")
    tft.add_argument("--out-dir", required=True)
    tft.add_argument("--overwrite", action="store_true")
    tft.add_argument("--resume-from", default=None)
    tft.add_argument("--encode-workers", type=int, default=8)
    tft.add_argument("--encode-backend", choices=["pool", "shm"], default="pool")
    tft.add_argument("--encode-slot-tokens", type=int, default=0)
    tft.add_argument("--encode-num-slots", type=int, default=0)

    tft.add_argument("--vocab-size", type=int, default=10000)
    tft.add_argument("--context-length", type=int, default=256)
    tft.add_argument("--d-model", type=int, default=512)
    tft.add_argument("--d-ff", type=int, default=None)
    tft.add_argument("--num-layers", type=int, default=4)
    tft.add_argument("--num-heads", type=int, default=16)
    tft.add_argument("--rope-theta", type=float, default=10000.0)

    tft.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    tft.add_argument("--learning-rate", type=float, default=3e-4)
    tft.add_argument("--min-learning-rate", type=float, default=3e-5)
    tft.add_argument("--warmup-iters", type=int, default=200)
    tft.add_argument("--cosine-cycle-iters", type=int, default=5000)
    tft.add_argument("--beta1", type=float, default=0.9)
    tft.add_argument("--beta2", type=float, default=0.95)
    tft.add_argument("--eps", type=float, default=1e-8)
    tft.add_argument("--weight-decay", type=float, default=0.1)
    tft.add_argument("--grad-clip", type=float, default=1.0)

    tft.add_argument("--batch-size", type=int, default=32)
    tft.add_argument("--max-steps", type=int, default=5000)
    tft.add_argument("--log-interval", type=int, default=10)
    tft.add_argument("--eval-interval", type=int, default=200)
    tft.add_argument("--eval-iters", type=int, default=50)
    tft.add_argument("--ckpt-interval", type=int, default=500)

    tft.add_argument("--no-rmsnorm", action="store_true")
    tft.add_argument("--norm-style", choices=["pre", "post"], default="pre")
    tft.add_argument("--no-rope", action="store_true")
    tft.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    tft.add_argument("--device", default="auto")
    tft.add_argument("--seed", type=int, default=1337)
    tft.add_argument("--matmul-precision", choices=["high", "medium"], default=None)
    tft.add_argument("--compile", action="store_true")
    tft.add_argument("--compile-backend", default=None)

    tft.add_argument("--wandb-project", default="")
    tft.add_argument("--wandb-name", default=None)

    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--vocab-path", required=True)
    gen.add_argument("--merges-path", required=True)
    gen.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])

    gen.add_argument("--context-length", type=int, default=256)
    gen.add_argument("--d-model", type=int, default=512)
    gen.add_argument("--d-ff", type=int, default=None)
    gen.add_argument("--num-layers", type=int, default=4)
    gen.add_argument("--num-heads", type=int, default=16)
    gen.add_argument("--rope-theta", type=float, default=10000.0)

    gen.add_argument("--no-rmsnorm", action="store_true")
    gen.add_argument("--norm-style", choices=["pre", "post"], default="pre")
    gen.add_argument("--no-rope", action="store_true")
    gen.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.add_argument("--stop-strings", nargs="*", default=[])
    gen.add_argument("--eos-token", default="<|endoftext|>")
    gen.add_argument("--seed", type=int, default=1337)
    gen.add_argument("--device", default="auto")
    gen.add_argument("--matmul-precision", choices=["high", "medium"], default=None)
    gen.add_argument("--compile", action="store_true")
    gen.add_argument("--compile-backend", default=None)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        _train(args)
        return
    if args.cmd == "train-from-text":
        _train_from_text(args)
        return
    if args.cmd == "generate":
        _generate(args)
        return
    raise RuntimeError(f"未知命令: {args.cmd}")


if __name__ == "__main__":
    main()
