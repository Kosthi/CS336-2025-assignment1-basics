"""
命令行入口（解析参数、加载 YAML、分发子命令）。

最佳实践要点：
1) parser 只负责定义参数，不做业务逻辑；
2) YAML 仅作为默认值来源，CLI 显式参数永远优先；
3) 子命令逻辑拆分到独立模块，main.py 保持很薄。
"""

from __future__ import annotations

import argparse
import sys

from . import yaml_config
from .generate import generate
from .text_pipeline import train_from_text
from .train import train


def build_parser() -> argparse.ArgumentParser:
    """定义完整 CLI（train / train-from-text / generate）。"""

    p = argparse.ArgumentParser(prog="cs336_basics.main")
    sub = p.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", default=None)
    train_p.add_argument("--train-data", required=True)
    train_p.add_argument("--valid-data", required=True)
    train_p.add_argument("--data-dtype", default="uint16")
    train_p.add_argument("--out-dir", required=True)
    train_p.add_argument("--resume-from", default=None)

    train_p.add_argument("--vocab-size", type=int, required=True)
    train_p.add_argument("--context-length", type=int, default=256)
    train_p.add_argument("--d-model", type=int, default=512)
    train_p.add_argument("--d-ff", type=int, default=None)
    train_p.add_argument("--num-layers", type=int, default=4)
    train_p.add_argument("--num-heads", type=int, default=16)
    train_p.add_argument("--rope-theta", type=float, default=10000.0)

    train_p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    train_p.add_argument("--learning-rate", type=float, default=3e-4)
    train_p.add_argument("--min-learning-rate", type=float, default=3e-5)
    train_p.add_argument("--warmup-iters", type=int, default=200)
    train_p.add_argument("--cosine-cycle-iters", type=int, default=None)
    train_p.add_argument("--beta1", type=float, default=0.9)
    train_p.add_argument("--beta2", type=float, default=0.95)
    train_p.add_argument("--eps", type=float, default=1e-8)
    train_p.add_argument("--weight-decay", type=float, default=0.1)
    train_p.add_argument("--grad-clip", type=float, default=1.0)

    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--max-steps", type=int, default=5000)
    train_p.add_argument("--log-interval", type=int, default=10)
    train_p.add_argument("--eval-interval", type=int, default=200)
    train_p.add_argument("--eval-iters", type=int, default=50)
    train_p.add_argument("--ckpt-interval", type=int, default=500)

    train_p.add_argument("--no-rmsnorm", action="store_true")
    train_p.add_argument("--norm-style", choices=["pre", "post"], default="pre")
    train_p.add_argument("--no-rope", action="store_true")
    train_p.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--matmul-precision", choices=["high", "medium"], default=None)
    train_p.add_argument("--compile", action="store_true")
    train_p.add_argument("--compile-backend", default=None)

    train_p.add_argument("--wandb-project", default="")
    train_p.add_argument("--wandb-name", default=None)

    tft_p = sub.add_parser("train-from-text")
    tft_p.add_argument("--config", default=None)
    tft_p.add_argument("--train-text", required=True)
    tft_p.add_argument("--valid-text", required=True)
    tft_p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    tft_p.add_argument("--data-dtype", default="uint16")
    tft_p.add_argument("--out-dir", required=True)
    tft_p.add_argument("--overwrite", action="store_true")
    tft_p.add_argument("--resume-from", default=None)
    tft_p.add_argument("--encode-workers", type=int, default=8)
    tft_p.add_argument("--encode-backend", choices=["pool", "shm"], default="pool")
    tft_p.add_argument("--encode-slot-tokens", type=int, default=0)
    tft_p.add_argument("--encode-num-slots", type=int, default=0)

    tft_p.add_argument("--vocab-size", type=int, default=10000)
    tft_p.add_argument("--context-length", type=int, default=256)
    tft_p.add_argument("--d-model", type=int, default=512)
    tft_p.add_argument("--d-ff", type=int, default=None)
    tft_p.add_argument("--num-layers", type=int, default=4)
    tft_p.add_argument("--num-heads", type=int, default=16)
    tft_p.add_argument("--rope-theta", type=float, default=10000.0)

    tft_p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    tft_p.add_argument("--learning-rate", type=float, default=3e-4)
    tft_p.add_argument("--min-learning-rate", type=float, default=3e-5)
    tft_p.add_argument("--warmup-iters", type=int, default=200)
    tft_p.add_argument("--cosine-cycle-iters", type=int, default=None)
    tft_p.add_argument("--beta1", type=float, default=0.9)
    tft_p.add_argument("--beta2", type=float, default=0.999)
    tft_p.add_argument("--eps", type=float, default=1e-8)
    tft_p.add_argument("--weight-decay", type=float, default=0.1)
    tft_p.add_argument("--grad-clip", type=float, default=1.0)

    tft_p.add_argument("--batch-size", type=int, default=32)
    tft_p.add_argument("--max-steps", type=int, default=5000)
    tft_p.add_argument("--log-interval", type=int, default=10)
    tft_p.add_argument("--eval-interval", type=int, default=200)
    tft_p.add_argument("--eval-iters", type=int, default=50)
    tft_p.add_argument("--ckpt-interval", type=int, default=500)

    tft_p.add_argument("--no-rmsnorm", action="store_true")
    tft_p.add_argument("--norm-style", choices=["pre", "post"], default="pre")
    tft_p.add_argument("--no-rope", action="store_true")
    tft_p.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    tft_p.add_argument("--device", default="auto")
    tft_p.add_argument("--seed", type=int, default=42)
    tft_p.add_argument("--matmul-precision", choices=["high", "medium"], default=None)
    tft_p.add_argument("--compile", action="store_true")
    tft_p.add_argument("--compile-backend", default=None)

    tft_p.add_argument("--wandb-project", default="")
    tft_p.add_argument("--wandb-name", default=None)

    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--config", default=None)
    gen_p.add_argument("--checkpoint", required=True)
    gen_p.add_argument("--vocab-path", required=True)
    gen_p.add_argument("--merges-path", required=True)
    gen_p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])

    gen_p.add_argument("--context-length", type=int, default=256)
    gen_p.add_argument("--d-model", type=int, default=512)
    gen_p.add_argument("--d-ff", type=int, default=None)
    gen_p.add_argument("--num-layers", type=int, default=4)
    gen_p.add_argument("--num-heads", type=int, default=16)
    gen_p.add_argument("--rope-theta", type=float, default=10000.0)

    gen_p.add_argument("--no-rmsnorm", action="store_true")
    gen_p.add_argument("--norm-style", choices=["pre", "post"], default="pre")
    gen_p.add_argument("--no-rope", action="store_true")
    gen_p.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    gen_p.add_argument("--prompt", required=True)
    gen_p.add_argument("--max-new-tokens", type=int, default=256)
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--top-p", type=float, default=0.9)
    gen_p.add_argument("--top-k", type=int, default=0)
    gen_p.add_argument("--repetition-penalty", type=float, default=1.0)
    gen_p.add_argument("--stop-strings", nargs="*", default=[])
    gen_p.add_argument("--eos-token", default="<|endoftext|>")
    gen_p.add_argument("--seed", type=int, default=42)
    gen_p.add_argument("--device", default="auto")
    gen_p.add_argument("--matmul-precision", choices=["high", "medium"], default=None)
    gen_p.add_argument("--compile", action="store_true")
    gen_p.add_argument("--compile-backend", default=None)
    return p


def parse_args_with_yaml_defaults(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """两阶段解析：先用 argv 定位子命令，再注入 YAML 默认值，再 parse_args。"""

    if argv and not argv[0].startswith("-"):
        cmd = argv[0]
        subparser = yaml_config.find_subparser(parser, cmd)
        if subparser is not None:
            config_path = yaml_config.extract_config_path(argv[1:])
            if config_path:
                full = yaml_config.load_yaml_config(config_path)
                cmd_config = yaml_config.normalize_config_keys(yaml_config.select_cmd_config(full, cmd))
                overridden = yaml_config.collect_overridden_dests(subparser, argv[1:])
                required_dests, unknown_keys = yaml_config.apply_config_defaults(subparser, cmd_config, overridden)
                if unknown_keys:
                    keys = ", ".join(unknown_keys)
                    raise SystemExit(f"config 里存在未知字段（cmd={cmd}）: {keys}")
                args = parser.parse_args(argv)
                yaml_config.enforce_required(required_dests, args)
                return args
    return parser.parse_args(argv)


def run_cli(argv: list[str] | None = None) -> None:
    """统一 CLI 入口，便于测试与复用。"""

    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parse_args_with_yaml_defaults(parser, argv)

    if args.cmd == "train":
        train(args)
        return
    if args.cmd == "train-from-text":
        train_from_text(args)
        return
    if args.cmd == "generate":
        generate(args)
        return
    raise RuntimeError(f"未知命令: {args.cmd}")
