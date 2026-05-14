"""Print or run LoRA target_modules ablation training/evaluation commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_rank_ablation import (
    adapter_size_command,
    eval_command,
    read_yaml,
    shell_join,
    train_command,
)


DEFAULT_CONFIGS = [
    "configs/ablations/sft_r32_qv.yaml",
    "configs/ablations/sft_r32_qkvo.yaml",
    "configs/ablations/sft_r32_attn_mlp.yaml",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run or print LoRA target_modules ablation commands."
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Target-module ablation YAML configs to run in order.",
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--with_eval", action="store_true", help="Also print/run eval commands.")
    parser.add_argument("--run", action="store_true", help="Actually run commands instead of printing them.")
    parser.add_argument("--eval_file", default="data/eval/eval_seed.json")
    parser.add_argument("--report_dir", default="outputs/ablations/target_modules_reports")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Print or run target_modules ablation commands."""
    args = parse_args()
    commands: list[list[str]] = []
    size_commands: list[str] = []

    for config_name in args.configs:
        config_path = Path(config_name)
        config = read_yaml(config_path)
        commands.append(train_command(args.python_bin, config_path))
        if args.with_eval:
            commands.append(
                eval_command(
                    args.python_bin,
                    config,
                    config_path,
                    args.eval_file,
                    args.report_dir,
                    args.max_new_tokens,
                    args.temperature,
                    args.eval_batch_size,
                )
            )
            size_commands.append(adapter_size_command(config))

    if not args.run:
        print("# LoRA target_modules ablation commands")
        for command in commands:
            print(shell_join(command))
        if size_commands:
            print("\n# Adapter size commands")
            for command in size_commands:
                print(command)
        return

    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    for command in commands:
        print(f"\n$ {shell_join(command)}", flush=True)
        subprocess.run(command, check=True)

    if args.with_eval:
        print("\n# Adapter sizes")
        for command in size_commands:
            print(f"$ {command}", flush=True)
            subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()
