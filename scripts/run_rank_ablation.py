"""Print or run LoRA rank ablation training/evaluation commands."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIGS = [
    "configs/ablations/sft_lora_r4.yaml",
    "configs/ablations/sft_lora_r8.yaml",
    "configs/ablations/sft_lora_r16.yaml",
    "configs/ablations/sft_lora_r32.yaml",
]


def read_yaml(path: Path) -> dict:
    """Read a YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def shell_join(parts: list[str]) -> str:
    """Return a copy-pasteable shell command."""
    return " ".join(shlex.quote(part) for part in parts)


def train_command(python_bin: str, config_path: Path) -> list[str]:
    """Build the train_sft.py command."""
    return [python_bin, "scripts/train_sft.py", "--config", str(config_path)]


def eval_command(
    python_bin: str,
    config: dict,
    config_path: Path,
    eval_file: str,
    report_dir: str,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Build the eval_sft.py command for a finished adapter."""
    model_name = config["model"]["model_name_or_path"]
    output_dir = config["training"]["output_dir"]
    rank = config["lora"]["r"]
    prompt_template = config.get("data", {}).get("prompt_template", "chat")
    torch_dtype = config.get("model", {}).get("torch_dtype", "auto")
    report_prefix = Path(report_dir) / f"eval_report_{config_path.stem}"
    return [
        python_bin,
        "scripts/eval_sft.py",
        "--model_name_or_path",
        model_name,
        "--adapter_path",
        output_dir,
        "--eval_file",
        eval_file,
        "--output_json",
        f"{report_prefix}.json",
        "--output_md",
        f"{report_prefix}.md",
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--prompt_template",
        prompt_template,
        "--torch_dtype",
        torch_dtype,
    ]


def adapter_size_command(config: dict) -> str:
    """Return a shell snippet for measuring adapter directory size."""
    output_dir = config["training"]["output_dir"]
    return f"du -sh {shlex.quote(output_dir)}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run or print LoRA rank ablation commands.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Ablation YAML configs to run in order.",
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--with_eval", action="store_true", help="Also print/run eval commands.")
    parser.add_argument("--run", action="store_true", help="Actually run commands instead of printing them.")
    parser.add_argument("--eval_file", default="data/eval/eval_seed.json")
    parser.add_argument("--report_dir", default="outputs/ablations/reports")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """Print or run rank ablation commands."""
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
                )
            )
            size_commands.append(adapter_size_command(config))

    if not args.run:
        print("# LoRA rank ablation commands")
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
