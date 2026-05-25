"""
DPO beta ablation runner.

Modes:
  --print-commands   Print all shell commands (default)
  --train            Run all three DPO training jobs sequentially
  --eval             Run all eval jobs (SFT-only baseline + 3 beta adapters)
  --summarize        Read completed JSON outputs and write the summary markdown
  --all              Train + eval + summarize in sequence
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER = "outputs/final_sft_r32_attn_mlp"
DPO_EVAL_FILE = "data/dpo/dpo_eval.jsonl"
RULE_EVAL_FILE = "data/eval/eval_seed.json"
SUMMARY_PATH = "docs/experiments/dpo_beta_ablation_summary.md"

BETA_CONFIGS = [
    {"beta": "005", "config": "configs/dpo_beta_005.yaml", "adapter": "outputs/dpo_beta_005"},
    {"beta": "010", "config": "configs/dpo_beta_010.yaml", "adapter": "outputs/dpo_beta_010"},
    {"beta": "030", "config": "configs/dpo_beta_030.yaml", "adapter": "outputs/dpo_beta_030"},
]


def train_commands() -> list[str]:
    return [
        f"python scripts/train_dpo.py --config {c['config']}"
        for c in BETA_CONFIGS
    ]


def eval_commands() -> list[str]:
    cmds = []
    cmds.append(
        f"python scripts/eval_dpo_preference.py"
        f" --model_name_or_path {MODEL}"
        f" --adapter_path {SFT_ADAPTER}"
        f" --eval_file {DPO_EVAL_FILE}"
        f" --output_json {SFT_ADAPTER}/preference_eval_beta_ablation.json"
        f" --output_md {SFT_ADAPTER}/preference_eval_beta_ablation.md"
        f" --batch_size 16"
    )
    for c in BETA_CONFIGS:
        adapter = c["adapter"]
        cmds.append(
            f"python scripts/eval_dpo_preference.py"
            f" --model_name_or_path {MODEL}"
            f" --adapter_path {adapter}"
            f" --eval_file {DPO_EVAL_FILE}"
            f" --output_json {adapter}/preference_eval.json"
            f" --output_md {adapter}/preference_eval.md"
            f" --batch_size 16"
        )
        cmds.append(
            f"python scripts/eval_sft.py"
            f" --model_name_or_path {MODEL}"
            f" --adapter_path {adapter}"
            f" --eval_file {RULE_EVAL_FILE}"
            f" --output_json {adapter}/rule_eval.json"
            f" --output_md {adapter}/rule_eval.md"
            f" --eval_batch_size 16"
        )
    return cmds


def run_commands(commands: list[str]) -> None:
    for cmd in commands:
        print(f"\n>>> {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"ERROR: command exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)


def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def extract_preference(data: dict | None) -> dict:
    if data is None:
        return {"win_rate": "N/A", "wins": "N/A", "total": "N/A", "chosen_score": "N/A", "rejected_score": "N/A", "margin": "N/A"}
    summary = data.get("summary", {})
    chosen = summary.get("average_chosen_score", float("nan"))
    rejected = summary.get("average_rejected_score", float("nan"))
    try:
        margin = round(chosen - rejected, 4)
    except TypeError:
        margin = "N/A"
    raw_wr = summary.get("chosen_win_rate", None)
    win_rate = round(raw_wr * 100, 2) if isinstance(raw_wr, float) else "N/A"
    return {
        "win_rate": win_rate,
        "wins": summary.get("chosen_wins", "N/A"),
        "total": summary.get("total_pairs", "N/A"),
        "chosen_score": round(chosen, 4) if isinstance(chosen, float) else "N/A",
        "rejected_score": round(rejected, 4) if isinstance(rejected, float) else "N/A",
        "margin": margin,
    }


def extract_rule(data: dict | None) -> dict:
    if data is None:
        return {
            "passed": "N/A", "total": "N/A", "pass_rate": "N/A",
            "leakage": "N/A", "truncation": "N/A",
            "no_extra_notes": "N/A", "mentions_international_office": "N/A",
            "mentions_official_office": "N/A", "has_closing": "N/A",
        }
    summary = data.get("summary", {})
    failed_counts = summary.get("failed_check_counts", {})
    raw_pr = summary.get("pass_rate", None)
    pass_rate = round(raw_pr * 100, 2) if isinstance(raw_pr, float) else "N/A"
    return {
        "passed": summary.get("passed_checks", "N/A"),
        "total": summary.get("total_checks", "N/A"),
        "pass_rate": pass_rate,
        "leakage": summary.get("final_response_prompt_leakage_count", 0),
        "truncation": summary.get("truncated_count", 0),
        "no_extra_notes": failed_counts.get("no_extra_notes", 0),
        "mentions_international_office": failed_counts.get("mentions_international_office", 0),
        "mentions_official_office": failed_counts.get("mentions_official_office", 0),
        "has_closing": failed_counts.get("has_closing", 0),
    }


def generate_summary() -> None:
    sft_pref = extract_preference(load_json(f"{SFT_ADAPTER}/preference_eval_beta_ablation.json"))

    rows_pref = []
    rows_rule = []
    beta_labels = {"005": "0.05", "010": "0.10 (baseline)", "030": "0.30"}

    for c in BETA_CONFIGS:
        label = beta_labels[c["beta"]]
        pref = extract_preference(load_json(f"{c['adapter']}/preference_eval.json"))
        rule = extract_rule(load_json(f"{c['adapter']}/rule_eval.json"))
        rows_pref.append((label, pref))
        rows_rule.append((label, rule))

    lines = [
        "# DPO Beta Ablation Summary",
        "",
        "## Setup",
        "",
        f"- Base model: `{MODEL}`",
        f"- SFT adapter: `{SFT_ADAPTER}`",
        "- DPO data: 151 pairs, 121 train / 30 eval",
        "- Rule eval: 60 samples, 311 checks",
        "- Epochs: 1 | LR: 5e-6 | LoRA r=32 | Seed: 42",
        "",
        "## Preference Eval (30-pair eval set)",
        "",
        "| Beta | Wins | Total | Win Rate | Chosen Score | Rejected Score | Margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| SFT-only (baseline) | {sft_pref['wins']} | {sft_pref['total']} | {sft_pref['win_rate']}% | {sft_pref['chosen_score']} | {sft_pref['rejected_score']} | {sft_pref['margin']} |",
    ]
    for label, p in rows_pref:
        lines.append(
            f"| {label} | {p['wins']} | {p['total']} | {p['win_rate']}% | {p['chosen_score']} | {p['rejected_score']} | {p['margin']} |"
        )

    lines += [
        "",
        "## Rule Eval (311 checks)",
        "",
        "| Beta | Passed | Total | Pass Rate | Leakage | Truncation | no_extra_notes | intl_office | official_office | has_closing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in rows_rule:
        lines.append(
            f"| {label} | {r['passed']} | {r['total']} | {r['pass_rate']}% | {r['leakage']} | {r['truncation']} | {r['no_extra_notes']} | {r['mentions_international_office']} | {r['mentions_official_office']} | {r['has_closing']} |"
        )

    lines += [
        "",
        "## Decision",
        "",
        "<!-- Fill in after reviewing results -->",
        "",
        "Best beta: **TBD**",
        "",
        "## Qualitative Notes",
        "",
        "<!-- Add observations about output quality, new failure patterns, or unexpected behavior -->",
        "",
        "## Future Work",
        "",
        "Preference data quality / label noise robustness can be tested later using 10% and 30% chosen-rejected swap noise.",
    ]

    out = Path(SUMMARY_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {SUMMARY_PATH}")


def print_all_commands() -> None:
    print("# === TRAINING ===")
    for cmd in train_commands():
        print(cmd)
    print("\n# === EVAL ===")
    for cmd in eval_commands():
        print(cmd)
    print(f"\n# === SUMMARIZE ===")
    print(f"python scripts/run_dpo_beta_ablation.py --summarize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO beta ablation runner.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--print-commands", action="store_true", default=False)
    group.add_argument("--train", action="store_true")
    group.add_argument("--eval", action="store_true")
    group.add_argument("--summarize", action="store_true")
    group.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train:
        run_commands(train_commands())
    elif args.eval:
        run_commands(eval_commands())
    elif args.summarize:
        generate_summary()
    elif args.all:
        run_commands(train_commands())
        run_commands(eval_commands())
        generate_summary()
    else:
        print_all_commands()


if __name__ == "__main__":
    main()
