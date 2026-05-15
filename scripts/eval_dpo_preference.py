"""Evaluate DPO preference win rate with average completion log probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def str_to_bool(value: str | bool) -> bool:
    """Parse a bool-like CLI value."""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def parse_torch_dtype(dtype_name: str) -> torch.dtype | str:
    """Convert dtype name to a torch dtype or auto."""
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
    return rows


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    """Load tokenizer, base model, and LoRA/DPO adapter."""
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": parse_torch_dtype(args.torch_dtype),
    }
    if args.use_4bit:
        if device.type != "cuda":
            raise RuntimeError("--use_4bit requires CUDA.")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"
    elif device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    if device.type != "cuda":
        model.to(device)
    model.eval()
    return tokenizer, model, device


def first_parameter_device(model: Any, fallback: torch.device) -> torch.device:
    """Find the device used by model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def add_eos(text: str, eos_token: str | None) -> str:
    """Append EOS to a response if available."""
    stripped = text.strip()
    if not eos_token or stripped.endswith(eos_token):
        return stripped
    return f"{stripped}{eos_token}"


def encode_pair(
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    add_response_eos: bool,
) -> tuple[list[int], int]:
    """Tokenize prompt and completion separately and return full ids plus prompt length."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_text = add_eos(completion, tokenizer.eos_token) if add_response_eos else completion.strip()
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        raise ValueError("Completion tokenized to zero tokens.")
    return prompt_ids + completion_ids, len(prompt_ids)


def score_batch(
    model: Any,
    tokenizer: Any,
    encoded: list[tuple[list[int], int]],
    device: torch.device,
) -> list[float]:
    """Compute average log probability for the completion span in each encoded sample."""
    if not encoded:
        return []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    max_len = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    completion_masks = torch.zeros((len(encoded), max_len - 1), dtype=torch.bool, device=device)

    for row_index, (ids, prompt_len) in enumerate(encoded):
        seq_len = len(ids)
        input_ids[row_index, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_index, :seq_len] = 1
        start = max(prompt_len - 1, 0)
        end = seq_len - 1
        completion_masks[row_index, start:end] = True

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        masked = token_log_probs * completion_masks
        counts = completion_masks.sum(dim=1).clamp_min(1)
        scores = masked.sum(dim=1) / counts
    return [float(score.detach().cpu()) for score in scores]


def summarize_group(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """Summarize win rates grouped by category or risk."""
    totals: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    for item in results:
        key = item.get(field, "")
        totals[key] += 1
        if item["chosen_win"]:
            wins[key] += 1
    return {
        key: {
            "total": totals[key],
            "wins": wins[key],
            "win_rate": wins[key] / totals[key] if totals[key] else 0.0,
        }
        for key in sorted(totals)
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a compact Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# DPO Preference Evaluation",
        "",
        f"- Total pairs: {summary['total_pairs']}",
        f"- Chosen wins: {summary['chosen_wins']}",
        f"- Chosen win rate: {summary['chosen_win_rate']:.2%}",
        f"- Average chosen score: {summary['average_chosen_score']:.4f}",
        f"- Average rejected score: {summary['average_rejected_score']:.4f}",
        f"- Batch size: {summary['batch_size']}",
        "",
        "## Per-Category Win Rate",
        "",
        "| Category | Wins | Total | Win Rate |",
        "|---|---:|---:|---:|",
    ]
    for category, item in summary["per_category"].items():
        lines.append(f"| `{category}` | {item['wins']} | {item['total']} | {item['win_rate']:.2%} |")

    lines.extend(
        [
            "",
            "## Per-Risk Win Rate",
            "",
            "| Risk | Wins | Total | Win Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for risk_level, item in summary["per_risk"].items():
        lines.append(f"| `{risk_level}` | {item['wins']} | {item['total']} | {item['win_rate']:.2%} |")

    lines.extend(
        [
            "",
            "## Pair Results",
            "",
            "| ID | Category | Risk | Chosen Score | Rejected Score | Win |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | `{item['category']}` | `{item['risk_level']}` | "
            f"{item['chosen_score']:.4f} | {item['rejected_score']:.4f} | {item['chosen_win']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DPO preference win rate.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default="outputs/dpo_r32_attn_mlp_v1")
    parser.add_argument("--eval_file", default="data/dpo/dpo_eval.jsonl")
    parser.add_argument("--output_json", default="outputs/dpo_r32_attn_mlp_v1/preference_eval.json")
    parser.add_argument("--output_md", default="outputs/dpo_r32_attn_mlp_v1/preference_eval.md")
    parser.add_argument("--prompt_field", default="formatted_prompt")
    parser.add_argument("--chosen_field", default="chosen")
    parser.add_argument("--rejected_field", default="rejected")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--add_eos_to_responses", type=str_to_bool, default=True)
    return parser.parse_args()


def main() -> None:
    """Run log-probability preference evaluation."""
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    rows = read_jsonl(Path(args.eval_file))
    for field in [args.prompt_field, args.chosen_field, args.rejected_field]:
        missing_ids = [row.get("id", "<missing-id>") for row in rows if field not in row]
        if missing_ids:
            raise ValueError(f"Missing field '{field}' for rows: {missing_ids}")

    tokenizer, model, device = load_model_and_tokenizer(args)
    input_device = first_parameter_device(model, device)

    results: list[dict[str, Any]] = []
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc="Scoring DPO pairs"):
        batch = rows[start : start + args.batch_size]
        chosen_encoded = [
            encode_pair(
                tokenizer,
                row[args.prompt_field],
                row[args.chosen_field],
                add_response_eos=args.add_eos_to_responses,
            )
            for row in batch
        ]
        rejected_encoded = [
            encode_pair(
                tokenizer,
                row[args.prompt_field],
                row[args.rejected_field],
                add_response_eos=args.add_eos_to_responses,
            )
            for row in batch
        ]
        batch_chosen_scores = score_batch(model, tokenizer, chosen_encoded, input_device)
        batch_rejected_scores = score_batch(model, tokenizer, rejected_encoded, input_device)

        for row, chosen_score, rejected_score in zip(
            batch,
            batch_chosen_scores,
            batch_rejected_scores,
        ):
            chosen_win = chosen_score > rejected_score
            chosen_scores.append(chosen_score)
            rejected_scores.append(rejected_score)
            results.append(
                {
                    "id": row.get("id", ""),
                    "category": row.get("category", ""),
                    "risk_level": row.get("risk_level", ""),
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "score_margin": chosen_score - rejected_score,
                    "chosen_win": chosen_win,
                }
            )

    chosen_wins = sum(1 for item in results if item["chosen_win"])
    summary = {
        "total_pairs": len(results),
        "chosen_wins": chosen_wins,
        "chosen_win_rate": chosen_wins / len(results) if results else 0.0,
        "average_chosen_score": sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0,
        "average_rejected_score": sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0.0,
        "per_category": summarize_group(results, "category"),
        "per_risk": summarize_group(results, "risk_level"),
        "batch_size": args.batch_size,
        "eval_file": args.eval_file,
        "adapter_path": args.adapter_path,
    }
    report = {"summary": summary, "results": results}
    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
