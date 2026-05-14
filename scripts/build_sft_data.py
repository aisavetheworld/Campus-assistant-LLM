"""Build SFT JSONL files from metadata-rich seed samples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from prompt_utils import build_prompt_from_sample
from prompt_utils import resolve_prompt_template


REQUIRED_FIELDS = [
    "id",
    "category",
    "task_type",
    "risk_level",
    "source_type",
    "user_language",
    "response_language",
    "output_format",
    "instruction",
    "input",
    "output",
]

ALLOWED_ENUMS = {
    "user_language": {"en", "zh", "mixed"},
    "response_language": {"en", "zh", "bilingual"},
    "output_format": {
        "plain_answer",
        "steps",
        "email_template",
        "steps_plus_email",
        "safe_escalation",
    },
}


def read_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a JSON file and require a top-level list of objects."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Every item in {path} must be a JSON object")
    return data


def validate_sample(sample: dict[str, Any], index: int) -> None:
    """Validate required SFT fields for one seed sample."""
    missing = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing:
        raise ValueError(f"Sample at index {index} is missing fields: {missing}")

    empty = [
        field
        for field in REQUIRED_FIELDS
        if not isinstance(sample[field], str) or not sample[field].strip()
    ]
    if empty:
        raise ValueError(f"Sample {sample.get('id', index)} has empty or non-string fields: {empty}")

    invalid_enums = {
        field: sample[field]
        for field, allowed_values in ALLOWED_ENUMS.items()
        if sample[field] not in allowed_values
    }
    if invalid_enums:
        raise ValueError(
            f"Sample {sample.get('id', index)} has invalid enum values: {invalid_enums}"
        )


def deduplicate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate samples based on instruction + input + output."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []

    for sample in samples:
        key = (
            sample["instruction"].strip(),
            sample["input"].strip(),
            sample["output"].strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)

    return unique


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load optional YAML config values used by data processing."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def load_chat_tokenizer(model_name_or_path: str, trust_remote_code: bool) -> Any:
    """Load tokenizer for chat-template rendering."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("prompt_template='chat' requires transformers to load the tokenizer.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(f"Tokenizer does not support apply_chat_template: {model_name_or_path}")
    return tokenizer


def process_samples(
    samples: list[dict[str, Any]],
    *,
    eos_marker: str,
    prompt_template: str,
    tokenizer: Any | None,
    model_name_or_path: str,
) -> list[dict[str, Any]]:
    """Preserve metadata and add the final SFT text field."""
    processed = []
    for sample in tqdm(samples, desc="Processing samples"):
        item = dict(sample)
        item["prompt_template"] = prompt_template
        item["text"] = build_prompt_from_sample(
            sample,
            output=sample["output"],
            eos_marker=eos_marker,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            model_name_or_path=model_name_or_path,
        )
        processed.append(item)
    return processed


def split_train_eval(
    samples: list[dict[str, Any]], eval_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle and split samples into train/eval partitions."""
    if not 0 <= eval_ratio < 1:
        raise ValueError("--eval_ratio must be in [0, 1)")

    items = list(samples)
    random.Random(seed).shuffle(items)

    if len(items) <= 1 or eval_ratio == 0:
        return items, []

    eval_count = max(1, int(round(len(items) * eval_ratio)))
    eval_count = min(eval_count, len(items) - 1)
    eval_items = items[:eval_count]
    train_items = items[eval_count:]
    return train_items, eval_items


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a UTF-8 JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def average_length(rows: list[dict[str, Any]], field: str) -> float:
    """Return average character length for a field."""
    if not rows:
        return 0.0
    return sum(len(str(row.get(field, ""))) for row in rows) / len(rows)


def print_stats(
    raw_count: int,
    processed: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    """Print useful dataset statistics."""
    stats = {
        "raw_count": raw_count,
        "deduplicated_count": len(processed),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "category_distribution": dict(Counter(row["category"] for row in processed)),
        "risk_level_distribution": dict(Counter(row["risk_level"] for row in processed)),
        "source_type_distribution": dict(Counter(row["source_type"] for row in processed)),
        "user_language_distribution": dict(Counter(row["user_language"] for row in processed)),
        "response_language_distribution": dict(
            Counter(row["response_language"] for row in processed)
        ),
        "output_format_distribution": dict(Counter(row["output_format"] for row in processed)),
        "prompt_template_distribution": dict(
            Counter(row.get("prompt_template", "legacy") for row in processed)
        ),
        "average_output_length": round(average_length(processed, "output"), 2),
        "average_text_length": round(average_length(processed, "text"), 2),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build SFT train/eval JSONL files.")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--input_file", default="data/raw/sft_seed.json")
    parser.add_argument("--train_output", default="data/processed/sft_train.jsonl")
    parser.add_argument("--eval_output", default="data/processed/sft_eval.jsonl")
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt_template",
        choices=["auto", "chat", "legacy"],
        default=None,
        help="Prompt format for the SFT text field. Defaults to config value, then auto.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Tokenizer/model name used when prompt_template resolves to chat.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=None)
    parser.add_argument(
        "--eos_marker",
        default="<|endoftext|>",
        help="Explicit marker appended after each training output.",
    )
    return parser.parse_args()


def main() -> None:
    """Build processed SFT data from raw metadata samples."""
    args = parse_args()
    config = load_yaml_config(args.config)
    config_model = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    config_data = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}

    model_name_or_path = (
        args.model_name_or_path
        or config_model.get("model_name_or_path")
        or "Qwen/Qwen2.5-1.5B-Instruct"
    )
    prompt_template_arg = args.prompt_template or config_data.get("prompt_template", "auto")
    prompt_template = resolve_prompt_template(prompt_template_arg, model_name_or_path)
    trust_remote_code = (
        bool(args.trust_remote_code)
        if args.trust_remote_code is not None
        else bool(config_model.get("trust_remote_code", True))
    )
    tokenizer = None
    eos_marker = args.eos_marker
    if prompt_template == "chat":
        tokenizer = load_chat_tokenizer(model_name_or_path, trust_remote_code)
        eos_marker = ""

    raw_path = Path(args.input_file)
    raw_samples = read_json_list(raw_path)

    for index, sample in enumerate(raw_samples):
        validate_sample(sample, index)

    unique_samples = deduplicate(raw_samples)
    processed = process_samples(
        unique_samples,
        eos_marker=eos_marker,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        model_name_or_path=model_name_or_path,
    )
    train_rows, eval_rows = split_train_eval(processed, args.eval_ratio, args.seed)

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)
    print_stats(len(raw_samples), processed, train_rows, eval_rows)


if __name__ == "__main__":
    main()
