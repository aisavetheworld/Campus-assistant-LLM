"""Validate and summarize placeholder DPO preference data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ["id", "prompt", "chosen", "rejected"]


def read_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list from disk."""
    if not path.exists():
        raise FileNotFoundError(f"DPO seed file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every DPO item must be a JSON object")
    return data


def validate_item(item: dict[str, Any], index: int) -> None:
    """Validate required DPO fields."""
    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        raise ValueError(f"DPO item at index {index} is missing fields: {missing}")
    empty = [
        field
        for field in REQUIRED_FIELDS
        if not isinstance(item[field], str) or not item[field].strip()
    ]
    if empty:
        raise ValueError(f"DPO item {item.get('id', index)} has empty fields: {empty}")
    if item["chosen"].strip() == item["rejected"].strip():
        raise ValueError(f"DPO item {item['id']} has identical chosen and rejected responses")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write DPO rows to JSONL for future training scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def average_length(rows: list[dict[str, Any]], field: str) -> float:
    """Return average character length for one field."""
    if not rows:
        return 0.0
    return sum(len(row[field]) for row in rows) / len(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate placeholder DPO data.")
    parser.add_argument("--input_file", default="data/dpo/dpo_seed.json")
    parser.add_argument(
        "--output_jsonl",
        default="",
        help="Optional future-use export path, e.g. data/dpo/dpo_train.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    """Validate DPO seed pairs and optionally export JSONL."""
    args = parse_args()
    rows = read_json_list(Path(args.input_file))
    for index, item in enumerate(rows):
        validate_item(item, index)

    stats = {
        "pair_count": len(rows),
        "average_prompt_length": round(average_length(rows, "prompt"), 2),
        "average_chosen_length": round(average_length(rows, "chosen"), 2),
        "average_rejected_length": round(average_length(rows, "rejected"), 2),
        "note": "DPO training is intentionally not implemented in Project 1.",
    }

    if args.output_jsonl:
        write_jsonl(Path(args.output_jsonl), rows)
        stats["exported_jsonl"] = args.output_jsonl

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
