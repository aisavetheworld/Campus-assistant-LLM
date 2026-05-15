"""Validate, audit, and preprocess DPO preference data.

This script intentionally does not train DPO. It prepares JSONL files that are
compatible with a future TRL DPOTrainer workflow.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from prompt_utils import SYSTEM_MESSAGE


REQUIRED_FIELDS = ["id", "category", "risk_level", "prompt", "chosen", "rejected"]

BAD_CHOSEN_MARKERS = [
    "Note:",
    "Explanation:",
    "Human:",
    "Assistant:",
    "### Instruction:",
    "### Response:",
]

EXTRA_NOTE_MARKERS = [
    "Note:",
    "Explanation:",
    "Additional Tips:",
    "The above draft",
    "This email",
    "Why this works",
]

OVERCONFIDENT_PATTERNS = [
    "you are definitely fine",
    "you will be approved",
    "your visa will be fine",
    "this will not affect your status",
    "you do not need to contact anyone",
    "guaranteed",
    "100% approved",
    "should be fine",
    "will be fine",
    "nothing you can do",
]

OFFICIAL_OFFICE_PHRASES = [
    "official office",
    "relevant office",
    "international student office",
    "international office",
    "international student services",
    "international student advisor",
    "visa advisor",
    "dso",
    "academic advisor",
    "academic advising",
    "department",
    "department advising",
    "registrar",
    "housing office",
    "housing team",
    "student mail",
    "mailroom",
    "mailroom team",
    "residential staff",
    "student health",
    "student health center",
    "healthcare provider",
    "doctor",
    "medical professional",
    "urgent care",
    "emergency services",
    "insurance office",
    "insurance provider",
    "immunization office",
    "official portal",
]


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


def word_count(text: str) -> int:
    """Count rough whitespace-separated words."""
    return len(text.strip().split())


def has_any(text: str, phrases: list[str]) -> bool:
    """Return whether any phrase appears in text, case-insensitively."""
    lower = text.lower()
    return any(phrase.lower() in lower for phrase in phrases)


def has_steps(text: str) -> bool:
    """Return whether text contains a simple numbered 1/2/3 step structure."""
    return all(re.search(rf"(^|\n)\s*{index}\.", text) for index in (1, 2, 3))


def has_email_format(text: str) -> bool:
    """Return whether text has the core email draft markers."""
    return (
        "subject:" in text.lower()
        and bool(re.search(r"\bDear\b", text))
        and bool(re.search(r"\b(Best regards|Sincerely|Regards)\b", text, flags=re.IGNORECASE))
    )


def pair_type(item_id: str) -> str:
    """Infer the DPO pair type from the ID prefix."""
    parts = item_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return item_id


def validate_required_fields(item: dict[str, Any], index: int) -> None:
    """Validate required DPO fields and non-empty content."""
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


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Validate all required fields and duplicate IDs."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, item in enumerate(rows):
        validate_required_fields(item, index)
        item_id = item["id"]
        if item_id in seen:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        raise ValueError(f"Duplicate DPO ids found: {sorted(set(duplicates))}")


def dpo_requirements(item: dict[str, Any]) -> list[str]:
    """Build DPO prompt requirements from pair type and risk metadata."""
    ptype = pair_type(item["id"])
    requirements = [
        "Answer as a campus administrative assistant for international students.",
        "Do not pretend to be an official office, professor, housing office, insurance office, legal advisor, or medical provider.",
        "Avoid absolute promises or guaranteed outcomes.",
        "Do not add notes, explanations, analysis, role labels, or a second draft after the answer.",
    ]

    if ptype == "dpo_steps_email":
        requirements.extend(
            [
                "Provide both numbered process steps and a complete email draft.",
                "The steps must come before the email draft.",
                'The email draft must include "Subject:", a "Dear" greeting, and a polite closing.',
            ]
        )
    elif ptype == "dpo_email_quality":
        requirements.extend(
            [
                "Only provide the email draft.",
                'The email draft must include "Subject:", a "Dear" greeting, and a polite closing.',
                "Keep the email concise and professional.",
            ]
        )
    else:
        requirements.extend(
            [
                "Use a numbered list with at least 3 steps.",
                "Mention the relevant official office or qualified professional when the issue is medium or high risk.",
            ]
        )

    if item["risk_level"] in {"medium", "high"}:
        requirements.append("For medium/high risk issues, include the relevant official office or qualified professional.")
    if item["risk_level"] == "high":
        requirements.append("For high-risk issues, avoid definitive legal, visa, medical, insurance, or disciplinary conclusions.")

    return requirements


def build_formatted_prompt(item: dict[str, Any]) -> str:
    """Build a chat-style prompt containing only system/user content."""
    requirements = "\n".join(f"- {line}" for line in dpo_requirements(item))
    user_content = (
        "Task metadata:\n"
        f"- Category: {item['category']}\n"
        f"- Risk level: {item['risk_level']}\n"
        f"- Preference type: {pair_type(item['id'])}\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Student prompt:\n{item['prompt'].strip()}"
    )
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_MESSAGE}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def rejected_flaws(item: dict[str, Any]) -> list[str]:
    """Detect common flaws intentionally present in rejected answers."""
    text = item["rejected"]
    ptype = pair_type(item["id"])
    flaws: list[str] = []
    if word_count(text) < 25:
        flaws.append("too_short")
    if item["risk_level"] in {"medium", "high"} and not has_any(text, OFFICIAL_OFFICE_PHRASES):
        flaws.append("no_official_office")
    if has_any(text, OVERCONFIDENT_PATTERNS):
        flaws.append("overconfident_claim")
    if ptype != "dpo_email_quality" and not has_steps(text):
        flaws.append("missing_steps")
    if ptype in {"dpo_steps_email", "dpo_email_quality"} and not has_email_format(text):
        flaws.append("poor_email_format")
    if has_any(text, EXTRA_NOTE_MARKERS):
        flaws.append("extra_notes")
    return flaws


def quality_record(item: dict[str, Any]) -> dict[str, Any]:
    """Return quality diagnostics for one DPO pair."""
    chosen_wc = word_count(item["chosen"])
    rejected_wc = word_count(item["rejected"])
    chosen_has_office = has_any(item["chosen"], OFFICIAL_OFFICE_PHRASES)
    chosen_bad_markers = [
        marker for marker in BAD_CHOSEN_MARKERS if marker.lower() in item["chosen"].lower()
    ]
    office_required = item["risk_level"] in {"medium", "high"}
    return {
        "id": item["id"],
        "pair_type": pair_type(item["id"]),
        "category": item["category"],
        "risk_level": item["risk_level"],
        "chosen_word_count": chosen_wc,
        "rejected_word_count": rejected_wc,
        "chosen_rejected_length_ratio": round(chosen_wc / max(rejected_wc, 1), 2),
        "chosen_has_official_office": chosen_has_office,
        "chosen_missing_official_office_when_required": office_required and not chosen_has_office,
        "chosen_bad_markers": chosen_bad_markers,
        "rejected_flaws": rejected_flaws(item),
    }


def processed_row(item: dict[str, Any]) -> dict[str, Any]:
    """Build one processed DPO JSONL row."""
    return {
        "id": item["id"],
        "category": item["category"],
        "risk_level": item["risk_level"],
        "prompt": item["prompt"].strip(),
        "chosen": item["chosen"].strip(),
        "rejected": item["rejected"].strip(),
        "formatted_prompt": build_formatted_prompt(item),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def average(values: list[float]) -> float:
    """Return the average of a list, or 0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate DPO data statistics."""
    flaw_counter: Counter[str] = Counter()
    for item in quality_rows:
        flaw_counter.update(item["rejected_flaws"])

    missing_office_ids = [
        item["id"] for item in quality_rows if item["chosen_missing_official_office_when_required"]
    ]
    chosen_bad_marker_ids = [
        item["id"] for item in quality_rows if item["chosen_bad_markers"]
    ]

    return {
        "pair_count": len(rows),
        "category_distribution": dict(Counter(row["category"] for row in rows)),
        "risk_level_distribution": dict(Counter(row["risk_level"] for row in rows)),
        "prefix_distribution": dict(Counter(pair_type(row["id"]) for row in rows)),
        "average_chosen_word_count": round(
            average([item["chosen_word_count"] for item in quality_rows]), 2
        ),
        "average_rejected_word_count": round(
            average([item["rejected_word_count"] for item in quality_rows]), 2
        ),
        "average_chosen_rejected_length_ratio": round(
            average([item["chosen_rejected_length_ratio"] for item in quality_rows]), 2
        ),
        "rejected_flaw_counts": dict(flaw_counter.most_common()),
        "chosen_missing_official_office_when_required_count": len(missing_office_ids),
        "chosen_missing_official_office_when_required_ids": missing_office_ids,
        "chosen_bad_marker_count": len(chosen_bad_marker_ids),
        "chosen_bad_marker_ids": chosen_bad_marker_ids,
    }


def split_rows(
    rows: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle and split rows into train/eval sets."""
    if not 0 < eval_ratio < 1:
        raise ValueError("--eval_ratio must be between 0 and 1")
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def markdown_table(distribution: dict[str, int]) -> list[str]:
    """Render a distribution dictionary as Markdown table rows."""
    return [f"| `{key}` | {value} |" for key, value in sorted(distribution.items())]


def write_audit_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a DPO data audit Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# DPO Data Audit",
        "",
        "## Summary",
        "",
        f"- DPO seed count: {summary['pair_count']}",
        f"- Train split: {report['train_count']}",
        f"- Eval split: {report['eval_count']}",
        f"- Eval ratio: {report['eval_ratio']}",
        f"- Seed: {report['seed']}",
        "",
        "## Category Distribution",
        "",
        "| Category | Count |",
        "|---|---:|",
        *markdown_table(summary["category_distribution"]),
        "",
        "## Risk Distribution",
        "",
        "| Risk Level | Count |",
        "|---|---:|",
        *markdown_table(summary["risk_level_distribution"]),
        "",
        "## Prefix Distribution",
        "",
        "| Prefix | Count |",
        "|---|---:|",
        *markdown_table(summary["prefix_distribution"]),
        "",
        "## Quality Statistics",
        "",
        f"- Average chosen word count: {summary['average_chosen_word_count']}",
        f"- Average rejected word count: {summary['average_rejected_word_count']}",
        f"- Average chosen/rejected length ratio: {summary['average_chosen_rejected_length_ratio']}",
        f"- Chosen missing official office when required: {summary['chosen_missing_official_office_when_required_count']}",
        f"- Chosen bad marker count: {summary['chosen_bad_marker_count']}",
        "",
        "## Common Chosen Patterns",
        "",
        "- Numbered steps for process guidance and safe escalation.",
        "- Official office or qualified professional referral for medium/high-risk issues.",
        "- No absolute guarantees for visa, medical, insurance, housing-contract, or academic-risk scenarios.",
        "- Email drafts include `Subject:`, `Dear`, and a polite closing.",
        "- `steps_plus_email` pairs include both process steps and an email draft.",
        "",
        "## Common Rejected Flaws",
        "",
        "| Flaw | Count |",
        "|---|---:|",
        *markdown_table(summary["rejected_flaw_counts"]),
        "",
        "## Known Limitations",
        "",
        "- This is still seed preference data, not a production-scale preference dataset.",
        "- Some rejected answers are intentionally obvious to make the first DPO pipeline easy to validate.",
        "- The next expansion should use observed SFT-only failures rather than more template duplication.",
        "- The audit does not replace human review of preference quality.",
        "- DPO training is intentionally not implemented yet.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate and preprocess DPO data.")
    parser.add_argument("--input_file", default="data/dpo/dpo_seed.json")
    parser.add_argument("--train_output", default="data/dpo/dpo_train.jsonl")
    parser.add_argument("--eval_output", default="data/dpo/dpo_eval.jsonl")
    parser.add_argument("--audit_md", default="docs/experiments/dpo_data_audit.md")
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Validate, audit, split, and export DPO seed pairs."""
    args = parse_args()
    rows = read_json_list(Path(args.input_file))
    validate_rows(rows)

    quality_rows = [quality_record(row) for row in rows]
    processed = [processed_row(row) for row in rows]
    train_rows, eval_rows = split_rows(processed, args.eval_ratio, args.seed)

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)

    report = {
        "input_file": args.input_file,
        "train_output": args.train_output,
        "eval_output": args.eval_output,
        "audit_md": args.audit_md,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "summary": summarize(rows, quality_rows),
        "quality": quality_rows,
        "note": "DPO training is intentionally not implemented in Project 1.",
    }
    write_audit_markdown(Path(args.audit_md), report)
    print(json.dumps(report["summary"] | {
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_output": args.train_output,
        "eval_output": args.eval_output,
        "audit_md": args.audit_md,
        "note": report["note"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
