"""Audit SFT data for prompt leakage and email-format contamination."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from prompt_utils import EXTRA_NOTE_MARKERS


EMAIL_OUTPUT_FORMATS = {"email_template", "steps_plus_email"}


def read_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                rows.append(row)
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSON or JSONL rows based on file suffix."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return read_json_list(path)


def find_pollution_markers(text: str) -> list[str]:
    """Find contamination markers in an output field."""
    lower = text.lower()
    return [marker for marker in EXTRA_NOTE_MARKERS if marker.lower() in lower]


def has_email_closing(text: str) -> bool:
    """Check that an email ends with a normal formal closing."""
    return bool(
        re.search(
            r"(Best regards|Sincerely|Regards),\s*\n\s*\[[^\]]*Name[^\]]*\]\s*$",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def audit_row(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Return all audit violations for one row."""
    violations = []
    row_id = row.get("id", f"line_{row.get('_line_number', 'unknown')}")
    output = str(row.get("output", ""))
    output_format = row.get("output_format", "")

    markers = find_pollution_markers(output)
    if markers:
        violations.append(
            {
                "source": source,
                "id": row_id,
                "type": "pollution_marker",
                "markers": markers,
            }
        )

    if output_format in EMAIL_OUTPUT_FORMATS:
        email_errors = []
        stripped = output.strip()
        if not stripped.startswith("Subject:"):
            email_errors.append("missing_subject_start")
        if not re.search(r"\bDear\b", stripped):
            email_errors.append("missing_dear_greeting")
        if not has_email_closing(stripped):
            email_errors.append("missing_terminal_closing")
        if email_errors:
            violations.append(
                {
                    "source": source,
                    "id": row_id,
                    "type": "email_format",
                    "errors": email_errors,
                }
            )

    return violations


def audit_file(path: Path) -> dict[str, Any]:
    """Audit one JSON/JSONL file."""
    rows = read_rows(path)
    violations = []
    for row in rows:
        violations.extend(audit_row(row, str(path)))
    return {
        "path": str(path),
        "row_count": len(rows),
        "violation_count": len(violations),
        "violations": violations,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit SFT data for output contamination.")
    parser.add_argument("--raw_file", default="data/raw/sft_seed.json")
    parser.add_argument("--train_file", default="data/processed/sft_train.jsonl")
    parser.add_argument("--eval_file", default="data/processed/sft_eval.jsonl")
    parser.add_argument(
        "--no_fail",
        action="store_true",
        help="Print violations but exit with code 0.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the audit."""
    args = parse_args()
    reports = [
        audit_file(Path(args.raw_file)),
        audit_file(Path(args.train_file)),
        audit_file(Path(args.eval_file)),
    ]
    total_violations = sum(report["violation_count"] for report in reports)
    summary = {
        "total_violations": total_violations,
        "files": reports,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if total_violations and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
