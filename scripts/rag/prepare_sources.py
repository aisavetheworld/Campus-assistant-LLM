"""Validate source metadata and raw text files.

Usage:
    python scripts/rag/prepare_sources.py \
        --sources_meta data/rag/ucsd_sources.json \
        --raw_dir data/rag/raw_sources \
        --output_dir outputs/rag_eval
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path


REQUIRED_FIELDS = ["source_id", "title", "url", "category", "source_type",
                   "date_collected", "usable_for", "notes"]
VALID_CATEGORIES = {"housing", "course_enrollment", "health_insurance", "international_students"}
NAV_WORDS = {"home", "menu", "navigation", "skip", "search", "login", "sign in",
             "contact us", "about us", "sitemap", "cookie", "privacy policy", "terms of use"}
MIN_CONTENT_WORDS = 50


def looks_like_nav_only(text: str) -> bool:
    words = text.lower().split()
    if len(words) < MIN_CONTENT_WORDS:
        return True
    nav_hits = sum(1 for w in words if w in NAV_WORDS)
    return nav_hits / len(words) > 0.3


def validate_sources(sources_meta: Path, raw_dir: Path) -> dict:
    sources = json.loads(sources_meta.read_text())
    results = []
    summary = {"total": len(sources), "ok": 0, "warnings": 0, "errors": 0,
                "missing_txt": [], "empty_txt": [], "nav_only": [], "field_errors": []}

    seen_ids = {}
    for entry in sources:
        sid = entry.get("source_id", "MISSING_ID")
        issues = []
        warnings = []

        # Duplicate check
        if sid in seen_ids:
            issues.append(f"Duplicate source_id (also at index {seen_ids[sid]})")
        seen_ids[sid] = sources.index(entry)

        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in entry:
                issues.append(f"Missing field: {field}")
            elif field == "usable_for" and not isinstance(entry[field], list):
                issues.append("usable_for must be a list")

        # Category check
        cat = entry.get("category", "")
        if cat and cat not in VALID_CATEGORIES:
            warnings.append(f"Unknown category: {cat}")

        # URL check
        if not entry.get("url"):
            warnings.append("url is empty")

        # date_collected check
        dc = entry.get("date_collected", "")
        if dc:
            try:
                date.fromisoformat(dc)
            except ValueError:
                issues.append(f"date_collected not ISO format: {dc}")
        else:
            warnings.append("date_collected is empty")

        # Raw text file check
        txt_path = raw_dir / f"{sid}.txt"
        if not txt_path.exists():
            issues.append(f"No raw text file: {txt_path}")
            summary["missing_txt"].append(sid)
        else:
            text = txt_path.read_text(encoding="utf-8").strip()
            word_count = len(text.split())
            if word_count == 0:
                issues.append("Raw text file is empty")
                summary["empty_txt"].append(sid)
            elif word_count < MIN_CONTENT_WORDS:
                warnings.append(f"Raw text very short ({word_count} words) — may be nav/login page")
                summary["nav_only"].append(sid)
            elif looks_like_nav_only(text):
                warnings.append("Raw text looks like navigation/footer only")
                summary["nav_only"].append(sid)
            else:
                word_count_ok = word_count

        status = "ERROR" if issues else ("WARNING" if warnings else "OK")
        if status == "OK":
            summary["ok"] += 1
        elif status == "WARNING":
            summary["warnings"] += 1
        else:
            summary["errors"] += 1
            if issues:
                summary["field_errors"].append(sid)

        results.append({
            "source_id": sid,
            "title": entry.get("title", ""),
            "category": entry.get("category", ""),
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "word_count": len((raw_dir / f"{sid}.txt").read_text(encoding="utf-8").split())
                          if (raw_dir / f"{sid}.txt").exists() else 0,
        })

    return {"summary": summary, "results": results}


def write_markdown_report(report: dict, out_path: Path) -> None:
    s = report["summary"]
    lines = [
        "# Source Validation Report",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|---|---|",
        f"| Total sources | {s['total']} |",
        f"| OK | {s['ok']} |",
        f"| Warnings | {s['warnings']} |",
        f"| Errors | {s['errors']} |",
        f"| Missing .txt | {len(s['missing_txt'])} |",
        f"| Empty .txt | {len(s['empty_txt'])} |",
        f"| Likely nav/login only | {len(s['nav_only'])} |",
        "",
        "## Per-Source Results",
        "",
        "| source_id | category | status | words | issues / warnings |",
        "|---|---|---|---|---|",
    ]
    for r in report["results"]:
        notes = "; ".join(r["issues"] + r["warnings"]) or "—"
        lines.append(f"| {r['source_id']} | {r['category']} | {r['status']} "
                     f"| {r['word_count']} | {notes} |")

    if s["missing_txt"]:
        lines += ["", "## Missing Raw Text Files", ""]
        for sid in s["missing_txt"]:
            lines.append(f"- {sid}")

    if s["nav_only"]:
        lines += ["", "## Likely Navigation / Login Pages (low content)", ""]
        for sid in s["nav_only"]:
            lines.append(f"- {sid}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources_meta", default="data/rag/ucsd_sources.json")
    parser.add_argument("--raw_dir", default="data/rag/raw_sources")
    parser.add_argument("--output_dir", default="outputs/rag_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Validating sources...")
    report = validate_sources(Path(args.sources_meta), Path(args.raw_dir))
    s = report["summary"]
    print(f"  Total: {s['total']} | OK: {s['ok']} | Warnings: {s['warnings']} | Errors: {s['errors']}")

    md_path = out_dir / "source_validation_report.md"
    json_path = out_dir / "source_validation_report.json"
    write_markdown_report(report, md_path)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"  Report → {md_path}")
    print(f"  Report → {json_path}")

    if s["missing_txt"]:
        print(f"\n  WARNING: {len(s['missing_txt'])} sources missing raw text: {s['missing_txt']}")
    if s["nav_only"]:
        print(f"  WARNING: {len(s['nav_only'])} sources look like nav/login pages: {s['nav_only']}")


if __name__ == "__main__":
    main()
