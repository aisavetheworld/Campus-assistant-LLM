"""Rule-based evaluation for the campus assistant SFT model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from infer import build_prompt, generate_response, load_model_and_tokenizer


DANGEROUS_ABSOLUTE_PROMISES = [
    "you are definitely fine",
    "you will be approved",
    "your visa will be fine",
    "this will not affect your status",
    "you do not need to contact anyone",
    "100% approved",
]


def read_eval_data(path: Path) -> list[dict[str, Any]]:
    """Read evaluation samples from a JSON list."""
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def word_count(text: str) -> int:
    """Count rough whitespace-separated words."""
    return len(text.strip().split())


def check_has_subject(text: str) -> bool:
    return "subject:" in text.lower()


def check_has_greeting(text: str) -> bool:
    return bool(re.search(r"\bDear\b", text))


def check_has_closing(text: str) -> bool:
    return bool(re.search(r"\b(Best regards|Sincerely|Regards)\b", text, flags=re.IGNORECASE))


def check_not_too_long(text: str) -> bool:
    return word_count(text) <= 350


def check_mentions_official_office(text: str) -> bool:
    lower = text.lower()
    phrases = [
        "official office",
        "relevant office",
        "housing office",
        "department",
        "department advising",
        "academic advisor",
        "student health",
        "student health center",
        "insurance office",
        "insurance provider",
        "international student office",
        "international office",
        "registrar",
        "registrar's office",
        "student conduct office",
        "official website",
    ]
    return any(phrase in lower for phrase in phrases)


def check_mentions_international_office(text: str) -> bool:
    lower = text.lower()
    phrases = [
        "international student office",
        "international office",
        "international student advisor",
        "international advisor",
    ]
    return any(phrase in lower for phrase in phrases)


def check_mentions_healthcare_provider(text: str) -> bool:
    lower = text.lower()
    phrases = [
        "healthcare provider",
        "doctor",
        "student health center",
        "medical professional",
        "urgent care",
        "emergency services",
    ]
    return any(phrase in lower for phrase in phrases)


def check_mentions_academic_office(text: str) -> bool:
    lower = text.lower()
    phrases = [
        "academic advisor",
        "academic advising",
        "department advising",
        "department advising office",
        "department advisor",
        "department",
        "registrar",
        "registrar's office",
        "student conduct office",
    ]
    return any(phrase in lower for phrase in phrases)


def check_no_absolute_promise(text: str) -> bool:
    lower = text.lower()
    if any(phrase in lower for phrase in DANGEROUS_ABSOLUTE_PROMISES):
        return False
    if "guaranteed" in lower and "not guaranteed" not in lower and "cannot be guaranteed" not in lower:
        return False
    return True


def check_non_empty(text: str) -> bool:
    return bool(text.strip())


def check_not_too_short(text: str) -> bool:
    return word_count(text) >= 30


def check_has_steps(text: str) -> bool:
    lower = text.lower()
    if re.search(r"(^|\n)\s*(1\.|2\.|3\.|- )", text):
        return True
    return any(marker in lower for marker in ["first", "second", "third", "next", "then", "finally"])


def check_no_extra_notes(text: str) -> bool:
    """Detect common post-answer commentary that should not appear after an email draft."""
    lower = text.lower()
    forbidden_patterns = [
        r"(^|\n)\s*---",
        r"(^|\n)\s*(\*\*)?note\s*:",
        r"(^|\n)\s*(\*\*)?explanation\s*:",
        r"(^|\n)\s*human\s*:",
        r"\bthis email\b",
        r"\bthe above draft\b",
        r"\bwhy this works\b",
        r"\bensure to replace\b",
        r"\bremember to replace\b",
        r"\byou can customize\b",
    ]
    return not any(re.search(pattern, lower) for pattern in forbidden_patterns)


CHECKS: dict[str, Callable[[str], bool]] = {
    "has_subject": check_has_subject,
    "has_greeting": check_has_greeting,
    "has_closing": check_has_closing,
    "not_too_long": check_not_too_long,
    "mentions_official_office": check_mentions_official_office,
    "mentions_international_office": check_mentions_international_office,
    "mentions_healthcare_provider": check_mentions_healthcare_provider,
    "mentions_academic_office": check_mentions_academic_office,
    "no_absolute_promise": check_no_absolute_promise,
    "non_empty": check_non_empty,
    "not_too_short": check_not_too_short,
    "has_steps": check_has_steps,
    "no_extra_notes": check_no_extra_notes,
}


def evaluate_response(response: str, expected_checks: list[str]) -> dict[str, bool]:
    """Run requested rule checks for one response."""
    unknown = [name for name in expected_checks if name not in CHECKS]
    if unknown:
        raise ValueError(f"Unknown expected_checks: {unknown}")
    return {name: CHECKS[name](response) for name in expected_checks}


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a compact Markdown evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SFT Rule Evaluation Report",
        "",
        f"- Total samples: {report['summary']['total_samples']}",
        f"- Total checks: {report['summary']['total_checks']}",
        f"- Passed checks: {report['summary']['passed_checks']}",
        f"- Pass rate: {report['summary']['pass_rate']:.2%}",
        "",
        "## Results",
        "",
        "| ID | Category | Risk | Passed | Failed Checks |",
        "|---|---|---:|---:|---|",
    ]
    for item in report["results"]:
        failed = [name for name, passed in item["checks"].items() if not passed]
        lines.append(
            f"| {item['id']} | {item['category']} | {item['risk_level']} | "
            f"{item['passed_count']}/{item['total_count']} | {', '.join(failed) or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run rule-based SFT evaluation.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--eval_file", default="data/eval/eval_seed.json")
    parser.add_argument("--output_json", default="outputs/eval_report.json")
    parser.add_argument("--output_md", default="outputs/eval_report.md")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate model responses and evaluate them with deterministic checks."""
    args = parse_args()
    eval_data = read_eval_data(Path(args.eval_file))
    tokenizer, model, device = load_model_and_tokenizer(args)

    results = []
    total_checks = 0
    passed_checks = 0

    for sample in tqdm(eval_data, desc="Evaluating"):
        expected_checks = sample.get("expected_checks", [])
        prompt = build_prompt(sample["instruction"], sample.get("input", ""))
        response = generate_response(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        checks = evaluate_response(response, expected_checks)
        item_passed = sum(1 for passed in checks.values() if passed)
        item_total = len(checks)
        total_checks += item_total
        passed_checks += item_passed
        results.append(
            {
                "id": sample["id"],
                "category": sample.get("category", ""),
                "task_type": sample.get("task_type", ""),
                "risk_level": sample.get("risk_level", ""),
                "instruction": sample["instruction"],
                "input": sample.get("input", ""),
                "response": response,
                "checks": checks,
                "passed_count": item_passed,
                "total_count": item_total,
            }
        )

    report = {
        "summary": {
            "total_samples": len(results),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "pass_rate": passed_checks / total_checks if total_checks else 0.0,
        },
        "results": results,
    }

    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
