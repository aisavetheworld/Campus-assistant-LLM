"""Rule-based evaluation for the campus assistant SFT model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
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
        "housing team",
        "relevant housing team",
        "student mail",
        "mailroom team",
        "mailroom",
        "department",
        "department advising",
        "academic advisor",
        "student health",
        "student health center",
        "immunization office",
        "official portal",
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
        "international student services",
        "international student advisor",
        "international advisor",
        "visa advisor",
        "dso",
    ]
    return any(phrase in lower for phrase in phrases)


def check_mentions_healthcare_provider(text: str) -> bool:
    lower = text.lower()
    phrases = [
        "healthcare provider",
        "doctor",
        "student health",
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


def check_min_word_count_by_format(text: str) -> bool:
    return word_count(text) >= 60


def check_has_steps(text: str) -> bool:
    lower = text.lower()
    if all(re.search(rf"(^|\n)\s*{index}\.", text) for index in (1, 2, 3, 4)):
        return True
    return all(marker in lower for marker in ["first", "second", "third"])


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
        r"\badditional tips\b",
        r"\bemail draft\b",
        r"\bplease replace\b",
        r"(^|\n)\s*###\s*output\s*:",
        r"(^|\n)\s*###\s*response\s*:",
        r"(^|\n)\s*assistant\s*:",
        r"(^|\n)\s*user\s*:",
    ]
    return not any(re.search(pattern, lower) for pattern in forbidden_patterns)


def check_no_prompt_leakage(text: str) -> bool:
    """Detect prompt or role-label leakage in the model response."""
    leakage_markers = [
        "Human:",
        "Assistant:",
        "User:",
        "### Instruction:",
        "### Response:",
        "### Output:",
        "### Refine:",
    ]
    return not any(marker.lower() in text.lower() for marker in leakage_markers)


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
    "min_word_count_by_format": check_min_word_count_by_format,
    "has_steps": check_has_steps,
    "no_extra_notes": check_no_extra_notes,
    "no_prompt_leakage": check_no_prompt_leakage,
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
        f"- Raw generation prompt leakage count: {report['summary'].get('raw_generation_prompt_leakage_count', 0)}",
        f"- Raw generation prompt leakage IDs: {', '.join(report['summary'].get('raw_generation_prompt_leakage_ids', [])) or '-'}",
        f"- Final response prompt leakage count: {report['summary'].get('final_response_prompt_leakage_count', 0)}",
        f"- Final response prompt leakage IDs: {', '.join(report['summary'].get('final_response_prompt_leakage_ids', [])) or '-'}",
        f"- Truncated count: {report['summary'].get('truncated_count', 0)}",
        f"- Early truncation count: {report['summary'].get('early_truncation_count', 0)}",
        f"- Early truncation IDs: {', '.join(report['summary'].get('early_truncation_ids', [])) or '-'}",
        f"- Late truncation count: {report['summary'].get('late_truncation_count', 0)}",
        f"- Not-too-short failures caused by stop truncation: {report['summary'].get('not_too_short_truncated_count', 0)}",
        f"- Not-too-short failures without stop truncation: {report['summary'].get('not_too_short_untruncated_count', 0)}",
        "",
        "## Failed Check Counts",
        "",
    ]
    for check_name, count in report["summary"].get("failed_check_counts", {}).items():
        lines.append(f"- `{check_name}`: {count}")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| ID | Category | Risk | Passed | Response Words | Raw Words | Truncated | Early | Stop Sequence | Failed Checks |",
            "|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for item in report["results"]:
        failed = [name for name, passed in item["checks"].items() if not passed]
        stop = item.get("stop_sequence_used") or "-"
        if stop != "-":
            stop = stop.replace("\n", "\\n")
        lines.append(
            f"| {item['id']} | {item['category']} | {item['risk_level']} | "
            f"{item['passed_count']}/{item['total_count']} | {item.get('response_word_count', 0)} | "
            f"{item.get('raw_response_word_count', 0)} | {item.get('was_truncated_by_stop_sequence', False)} | "
            f"{item.get('is_early_truncation', False)} | `{stop}` | {', '.join(failed) or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_failed_checks(results: list[dict[str, Any]]) -> dict[str, int]:
    """Count failed checks across all evaluated samples."""
    counter: Counter[str] = Counter()
    for item in results:
        for check_name, passed in item["checks"].items():
            if not passed:
                counter[check_name] += 1
    return dict(counter.most_common())


def summarize_short_failures(results: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Split not_too_short failures by whether stop truncation happened."""
    truncated = []
    untruncated = []
    for item in results:
        if item["checks"].get("not_too_short") is not False:
            continue
        if item.get("was_truncated_by_stop_sequence"):
            truncated.append(item["id"])
        else:
            untruncated.append(item["id"])
    return truncated, untruncated


def summarize_truncation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize stop-sequence truncation, including early truncation IDs."""
    truncated_ids = [
        item["id"] for item in results if item.get("was_truncated_by_stop_sequence")
    ]
    early_ids = [
        item["id"]
        for item in results
        if item.get("was_truncated_by_stop_sequence")
        and item.get("response_word_count", 0) < 40
    ]
    late_ids = [item_id for item_id in truncated_ids if item_id not in set(early_ids)]
    return {
        "truncated_count": len(truncated_ids),
        "truncated_ids": truncated_ids,
        "early_truncation_count": len(early_ids),
        "early_truncation_ids": early_ids,
        "late_truncation_count": len(late_ids),
        "late_truncation_ids": late_ids,
    }


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
    parser.add_argument("--prompt_template", choices=["auto", "chat", "legacy"], default="auto")
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
    prompt_leakage_count = 0
    prompt_leakage_ids = []
    raw_prompt_leakage_count = 0
    raw_prompt_leakage_ids = []

    for sample in tqdm(eval_data, desc="Evaluating"):
        expected_checks = sample.get("expected_checks", [])
        prompt = build_prompt(
            sample["instruction"],
            sample.get("input", ""),
            category=sample.get("category", "general"),
            risk_level=sample.get("risk_level", "low"),
            output_format=sample.get("output_format", "plain_answer"),
            user_language=sample.get("user_language", "mixed"),
            response_language=sample.get("response_language", "en"),
            prompt_template=args.prompt_template,
            tokenizer=tokenizer,
            model_name_or_path=args.model_name_or_path,
        )
        response, generation_info = generate_response(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            return_generation_info=True,
        )
        checks = evaluate_response(response, expected_checks)
        prompt_leakage = not check_no_prompt_leakage(response)
        if prompt_leakage:
            prompt_leakage_count += 1
            prompt_leakage_ids.append(sample["id"])
        raw_prompt_leakage = bool(generation_info.get("raw_prompt_leakage_detected"))
        if raw_prompt_leakage:
            raw_prompt_leakage_count += 1
            raw_prompt_leakage_ids.append(sample["id"])
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
                "raw_response": generation_info.get("raw_response", response),
                "response_length": len(response),
                "response_word_count": word_count(response),
                "raw_response_word_count": generation_info.get(
                    "raw_response_word_count", word_count(generation_info.get("raw_response", response))
                ),
                "was_truncated_by_stop_sequence": generation_info[
                    "was_truncated_by_stop_sequence"
                ],
                "stop_sequence_used": generation_info["stop_sequence_used"],
                "is_prompt_leakage_stop": generation_info.get("is_prompt_leakage_stop", False),
                "is_extra_note_stop": generation_info.get("is_extra_note_stop", False),
                "is_early_truncation": generation_info.get("is_early_truncation", False),
                "raw_response_length": generation_info["raw_response_length"],
                "raw_prompt_leakage": raw_prompt_leakage,
                "checks": checks,
                "prompt_leakage": prompt_leakage,
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
            "prompt_leakage_count": prompt_leakage_count,
            "prompt_leakage_ids": prompt_leakage_ids,
            "failed_check_counts": summarize_failed_checks(results),
        },
        "results": results,
    }
    short_truncated_ids, short_untruncated_ids = summarize_short_failures(results)
    truncation_summary = summarize_truncation(results)
    report["summary"].update(
        {
            **truncation_summary,
            "raw_generation_prompt_leakage_count": raw_prompt_leakage_count,
            "raw_generation_prompt_leakage_ids": raw_prompt_leakage_ids,
            "final_response_prompt_leakage_count": prompt_leakage_count,
            "final_response_prompt_leakage_ids": prompt_leakage_ids,
            "raw_prompt_leakage_count": raw_prompt_leakage_count,
            "raw_prompt_leakage_ids": raw_prompt_leakage_ids,
            "postprocessed_prompt_leakage_count": prompt_leakage_count,
            "postprocessed_prompt_leakage_ids": prompt_leakage_ids,
            "not_too_short_truncated_count": len(short_truncated_ids),
            "not_too_short_truncated_ids": short_truncated_ids,
            "not_too_short_untruncated_count": len(short_untruncated_ids),
            "not_too_short_untruncated_ids": short_untruncated_ids,
        }
    )

    write_json(Path(args.output_json), report)
    write_markdown(Path(args.output_md), report)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
