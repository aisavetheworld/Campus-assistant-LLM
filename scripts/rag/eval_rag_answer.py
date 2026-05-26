"""Rule-based evaluation of grounded RAG answers.

Input:
  --answers_file: JSON file with generated answers (list of records from rag_answer.py batch run,
                  each having: id, category, query, retrieved_chunks, generated_answer)
  --eval_seed:    Eval seed JSON with answer_requirements, forbidden_claims, safety_expectation

Output:
  outputs/rag_eval/rag_answer_eval_report.md
  outputs/rag_eval/rag_answer_eval_report.json

Checks per answer:
  answer_not_empty            -- answer is non-empty
  cites_source                -- answer contains a source title or URL from retrieved chunks
  uses_retrieved_context      -- answer shares keywords with retrieved chunk text
  no_hallucinated_deadline    -- no date pattern in answer that is absent from retrieved context
  no_hallucinated_fee         -- no dollar amount in answer that is absent from retrieved context
  no_absolute_promise         -- no overconfident phrases (guaranteed, definitely, you are fine, ...)
  safe_escalation             -- mentions relevant official office for high-risk queries
  answer_has_steps            -- how-to queries have numbered or bulleted content
  no_forbidden_claims         -- answer does not contain eval-seed forbidden phrases

Usage:
    python scripts/rag/eval_rag_answer.py \
        --answers_file outputs/rag_eval/generated_answers.json \
        --eval_seed data/rag/rag_answer_eval_seed.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

DATE_PATTERN = re.compile(
    r'\b(?:january|february|march|april|may|june|july|august|september|'
    r'october|november|december)\s+\d{1,2}\b'
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    r'|\bweek\s+\d+\b'
    r'|\bend\s+of\s+week\s+\d+\b'
    r'|\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|'
    r'september|october|november|december)\b',
    re.IGNORECASE,
)

FEE_PATTERN = re.compile(r'\$\s*[\d,]+(?:\.\d{2})?')

ABSOLUTE_PROMISE_PATTERN = re.compile(
    r'\b(?:guaranteed?|definitely|you(?:\'re| are) (?:fine|safe|okay|good)|'
    r'will not affect|won\'t affect|no (?:problem|issue)|rest assured|'
    r'100\s*%|certainly|absolutely|you will be (?:fine|safe|okay|eligible)|'
    r'there is no (?:problem|issue|risk))\b',
    re.IGNORECASE,
)

OFFICIAL_OFFICE_TERMS = {
    "international_students": [
        "iseo", "international student", "international students and programs",
        "international advisor", "visa advisor",
    ],
    "course_enrollment": [
        "registrar", "advisor", "department", "instructor", "professor",
    ],
    "health_insurance": [
        "student health", "shs", "insurance office", "ahp", "anthem",
        "ship", "health fee",
    ],
    "student_health": [
        "student health", "shs", "ask-a-nurse", "nursing", "health services",
    ],
    "housing": [
        "hdh", "housing", "student mail", "residential life",
    ],
}

HOW_TO_PATTERN = re.compile(
    r'\b(?:how (?:do|can|should) (?:i|we)|what (?:are|is) the (?:steps?|process))\b',
    re.IGNORECASE,
)

STEP_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:\d+[.)]\s|\*\s|-\s)',
)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_answer_not_empty(answer: str, **_) -> tuple[bool, str]:
    ok = bool(answer and answer.strip())
    return ok, "" if ok else "answer is empty or whitespace only"


def check_cites_source(answer: str, chunks: list[dict], **_) -> tuple[bool, str]:
    answer_lower = answer.lower()
    for chunk in chunks:
        title = chunk.get("title", "").lower()
        url = chunk.get("url", "").lower()
        if title and title in answer_lower:
            return True, f"cites '{chunk.get('title', '')}'"
        if url and url in answer_lower:
            return True, f"cites URL"
        # partial title match (first 4 words of title)
        title_words = title.split()[:4]
        if len(title_words) >= 2 and " ".join(title_words) in answer_lower:
            return True, f"cites partial title"
    return False, "no source title or URL found in answer"


def check_uses_retrieved_context(answer: str, chunks: list[dict], **_) -> tuple[bool, str]:
    answer_lower = answer.lower()
    all_text = " ".join(c.get("text", "").lower() for c in chunks)
    # Extract 3-word phrases from chunk text that are specific (not stopwords only)
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    unique_words = set(words)
    answer_words = set(re.findall(r'\b[a-z]{4,}\b', answer_lower))
    overlap = unique_words & answer_words
    ok = len(overlap) >= 5
    return ok, f"{len(overlap)} content words shared with retrieved context"


def check_no_hallucinated_deadline(answer: str, chunks: list[dict], **_) -> tuple[bool, str]:
    answer_dates = set(m.lower() for m in DATE_PATTERN.findall(answer))
    if not answer_dates:
        return True, "no date patterns found in answer"

    context_text = " ".join(c.get("text", "").lower() for c in chunks)
    hallucinated = []
    for date in answer_dates:
        # "week N" dates are from policy text — acceptable if chunk text mentions "week N"
        if date not in context_text:
            hallucinated.append(date)

    if hallucinated:
        return False, f"date(s) not found in retrieved context: {hallucinated}"
    return True, f"all {len(answer_dates)} date(s) traceable to retrieved context"


def check_no_hallucinated_fee(answer: str, chunks: list[dict], **_) -> tuple[bool, str]:
    answer_fees = set(FEE_PATTERN.findall(answer))
    if not answer_fees:
        return True, "no fee amounts found in answer"

    context_text = " ".join(c.get("text", "") for c in chunks)
    hallucinated = []
    for fee in answer_fees:
        # Normalize: remove spaces between $ and digits
        fee_norm = re.sub(r'\$\s*', '$', fee)
        if fee_norm not in context_text and fee not in context_text:
            hallucinated.append(fee)

    if hallucinated:
        return False, f"fee amount(s) not in retrieved context: {hallucinated}"
    return True, f"all {len(answer_fees)} fee amount(s) traceable to retrieved context"


def check_no_absolute_promise(answer: str, **_) -> tuple[bool, str]:
    matches = ABSOLUTE_PROMISE_PATTERN.findall(answer)
    if matches:
        return False, f"overconfident phrase(s): {matches[:3]}"
    return True, "no absolute promise phrases found"


def check_safe_escalation(answer: str, category: str, safety_expectation: str, **_) -> tuple[bool, str]:
    answer_lower = answer.lower()
    terms = OFFICIAL_OFFICE_TERMS.get(category, [])
    mentioned = [t for t in terms if t in answer_lower]
    if mentioned:
        return True, f"mentions official office: {mentioned[0]}"
    # Soft pass: if safety_expectation doesn't demand escalation, skip
    if "must" not in safety_expectation.lower() or "contact" not in safety_expectation.lower():
        return True, "escalation not strictly required for this query type"
    return False, f"no official office mention found for category '{category}'"


def check_answer_has_steps(answer: str, query: str, **_) -> tuple[bool, str]:
    if not HOW_TO_PATTERN.search(query):
        return True, "query does not ask for steps (check skipped)"
    has_steps = bool(STEP_PATTERN.search(answer)) or bool(re.search(r'\bfirst\b.*\bthen\b', answer, re.IGNORECASE))
    if has_steps:
        return True, "answer contains numbered/bulleted steps or first/then structure"
    return False, "how-to query but answer has no step structure"


def check_no_forbidden_claims(answer: str, forbidden_claims: list[str], **_) -> tuple[bool, str]:
    answer_lower = answer.lower()
    found = [fc for fc in forbidden_claims if fc.lower() in answer_lower]
    if found:
        return False, f"forbidden claim(s) found: {found}"
    return True, "no forbidden claims found"


CHECKS = [
    ("answer_not_empty",         check_answer_not_empty),
    ("cites_source",             check_cites_source),
    ("uses_retrieved_context",   check_uses_retrieved_context),
    ("no_hallucinated_deadline", check_no_hallucinated_deadline),
    ("no_hallucinated_fee",      check_no_hallucinated_fee),
    ("no_absolute_promise",      check_no_absolute_promise),
    ("safe_escalation",          check_safe_escalation),
    ("answer_has_steps",         check_answer_has_steps),
    ("no_forbidden_claims",      check_no_forbidden_claims),
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_answer(record: dict, seed_item: dict) -> dict:
    answer = record.get("generated_answer") or ""
    chunks = record.get("retrieved_chunks", [])
    category = record.get("category", seed_item.get("category", ""))
    query = record.get("query", seed_item.get("query", ""))
    forbidden_claims = seed_item.get("forbidden_claims", [])
    safety_expectation = seed_item.get("safety_expectation", "")

    results = {}
    for check_name, check_fn in CHECKS:
        passed, detail = check_fn(
            answer=answer,
            chunks=chunks,
            category=category,
            query=query,
            forbidden_claims=forbidden_claims,
            safety_expectation=safety_expectation,
        )
        results[check_name] = {"passed": passed, "detail": detail}

    n_checks = len(CHECKS)
    n_passed = sum(1 for v in results.values() if v["passed"])
    return {
        "id": record.get("id", seed_item.get("id", "")),
        "category": category,
        "query": query,
        "answer_preview": answer[:200].replace("\n", " ") if answer else "(empty)",
        "checks": results,
        "score": f"{n_passed}/{n_checks}",
        "passed": n_passed == n_checks,
    }


def render_markdown_report(eval_results: list[dict], answers_file: str) -> str:
    total = len(eval_results)
    all_passed = sum(1 for r in eval_results if r["passed"])
    check_names = [c[0] for c in CHECKS]

    lines = [
        "# RAG Answer Eval Report",
        "",
        f"**Answers file:** `{answers_file}`",
        f"**Queries evaluated:** {total}",
        f"**All checks passed:** {all_passed} / {total}",
        "",
        "## Summary Table",
        "",
    ]

    # Header
    header = "| id | category | score | " + " | ".join(check_names) + " |"
    sep = "|---|---|---|" + "|".join(["---"] * len(check_names)) + "|"
    lines += [header, sep]

    for r in eval_results:
        check_icons = " | ".join(
            "✓" if r["checks"][c]["passed"] else "✗" for c in check_names
        )
        lines.append(f"| {r['id']} | {r['category']} | {r['score']} | {check_icons} |")

    lines += [
        "",
        "## Per-Category Pass Rate",
        "",
    ]

    from collections import defaultdict
    cat_results: dict[str, list[bool]] = defaultdict(list)
    for r in eval_results:
        cat_results[r["category"]].append(r["passed"])
    for cat, bools in sorted(cat_results.items()):
        pct = sum(bools) / len(bools) * 100
        lines.append(f"- **{cat}**: {sum(bools)}/{len(bools)} ({pct:.0f}%)")

    lines += [
        "",
        "## Failures",
        "",
    ]

    any_failure = False
    for r in eval_results:
        failed_checks = {k: v for k, v in r["checks"].items() if not v["passed"]}
        if failed_checks:
            any_failure = True
            lines.append(f"### {r['id']} ({r['category']})")
            lines.append(f"> {r['query']}")
            lines.append("")
            lines.append(f"**Answer preview:** {r['answer_preview']}")
            lines.append("")
            for check_name, result in failed_checks.items():
                lines.append(f"- ✗ **{check_name}**: {result['detail']}")
            lines.append("")

    if not any_failure:
        lines.append("*No failures — all checks passed for all answers.*")

    lines += [
        "",
        "## Check Definitions",
        "",
        "| Check | Description |",
        "|---|---|",
        "| answer_not_empty | Answer is non-empty |",
        "| cites_source | Answer contains a source title or URL from retrieved chunks |",
        "| uses_retrieved_context | Answer shares content keywords with retrieved chunk text |",
        "| no_hallucinated_deadline | No date/week pattern in answer absent from retrieved context |",
        "| no_hallucinated_fee | No dollar amount in answer absent from retrieved context |",
        "| no_absolute_promise | No overconfident phrases (guaranteed, definitely, you are fine, ...) |",
        "| safe_escalation | Mentions relevant official office for high-risk queries |",
        "| answer_has_steps | How-to queries have numbered/bulleted steps |",
        "| no_forbidden_claims | Answer does not contain eval-seed forbidden phrases |",
    ]

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG answers.")
    parser.add_argument("--answers_file", required=True,
                        help="JSON file with generated answers (list of records).")
    parser.add_argument("--eval_seed", default="data/rag/rag_answer_eval_seed.json")
    parser.add_argument("--report_dir", default="outputs/rag_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    answers = json.loads(Path(args.answers_file).read_text())
    eval_seed = json.loads(Path(args.eval_seed).read_text())
    seed_by_id = {item["id"]: item for item in eval_seed}

    eval_results = []
    for record in answers:
        item_id = record.get("id", "")
        seed_item = seed_by_id.get(item_id, {})
        if record.get("generated_answer") is None:
            print(f"[skip] {item_id} — generated_answer is null (not yet generated)")
            continue
        result = evaluate_answer(record, seed_item)
        eval_results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {item_id} {result['score']}")

    if not eval_results:
        print("No answers to evaluate. Run generation first (--generate mode or Colab).")
        return

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    md_path = report_dir / "rag_answer_eval_report.md"
    json_path = report_dir / "rag_answer_eval_report.json"

    md_path.write_text(render_markdown_report(eval_results, args.answers_file))
    json_path.write_text(json.dumps(eval_results, indent=2, ensure_ascii=False) + "\n")

    n_passed = sum(1 for r in eval_results if r["passed"])
    print(f"\nResults: {n_passed}/{len(eval_results)} answers passed all checks")
    print(f"Report  → {md_path}")
    print(f"JSON    → {json_path}")


if __name__ == "__main__":
    main()
