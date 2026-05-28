"""Rule-based evaluation of grounded RAG answers.

Delegates all per-answer checks to scripts/rag/answer_validators.py so that
runtime constraint enforcement (rag_answer.py) and post-hoc eval share the
same logic.

Checks reported:
  answer_not_empty
  cites_source
  uses_retrieved_context
  no_hallucinated_deadline
  no_hallucinated_fee
  no_absolute_promise
  safe_escalation
  answer_has_steps               (skipped for non-how-to queries)
  no_extra_notes                 (NEW: rejects Note:/Explanation:/AI self-ref)
  no_forbidden_claims            (eval-seed driven)
  insufficient_context_behavior  (NEW: low-confidence must use fallback)

Input:
  --answers_file: JSON file from rag_answer.py --batch_generate (each record
                  has: id, category, query, retrieved_chunks, generated_answer,
                  and optionally low_confidence/fallback_triggered/validation)
  --eval_seed:    Eval seed JSON with forbidden_claims, safety_expectation

Usage:
    python scripts/rag/eval_rag_answer.py \
        --answers_file outputs/rag_eval/generated_answers_dpo.json \
        --report_suffix dpo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from answer_validators import (
    CHECKS, load_config, run_all_checks, all_passed,
)


def evaluate_answer(record: dict, seed_item: dict, constraint_config: dict) -> dict:
    answer = record.get("generated_answer") or ""
    chunks = record.get("retrieved_chunks", [])
    category = record.get("category", seed_item.get("category", ""))
    query = record.get("query", seed_item.get("query", ""))
    forbidden_claims = seed_item.get("forbidden_claims", [])
    safety_expectation = seed_item.get("safety_expectation", "")
    low_confidence = bool(record.get("low_confidence", False))

    results = run_all_checks(
        answer=answer, chunks=chunks, query=query, category=category,
        forbidden_claims=forbidden_claims, safety_expectation=safety_expectation,
        low_confidence=low_confidence, config=constraint_config,
    )
    n_checks = len(results)
    n_passed = sum(1 for v in results.values() if v["passed"])
    return {
        "id": record.get("id", seed_item.get("id", "")),
        "category": category,
        "query": query,
        "answer_preview": answer[:200].replace("\n", " ") if answer else "(empty)",
        "checks": results,
        "score": f"{n_passed}/{n_checks}",
        "passed": n_passed == n_checks,
        # Runtime metadata from rag_answer.py (when constraints are enabled)
        "constraints_enabled": bool(record.get("constraints_enabled", False)),
        "attempts": record.get("attempts"),
        "fallback_triggered": record.get("fallback_triggered"),
        "fallback_reason": record.get("fallback_reason", ""),
    }


def render_markdown_report(eval_results: list[dict], answers_file: str) -> str:
    total = len(eval_results)
    all_pass = sum(1 for r in eval_results if r["passed"])
    check_names = [c[0] for c in CHECKS]

    # Runtime stats (when constraints are enabled)
    constrained = [r for r in eval_results if r.get("constraints_enabled")]
    fallback_n = sum(1 for r in constrained if r.get("fallback_triggered"))
    retry_n = sum(1 for r in constrained if (r.get("attempts") or 0) > 1)

    lines = [
        "# RAG Answer Eval Report",
        "",
        f"**Answers file:** `{answers_file}`",
        f"**Queries evaluated:** {total}",
        f"**All checks passed:** {all_pass} / {total}",
    ]
    if constrained:
        lines += [
            f"**Constraints enabled:** {len(constrained)} / {total}",
            f"**Retries:** {retry_n} / {len(constrained)}",
            f"**Fallback triggered:** {fallback_n} / {len(constrained)}",
        ]
    lines += ["", "## Summary Table", ""]

    header = "| id | category | score | " + " | ".join(check_names) + " |"
    sep = "|---|---|---|" + "|".join(["---"] * len(check_names)) + "|"
    lines += [header, sep]
    for r in eval_results:
        icons = " | ".join(
            "✓" if r["checks"][c]["passed"] else "✗" for c in check_names
        )
        lines.append(f"| {r['id']} | {r['category']} | {r['score']} | {icons} |")

    lines += ["", "## Per-Category Pass Rate", ""]
    cat_results: dict[str, list[bool]] = defaultdict(list)
    for r in eval_results:
        cat_results[r["category"]].append(r["passed"])
    for cat, bools in sorted(cat_results.items()):
        pct = sum(bools) / len(bools) * 100
        lines.append(f"- **{cat}**: {sum(bools)}/{len(bools)} ({pct:.0f}%)")

    lines += ["", "## Per-Check Pass Rate", ""]
    for c in check_names:
        passed = sum(1 for r in eval_results if r["checks"][c]["passed"])
        pct = passed / total * 100 if total else 0
        lines.append(f"- **{c}**: {passed}/{total} ({pct:.0f}%)")

    lines += ["", "## Failures", ""]
    any_failure = False
    for r in eval_results:
        failed = {k: v for k, v in r["checks"].items() if not v["passed"]}
        if failed:
            any_failure = True
            lines.append(f"### {r['id']} ({r['category']})")
            lines.append(f"> {r['query']}")
            if r.get("fallback_triggered"):
                lines.append(f"*Fallback triggered: {r.get('fallback_reason', '')}*")
            lines.append("")
            lines.append(f"**Answer preview:** {r['answer_preview']}")
            lines.append("")
            for name, result in failed.items():
                lines.append(f"- ✗ **{name}**: {result['detail']}")
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
        "| cites_source | Has a `Sources:` section AND references a retrieved title/URL |",
        "| uses_retrieved_context | Shares ≥5 content words with retrieved chunks |",
        "| no_hallucinated_deadline | No date/week pattern in answer absent from retrieved context |",
        "| no_hallucinated_fee | No $ amount in answer absent from retrieved context |",
        "| no_absolute_promise | No definitely/guaranteed/you-are-fine etc. |",
        "| safe_escalation | High-risk queries mention the relevant official office |",
        "| answer_has_steps | How-to queries have numbered or bulleted steps |",
        "| no_extra_notes | No Note:/Explanation:/AI self-reference/template labels |",
        "| no_forbidden_claims | Avoids eval-seed forbidden phrases |",
        "| insufficient_context_behavior | Low-confidence queries return the fallback message |",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG answers.")
    parser.add_argument("--answers_file", required=True,
                        help="JSON file with generated answers (list of records).")
    parser.add_argument("--eval_seed", default="data/rag/rag_answer_eval_seed.json")
    parser.add_argument("--report_dir", default="outputs/rag_eval")
    parser.add_argument("--report_suffix", default="",
                        help="Suffix for report filenames: rag_answer_eval_report_<suffix>")
    parser.add_argument("--constraint_config", default="configs/rag_generation.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constraint_config = load_config(args.constraint_config)

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
        result = evaluate_answer(record, seed_item, constraint_config)
        eval_results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {item_id} {result['score']}")

    if not eval_results:
        print("No answers to evaluate. Run generation first (--generate or batch).")
        return

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.report_suffix}" if args.report_suffix else ""
    md_path = report_dir / f"rag_answer_eval_report{suffix}.md"
    json_path = report_dir / f"rag_answer_eval_report{suffix}.json"

    md_path.write_text(render_markdown_report(eval_results, args.answers_file))
    json_path.write_text(json.dumps(eval_results, indent=2, ensure_ascii=False) + "\n")

    n_passed = sum(1 for r in eval_results if r["passed"])
    print(f"\nResults: {n_passed}/{len(eval_results)} answers passed all checks")
    print(f"Report  → {md_path}")
    print(f"JSON    → {json_path}")


if __name__ == "__main__":
    main()
