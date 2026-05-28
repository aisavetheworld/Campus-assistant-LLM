"""Shared post-generation validators for grounded RAG answers.

Used by:
  - scripts/rag/rag_answer.py (runtime constraint enforcement + retry)
  - scripts/rag/eval_rag_answer.py (post-hoc eval reporting)

Each validator returns (passed: bool, detail: str, fix_hint: str).
fix_hint is the phrase to inject into a retry prompt (e.g. "Do not include
absolute promises such as 'guaranteed'."). Empty string when passed=True.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import yaml

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("configs/rag_generation.yaml")


def load_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Patterns (compiled once)
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

_GUARANTEE_PATTERN = re.compile(r'\bguaranteed?\b', re.IGNORECASE)
_NEGATION_BEFORE = re.compile(r'\b(?:not|no|cannot|can\'t|never|without)\s*$', re.IGNORECASE)
_COPULA_BEFORE = re.compile(r'\b(?:is|are|was|were|will be|be|am)\s*$', re.IGNORECASE)

HOW_TO_PATTERN = re.compile(
    r'\b(?:how (?:do|can|should) (?:i|we)|what (?:are|is) the (?:steps?|process))\b',
    re.IGNORECASE,
)

STEP_PATTERN = re.compile(r'(?:^|\n)\s*(?:\d+[.)]\s|\*\s|-\s)')

SOURCES_HEADER_PATTERN = re.compile(r'(?i)\bsources?\s*:')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_fee(fee: str) -> str:
    return re.sub(r'[$,\s]', '', fee)


def _is_fallback_message(answer: str, fallback_message: str) -> bool:
    """Loose match: answer contains the core fallback phrase."""
    core = "could not verify"
    return core in answer.lower() and "official" in answer.lower()


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def check_answer_not_empty(answer: str, **_) -> tuple[bool, str, str]:
    if answer and answer.strip():
        return True, "", ""
    return False, "answer is empty or whitespace only", "Provide a non-empty answer."


def check_cites_source(answer: str, chunks: list[dict], config: dict, **_) -> tuple[bool, str, str]:
    if not config.get("require_source_citation", True):
        return True, "citation not required", ""

    # Must have a "Sources:" header somewhere in the answer
    if not SOURCES_HEADER_PATTERN.search(answer):
        return False, "no 'Sources:' section found", \
            "End the answer with a 'Sources:' section listing the source titles or URLs."

    # And must reference at least one retrieved source by title/url
    answer_lower = answer.lower()
    for chunk in chunks:
        title = (chunk.get("title") or "").lower()
        url = (chunk.get("url") or "").lower()
        if title and title in answer_lower:
            return True, f"cites '{chunk.get('title', '')}'", ""
        if url and url in answer_lower:
            return True, "cites URL", ""
        title_words = title.split()[:4]
        if len(title_words) >= 2 and " ".join(title_words) in answer_lower:
            return True, "cites partial title", ""
    return False, "Sources section present but no retrieved title/URL matched", \
        "In the Sources section, list the exact source titles or URLs from the retrieved context."


def check_uses_retrieved_context(answer: str, chunks: list[dict], **_) -> tuple[bool, str, str]:
    answer_lower = answer.lower()
    all_text = " ".join((c.get("text") or "").lower() for c in chunks)
    unique_words = set(re.findall(r'\b[a-z]{4,}\b', all_text))
    answer_words = set(re.findall(r'\b[a-z]{4,}\b', answer_lower))
    overlap = unique_words & answer_words
    ok = len(overlap) >= 5
    if ok:
        return True, f"{len(overlap)} content words shared with retrieved context", ""
    return False, f"only {len(overlap)} content words shared with retrieved context", \
        "Base the answer on the retrieved context and reuse its terminology."


def check_no_hallucinated_deadline(answer: str, chunks: list[dict], **_) -> tuple[bool, str, str]:
    answer_dates = set(m.lower() for m in DATE_PATTERN.findall(answer))
    if not answer_dates:
        return True, "no date patterns in answer", ""

    context_text = " ".join((c.get("text") or "").lower() for c in chunks)
    hallucinated = [d for d in answer_dates if d not in context_text]
    if hallucinated:
        return False, f"date(s) not in retrieved context: {hallucinated}", \
            f"Do not state these dates/weeks (not in sources): {hallucinated}. " \
            "Only mention deadlines that appear verbatim in the retrieved context."
    return True, f"all {len(answer_dates)} date(s) traceable to retrieved context", ""


def check_no_hallucinated_fee(answer: str, chunks: list[dict], query: str = "", **_) -> tuple[bool, str, str]:
    answer_fees = set(FEE_PATTERN.findall(answer))
    if not answer_fees:
        return True, "no fee amounts in answer", ""

    query_fees = set(_normalize_fee(f) for f in FEE_PATTERN.findall(query))
    context_text = " ".join((c.get("text") or "") for c in chunks)
    context_fees = set(_normalize_fee(f) for f in FEE_PATTERN.findall(context_text))

    hallucinated = []
    for fee in answer_fees:
        norm = _normalize_fee(fee)
        if norm in query_fees:
            continue
        if norm not in context_fees:
            hallucinated.append(fee)

    if hallucinated:
        return False, f"fee amount(s) not in retrieved context: {hallucinated}", \
            f"Do not state these dollar amounts (not in sources): {hallucinated}. " \
            "Only mention fees that appear verbatim in the retrieved context."
    return True, f"all {len(answer_fees)} fee amount(s) traceable to retrieved context", ""


def check_no_absolute_promise(answer: str, config: dict, **_) -> tuple[bool, str, str]:
    forbidden = config.get("forbidden_absolute_phrases", [])
    answer_lower = answer.lower()
    matches = [p for p in forbidden if p.lower() in answer_lower]

    # "guaranteed" handled separately to allow "housing guarantee" noun-compound
    # and negated forms like "not guaranteed".
    for m in _GUARANTEE_PATTERN.finditer(answer):
        before = answer[max(0, m.start() - 35):m.start()]
        if _NEGATION_BEFORE.search(before):
            continue
        if re.search(r'\b\w+\s+$', before) and not _COPULA_BEFORE.search(before):
            continue
        matches.append(m.group())

    if matches:
        return False, f"overconfident phrase(s): {matches[:3]}", \
            f"Do not use absolute-promise phrases: {matches[:3]}. " \
            "Use cautious language ('may', 'typically', 'in most cases')."
    return True, "no absolute promise phrases found", ""


def check_safe_escalation(answer: str, category: str, safety_expectation: str, config: dict, **_) -> tuple[bool, str, str]:
    answer_lower = answer.lower()
    escalation = config.get("escalation_offices", {})
    terms = [t.lower() for t in escalation.get(category, [])]
    mentioned = [t for t in terms if t in answer_lower]
    if mentioned:
        return True, f"mentions official office: {mentioned[0]}", ""
    # Soft pass if safety_expectation doesn't strictly demand escalation
    if "must" not in safety_expectation.lower() or "contact" not in safety_expectation.lower():
        return True, "escalation not strictly required for this query type", ""
    offices = escalation.get(category, [])
    return False, f"no official office mention for category '{category}'", \
        f"Recommend contacting the relevant office: {', '.join(offices[:2])}."


def check_answer_has_steps(answer: str, query: str, **_) -> tuple[bool, str, str]:
    if not HOW_TO_PATTERN.search(query):
        return True, "query does not ask for steps (check skipped)", ""
    has_steps = bool(STEP_PATTERN.search(answer)) or bool(
        re.search(r'\bfirst\b.*\bthen\b', answer, re.IGNORECASE)
    )
    if has_steps:
        return True, "answer contains numbered/bulleted steps or first/then structure", ""
    return False, "how-to query but answer has no step structure", \
        "Structure the answer as numbered steps (1. 2. 3.) since the question asks how to do something."


def check_no_extra_notes(answer: str, config: dict, **_) -> tuple[bool, str, str]:
    patterns = config.get("forbidden_meta_patterns", [])
    found = [p for p in patterns if p.lower() in answer.lower()]
    if found:
        return False, f"meta-commentary markers found: {found[:3]}", \
            f"Do not include meta-commentary markers like {found[:3]}. " \
            "Write the answer directly without 'Note:', 'Explanation:', or AI self-references."
    return True, "no meta-commentary markers found", ""


def check_no_forbidden_claims(answer: str, forbidden_claims: list[str], **_) -> tuple[bool, str, str]:
    """Eval-seed driven check: forbidden_claims per query."""
    if not forbidden_claims:
        return True, "no forbidden_claims defined for this query", ""
    answer_lower = answer.lower()
    found = [fc for fc in forbidden_claims if fc.lower() in answer_lower]
    if found:
        return False, f"forbidden claim(s) found: {found}", \
            f"Do not assert: {found}. Use conditional language instead."
    return True, "no forbidden claims found", ""


def check_insufficient_context_behavior(
    answer: str, low_confidence: bool, config: dict, **_
) -> tuple[bool, str, str]:
    """If retrieval was low-confidence, answer MUST be the fallback message
    (or at minimum acknowledge inability to verify)."""
    if not low_confidence:
        return True, "retrieval confidence acceptable; check not applicable", ""
    fallback_msg = config.get("fallback_message", "could not verify")
    if _is_fallback_message(answer, fallback_msg):
        return True, "low-confidence query returned fallback as expected", ""
    return False, "low-confidence query but answer did not fall back to safe message", \
        "Return the fallback message: 'I could not verify this from the retrieved official sources.'"


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

ValidatorFn = Callable[..., tuple[bool, str, str]]

CHECKS: list[tuple[str, ValidatorFn]] = [
    ("answer_not_empty",                check_answer_not_empty),
    ("cites_source",                    check_cites_source),
    ("uses_retrieved_context",          check_uses_retrieved_context),
    ("no_hallucinated_deadline",        check_no_hallucinated_deadline),
    ("no_hallucinated_fee",             check_no_hallucinated_fee),
    ("no_absolute_promise",             check_no_absolute_promise),
    ("safe_escalation",                 check_safe_escalation),
    ("answer_has_steps",                check_answer_has_steps),
    ("no_extra_notes",                  check_no_extra_notes),
    ("no_forbidden_claims",             check_no_forbidden_claims),
    ("insufficient_context_behavior",   check_insufficient_context_behavior),
]


def run_all_checks(
    answer: str,
    chunks: list[dict],
    query: str = "",
    category: str = "",
    forbidden_claims: list[str] | None = None,
    safety_expectation: str = "",
    low_confidence: bool = False,
    config: dict | None = None,
) -> dict:
    """Run every validator and return a dict keyed by check name.

    Returns:
        {
          check_name: {"passed": bool, "detail": str, "fix_hint": str},
          ...
        }
    """
    if config is None:
        config = load_config()
    forbidden_claims = forbidden_claims or []

    kwargs = dict(
        answer=answer,
        chunks=chunks,
        query=query,
        category=category,
        forbidden_claims=forbidden_claims,
        safety_expectation=safety_expectation,
        low_confidence=low_confidence,
        config=config,
    )

    results = {}
    for name, fn in CHECKS:
        passed, detail, fix_hint = fn(**kwargs)
        results[name] = {"passed": passed, "detail": detail, "fix_hint": fix_hint}
    return results


def collect_fix_hints(results: dict) -> list[str]:
    """Return all fix_hints for failed checks, deduplicated."""
    hints = []
    seen = set()
    for r in results.values():
        if not r["passed"] and r["fix_hint"]:
            h = r["fix_hint"]
            if h not in seen:
                hints.append(h)
                seen.add(h)
    return hints


def all_passed(results: dict) -> bool:
    return all(r["passed"] for r in results.values())
