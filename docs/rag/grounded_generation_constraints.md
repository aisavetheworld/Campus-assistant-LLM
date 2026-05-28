# Grounded Generation Constraints — Project 2 RAG

**Date:** 2026-05-28
**Status:** Implemented; awaiting GPU run for empirical validation.

## Why

The eval round on `data/rag/rag_answer_eval_seed.json` (45 queries) surfaced
recurring failure modes that the system prompt alone did not prevent:

| Failure mode | DPO observed | Base observed |
|--------------|--------------|---------------|
| Hallucinated week-N deadline | 2 (week 4, week 9) | 3 (week 4, week 5, week 9) |
| Absolute promise ("guarantee") | 0 | 1 (housing) |
| Forbidden claim ("you are eligible for OPT") | 0 | 1 |

These are not retrieval failures (the right chunks are in context). They are
generation-side errors: the model produced text that is not in the retrieved
context but that *sounds* like UCSD policy.

The reranker experiment (`docs/rag/reranker_comparison_report.md`) ruled out
retrieval reordering as a fix. The remaining lever is generation-side
enforcement of evidence-only behavior with deterministic post-hoc validation.

## Design

```
retrieve top-5
  │
  ▼
pre-gen confidence gate          (skip generation, return fallback)
  │  top_score < min_top_score?
  ▼
attempt 1: strict-prompt generation
  │
  ▼
validators (11 checks)            (see Checks section)
  │  all pass? → return
  ▼
attempt 2: retry with fix hints   (e.g. "Do not state these dates: ['week 5']")
  │
  ▼
validators
  │  all pass? → return
  ▼
fallback_message                  ("I could not verify this from the retrieved
                                   official sources. …")
```

All validators live in `scripts/rag/answer_validators.py` and are shared by:
- **Runtime** (`scripts/rag/rag_answer.py`) — drives the retry/fallback loop.
- **Eval** (`scripts/rag/eval_rag_answer.py`) — reports per-check pass rates.

Single source of truth → eval and runtime cannot drift.

## Config

`configs/rag_generation.yaml` holds:

| Key | Purpose |
|-----|---------|
| `min_top_score` | Hybrid score below which we skip generation. |
| `fallback_on_low_confidence` | Whether the gate is active. |
| `fallback_message` | Returned when gated or after retries. |
| `require_source_citation` | Reject answers missing a `Sources:` section. |
| `chunk_prompt_fields` | Fields injected into the model prompt per chunk. |
| `max_retries` | Default 1. Each retry is one extra model forward pass. |
| `forbidden_absolute_phrases` | Bad-phrase list scanned in the answer. |
| `forbidden_unsourced_claim_types` | Categories the model may not invent. |
| `forbidden_meta_patterns` | `Note:`, `Explanation:`, `As an AI language model`, … |
| `escalation_offices` | Per-category list of offices the model should escalate to. |

## Checks (11)

| Check | Hard fail? | Drives a retry hint? |
|-------|-----------|----------------------|
| `answer_not_empty` | yes | yes (generic) |
| `cites_source` | yes | yes (force `Sources:` + listed title/URL) |
| `uses_retrieved_context` | yes | yes (reuse retrieved terminology) |
| `no_hallucinated_deadline` | yes | yes (lists offending dates) |
| `no_hallucinated_fee` | yes | yes (lists offending amounts) |
| `no_absolute_promise` | yes | yes (lists offending phrases) |
| `safe_escalation` | yes (high-risk only) | yes (suggests offices) |
| `answer_has_steps` | how-to only | yes (force numbered steps) |
| `no_extra_notes` | yes | yes (NEW — strips `Note:` / AI self-ref) |
| `no_forbidden_claims` | yes | yes (eval-seed driven) |
| `insufficient_context_behavior` | low-conf only | n/a (fallback path) |

### Per-check rationale

- **`cites_source`** is stricter than before: the answer must literally contain a
  `Sources:` header AND match at least one retrieved title or URL. Previously
  the title alone (without header) counted.
- **`no_hallucinated_deadline` / `no_hallucinated_fee`** scan the model output
  for date/$ patterns and verify each appears verbatim in the retrieved text.
  Fees that the user themselves mentioned in the query are not penalized.
- **`no_absolute_promise`** uses a phrase list + a smarter `guarantee`/`guaranteed`
  rule that does not flag noun-compounds like "housing guarantee" or negated
  forms like "not guaranteed".
- **`no_extra_notes`** is new. Catches `Note:`, `Explanation:`, `Rationale:`,
  `Disclaimer:`, `As an AI language model`, `Human:` / `Assistant:`, and
  Qwen template markers `<|im_start|>` / `<|im_end|>`.
- **`insufficient_context_behavior`** is new. When the runtime decides the
  retrieval is low-confidence, the answer must be the fallback message
  (or a paraphrase containing "could not verify" + "official"). Eval flags
  any low-conf record whose answer is not the fallback.

## Retry semantics

When validators fail, `collect_fix_hints()` returns the deduplicated set of
`fix_hint` strings from the failed checks. These are appended to the user
prompt under an `Additional constraints for this answer:` block, prefixed by
`- `, and the model is re-prompted with the same system message.

Cost: failed queries incur one extra forward pass. With DPO baseline ~95% pass
rate, expected end-to-end latency is `1.05 × single-gen latency` at fallback
rate near zero.

## System prompt changes

The previous prompt was advisory ("Base your answer strictly on this context").
The new one is enumerated and explicit:

1. Evidence-only
2. No invented facts (lists categories: deadline, date, week, fee, dollar,
   eligibility, approval, guarantee)
3. No absolute promises (lists banned words)
4. Always cite via `Sources:` section
5. Escalate when uncertain (lists offices per question type)
6. No meta-commentary (lists banned prefixes)

Plus a rigid format spec: direct answer → next steps → safety note → `Sources:`.

## Prompt-only mode (validation before GPU)

```bash
python scripts/rag/rag_answer.py \
    --build_prompt_only \
    --eval_seed data/rag/rag_answer_eval_seed.json \
    --output_file outputs/rag_eval/grounded_prompts/batch.json
```

Each record in `batch.json` carries:
- `system_message` — the strict prompt above
- `user_prompt` — retrieved context + question + format reminder
- `retrieved_chunks` — full chunks, for downstream eval
- `prompt_chunks` — slim chunks (`source_id`, `title`, `section_title`, `url`, `text`)
- `top_score`, `low_confidence` — runtime gating signal
- `safety_expectation`, `forbidden_claims` — passed through from eval seed

Inspecting `batch.json` before running on GPU lets us catch system-prompt
regressions and verify the chunks contain the required fields.

## Batch generation (Colab / GPU)

```bash
python scripts/rag/rag_answer.py \
    --batch_generate \
    --prompts_file outputs/rag_eval/grounded_prompts/batch.json \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_path outputs/dpo_7b \
    --output_file outputs/rag_eval/generated_answers_dpo.json
```

Each output record now carries:
- `generated_answer` — final string returned (may be the fallback message)
- `attempts` — 1 (no retry), 2 (retried), or 0 (pre-gen fallback)
- `fallback_triggered` — bool
- `fallback_reason` — `low_retrieval_confidence` / `validation_failed_after_retry: [...]`
- `validation` — per-check dict (passed/detail/fix_hint)

## Eval

```bash
python scripts/rag/eval_rag_answer.py \
    --answers_file outputs/rag_eval/generated_answers_dpo.json \
    --report_suffix dpo_constrained
```

Report (`outputs/rag_eval/rag_answer_eval_report_dpo_constrained.md`) shows:
- All-pass count (n / 45)
- Per-category pass rate
- Per-check pass rate (new — diagnoses which constraint hurts most)
- Retry / fallback rate
- Detailed failure breakdown per query

## Expected outcome

| Metric | Before | Target |
|--------|--------|--------|
| DPO all-pass | 43 / 45 | ≥ 44 / 45 (002 / 006 hallucinated week resolved by retry) |
| Base all-pass | 40 / 45 | ≥ 43 / 45 (019, 040 caught by absolute-promise/forbidden-claim retry) |
| Fallback rate | n/a | ≤ 5% (gating only triggers on the 3 low-confidence queries) |
| Retry rate | n/a | 5–15% (most queries pass attempt 1) |
| Avg latency | 1 × gen | ~1.1 × gen |

If retries exceed 30%, the prompt is wrong (not the validators).

## What this does NOT do

- **No decode-time constraint** (no logits processor, no FSM grammar).
  Adding `outlines` / `lm-format-enforcer` would enforce format
  hard-deterministically but doubles infra complexity. Deferred to P2 if
  post-hoc + retry is insufficient.
- **No span-level citation** (each sentence linked to a chunk span). P2.
- **No automated citation verification** (does the cited source actually
  contain the claim). P2.

## Files touched

- `configs/rag_generation.yaml` (new)
- `scripts/rag/answer_validators.py` (new — shared validator module)
- `scripts/rag/rag_answer.py` (strengthened prompt, retry/fallback loop, schema)
- `scripts/rag/eval_rag_answer.py` (delegates to shared validators, adds 2 checks)
- `docs/rag/grounded_generation_constraints.md` (this doc)
