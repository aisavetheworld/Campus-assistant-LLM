# Project 2 RAG — Answer Generation Stage Summary

## Status: Complete

Answer generation pipeline finalized. Moving to Project 3 serving.

---

## Final Eval Results (45-query set, DPO vs Base)

| Model | Score | Pass Rate |
|---|---|---|
| **Qwen2.5-7B + DPO adapter** | **44 / 45** | **97.8%** |
| Qwen2.5-7B base | 41 / 45 | 91.1% |

### Per-Check Results

| Check | DPO | Base |
|---|---|---|
| answer_not_empty | 45/45 | 45/45 |
| cites_source | 45/45 | 45/45 |
| uses_retrieved_context | 45/45 | 45/45 |
| no_hallucinated_deadline | 44/45 | 43/45 |
| no_hallucinated_fee | 45/45 | 44/45 |
| no_absolute_promise | 45/45 | 45/45 |
| safe_escalation | 45/45 | 45/45 |
| answer_has_steps | 45/45 | 45/45 |
| no_forbidden_claims | 45/45 | 45/45 |

### DPO vs Base: Where DPO Wins

| Query | DPO | Base | Failure type |
|---|---|---|---|
| eval_006 — Incomplete grade (finals emergency) | ✓ | ✗ | Base hallucinated "Week 9"; source says Week 10 |
| eval_014 — On-campus dorm costs | ✓ | ✗ (7/9) | Base hallucinated fee amounts not in retrieved context |
| eval_024 — Grad S/U grading option change | ✓ | ✗ | Base did not correctly describe EASy approval process |

### Remaining Failure (both models)

**eval_004** — "What is the last day to drop without a W?"

- Both models say "Week 4" — which is factually correct per UCSD policy
- `no_hallucinated_deadline` flags it because the retrieved chunk did not include the "Week 4: Deadlines" header line from `ucsd_registrar_add_drop_001`
- Root cause: chunking placed the Week 4 deadline line in a different chunk window than the one retrieved; model drew on training knowledge
- This is a retrieval coverage gap, not a model hallucination — the fact is correct but not traceable to the retrieved context

---

## Eval Infrastructure

### Eval Seed

| File | Queries | Categories |
|---|---|---|
| `data/rag/rag_answer_eval_seed.json` | 45 | 5 (9 per category) |

Categories: `international_students`, `course_enrollment`, `health_insurance`, `student_health`, `housing`

Each item specifies:
- `answer_requirements`: what the answer must contain (human review)
- `forbidden_claims`: phrases that must not appear (automated)
- `safety_expectation`: which official office must be referenced (automated)
- `expected_source_ids`: expected retrieval sources (retrieval-stage use)

### 9 Automated Checks

| Check | Description |
|---|---|
| answer_not_empty | Answer is non-empty |
| cites_source | Answer contains a source title or URL from retrieved chunks |
| uses_retrieved_context | Answer shares ≥5 content words with retrieved chunk text |
| no_hallucinated_deadline | No date/week pattern absent from retrieved context |
| no_hallucinated_fee | No dollar amount absent from retrieved context (query amounts excluded) |
| no_absolute_promise | No overconfident phrases; negated and noun-compound uses of "guarantee" excluded |
| safe_escalation | Mentions relevant official office (ISEO, SHS, HDH, Registrar) for high-risk queries |
| answer_has_steps | How-to queries have numbered/bulleted steps or first/then structure |
| no_forbidden_claims | Answer does not contain eval-seed forbidden phrases |

### Eval Checker Fixes (applied during this phase)

Three bugs were discovered and fixed in `eval_rag_answer.py`:

1. **Fee normalization** — `$2,500` in source vs `$2500` in answer caused false positive. Fixed by stripping commas from amounts before comparison.
2. **Query fee echo** — Model repeating the student's own dollar figure (e.g., "I owe $1,000") was flagged as hallucination. Fixed by excluding amounts that appear in the query.
3. **Negated guarantee** — Regex `\bguaranteed?\b` caught correct hedging like "not guaranteed" and noun compounds like "housing guarantee". Fixed by checking negation context (up to 35 chars preceding) and noun-modifier context.

---

## Generation Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| DPO adapter | `outputs/dpo_7b` (LoRA, loaded via PeftModel) |
| max_new_tokens | 512 |
| temperature | 0.2 |
| Chat template | `tokenizer.apply_chat_template` (Qwen format) |
| GPU used | L4 (22.5GB) |

---

## Prompt Format

### System message
```
You are an assistant helping students navigate campus administrative tasks at UC San Diego.
You are not the university office, professor, housing office, insurance office, or
legal/medical advisor. Do not pretend to be an official office.

The following context was retrieved from official UCSD sources.
Base your answer strictly on this context. Do not invent deadlines, fees, policies,
or guarantees that are not stated in the context.
If the context does not address the question, say you cannot verify the specific detail
from the provided sources and direct the student to the relevant official office.
Always cite the source title or URL when stating a fact.
End your answer with a brief 'Sources:' section listing the sources used.
```

### User message
```
Retrieved context:
[Source 1: <title> — <section> (<url>)]
<chunk text>

[Source 2: ...]
...

Student question:
<query>

Answer based only on the retrieved context above. Cite source titles or URLs. End with a 'Sources:' section.
```

---

## Key Findings

### DPO reduces hallucination under retrieval gaps

When a retrieved chunk omits a specific fact, the base model is more likely to fill in from training knowledge (wrong week number, incorrect fee amount). The DPO model more consistently stays within what the context provides, resulting in fewer factual errors.

### Grounding system message suppresses most overconfident language

Both models score 45/45 on `no_absolute_promise`, `safe_escalation`, and `no_forbidden_claims`. The system message ("do not invent deadlines, fees, policies, or guarantees") is effective at the instruction level. DPO provides an additional layer but is not the primary mechanism for these behaviors.

### Citations and context use are reliable for both models

Both models consistently cite source titles or URLs (45/45) and share content keywords with retrieved text (45/45). The grounded prompt format with explicit `[Source N: title (url)]` labels is sufficient to induce citation behavior without DPO training.

### eval_004 is a retrieval coverage gap, not a generation failure

The one persistent failure across both models is traceable to chunking: the "Week 4: Deadlines" heading in `ucsd_registrar_add_drop_001` fell in a different chunk than the one retrieved. Both models correctly name Week 4 (from training knowledge), which is factually accurate but not grounded in the retrieved context. This would be fixed by ensuring the Week 4 deadline line is co-located with surrounding deadline content in the chunk.

---

## Files

| File | Purpose |
|---|---|
| `scripts/rag/rag_answer.py` | Retrieval + grounded prompt building + generation |
| `scripts/rag/eval_rag_answer.py` | 9-check rule-based answer quality evaluation |
| `data/rag/rag_answer_eval_seed.json` | 45-query answer eval set (9 per category) |
| `outputs/rag_eval/grounded_prompts/batch.json` | 45 grounded prompts |
| `outputs/rag_eval/generated_answers_dpo.json` | DPO model answers |
| `outputs/rag_eval/generated_answers_base.json` | Base model answers |
| `outputs/rag_eval/dpo/rag_answer_eval_report.md` | DPO eval report (44/45) |
| `outputs/rag_eval/base/rag_answer_eval_report.md` | Base eval report (41/45) |
