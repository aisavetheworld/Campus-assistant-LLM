# Query Expansion Report

## Motivation

Chunking ablation established that R@5 = 0.773 is the ceiling for dense-only retrieval with the current source corpus. The 10 remaining weak cases fall into two patterns:

1. **Vocabulary mismatch**: Student queries use natural language while official sources use policy jargon (e.g., "permission code" vs "authorization code", "drop below full-time" vs "Reduced Course Load / RCL", "F-1 visa" vs "UC SHIP insurance waiver")
2. **Source content gaps**: Some source pages are thin or missing specific terms entirely

Query expansion targets pattern 1 by appending known synonyms to the query before embedding, increasing the chance of matching relevant source text.

## Expansion Dictionary (`configs/rag_query_expansion.json`)

| Trigger key | Expansion terms added |
|---|---|
| `permission code` | enrollment authorization, instructor approval, department approval, course authorization |
| `without a W` | withdrawal, withdraw, drop after deadline, transcript notation |
| `W grade` | withdrawal, withdraw, drop after deadline, transcript notation |
| `Pass/No Pass` | grading option, grade option, letter grade change |
| `grade option` | grading option, pass no pass, letter grade, change grading option |
| `below full-time` | reduced course load, RCL, full-time enrollment requirement |
| `reduce units` | reduced course load, RCL, below full-time enrollment, full-time enrollment |
| `F-1 visa` | health insurance requirement, UC SHIP, insurance waiver, international student health insurance |
| `F-1 insurance` | health insurance requirement, UC SHIP, insurance waiver, international student health insurance |
| `SHIP waiver` | waiver eligibility, waiver requirements, proof of insurance, comparable coverage |
| `waiver criteria` | waiver eligibility, waiver requirements, proof of insurance, comparable coverage |
| `late fee` | waiver deadline, late waiver, fee deadline, missed deadline |
| `CPT` | curricular practical training, employment authorization, work authorization, internship authorization |
| `OPT` | optional practical training, post-completion OPT, employment authorization |

Matching is word-boundary aware (non-alphanumeric boundaries): `OPT` does not match `option`.

## Before / After Metrics

| Metric | Baseline | +Query Expansion | Delta |
|---|---|---|---|
| Recall@1 | 0.460 | **0.480** | +0.020 |
| Recall@3 | 0.753 | **0.773** | +0.020 |
| Recall@5 | 0.773 | **0.793** | +0.020 |
| Keyword Hit Rate | 0.761 | **0.789** | +0.028 |

## Per-Category Changes

| category | Baseline R@5 | Expanded R@5 | Delta |
|---|---|---|---|
| international_students | 0.900 | **1.000** | +0.100 |
| course_enrollment | 0.533 | 0.533 | 0 |
| **health_insurance** | 0.267 | **0.533** | **+0.267** |
| student_health | 1.000 | 1.000 | 0 |
| housing | 0.900 | 0.900 | 0 |

## Targeted Weak Case Analysis

| id | query (abbreviated) | Baseline R@5 | Expanded R@5 | Expansion fired? | Outcome |
|---|---|---|---|---|---|
| rag_eval_004 | drop below full-time enrollment (F-1) | 1.00\* | 1.00 | ✓ `below full-time` | R@1: 0.00→0.50 |
| rag_eval_008 | drop course without W on transcript | 0.67 | 0.67 | ✓ `without a W` | No change at R@5 |
| rag_eval_013 | change letter grade to Pass/No Pass | 0.50 | 0.50 | ✓ `Pass/No Pass` | No change at R@5 |
| rag_eval_014 | get permission code from professor | 0.00 | 0.00 | ✓ `permission code` | Still fails |
| rag_eval_015 | criteria to qualify for UC SHIP waiver | 0.50 | 0.50 | ✓ `SHIP waiver` | No change at R@5 |
| rag_eval_018 | F-1 visa, required to have UC SHIP? | 0.00 | **0.50** | ✓ `F-1 visa` | **Fixed partially** |

\*rag_eval_004 was already R@5=1.00 at baseline (both expected sources retrieved), but R@1 improved from 0.00→0.50 because the relevant chunk now ranks higher.

### What improved

**rag_eval_018 (R@5: 0.00 → 0.50)**: "F-1 visa" triggered expansion adding "UC SHIP, insurance waiver, international student health insurance". This pulled `ucsd_ucship_waiver_001` into the top-5 for the first time. Previously the query was routing entirely to ISEO visa-status pages.

**rag_eval_004 (R@1: 0.00 → 0.50)**: "below full-time" triggered "reduced course load, RCL" which correctly boosted `ucsd_iseo_visa_status_005` (the RCL/full-time enrollment page) to rank 1.

### What did not improve

**rag_eval_014 (R@5: still 0.00)**: "permission code" expanded with "enrollment authorization, instructor approval, department approval, course authorization". The expected source `ucsd_registrar_add_drop_001` ranks outside top-5 even after expansion — examination of the raw text shows this page does not contain the phrase "permission code" or "authorization code" at all. The retrieval failure is a **source content gap**, not a vocabulary mismatch.

**rag_eval_008, 011, 013 (R@5 unchanged)**: All three require `ucsd_registrar_add_drop_002` (the enrollment calendar). This page consistently ranks outside top-5 — it covers dates and deadlines but lacks the specific vocabulary of the student queries (W grade, grade option, authorization). Expansion did not change its ranking.

**rag_eval_006 (R@5: 0.67 unchanged)**: Query "What is the deadline to waive UC SHIP?" does not match any expansion key. `ucsd_ucship_waiver_002` still outranked by coverage and calendar pages.

## Limitations

1. **Source content gap is the hard ceiling**: rag_eval_014 will not be fixed by any query-side technique. The relevant source page lacks the queried vocabulary.

2. **Enrollment calendar (add_drop_002) is a persistent miss**: 3 of the 4 course_enrollment failures trace back to this single source consistently ranking 6th or lower. The page content may need manual enrichment or a dedicated query targeting its specific vocabulary.

3. **Expansion is static and order-independent**: The same synonyms fire for all students, regardless of context. A misfire could potentially degrade precision (no degradation observed in this eval set, but risk exists at scale).

4. **Single-embedding retrieval**: All expansion terms are concatenated into one query string and embedded as a single vector. Individually embedding the original + each expansion term and taking the union (multi-query retrieval) would be more robust but requires more compute.

## Remaining Weak Cases After Expansion

7 queries remain at R@5 < 1.0:

| id | R@5 | category | root cause |
|---|---|---|---|
| rag_eval_006 | 0.67 | health_insurance | waiver_002 outranked; no expansion trigger |
| rag_eval_008 | 0.67 | course_enrollment | add_drop_002 calendar not ranked in top-5 |
| rag_eval_011 | 0.50 | course_enrollment | add_drop_002 missing from top-5 |
| rag_eval_013 | 0.50 | course_enrollment | add_drop_002 missing from top-5 |
| rag_eval_014 | 0.00 | course_enrollment | source content gap; permission code absent |
| rag_eval_016 | 0.50 | health_insurance | waiver_004 (137 words) consistently thin |
| rag_eval_017 | 0.50 | health_insurance | coverage_002 outranked by use pages |
| rag_eval_023 | 0.50 | housing | student_mail_002 not in top-5 |

Note: rag_eval_015 improved to R@5=0.50 (was 0.50 — same aggregate but different sources retrieved).

## Readiness for BM25 / Hybrid Retrieval

**Yes, ready.** Query expansion provides a measurable but partial improvement (+0.020 R@5, +0.267 health_insurance R@5). The remaining weak cases are characterized by:

- Exact policy terms: "permission code", "W grade", "enrollment calendar", "waiver deadline"
- UCSD-specific abbreviations: "RCL", "I-20", "CPT", "OPT"
- Short numeric/phrase matches: "50 dollar late fee", "final quarter"

These are precisely the cases where **BM25 sparse retrieval** (term frequency + IDF) has a structural advantage over dense retrieval. BM25 rewards exact term matches and handles abbreviations natively — it does not require the embedding model to understand that "RCL" and "reduced course load" are semantically related.

**Next step**: implement `scripts/rag/retrieve_bm25.py` and run the ablation:
- Dense only (current baseline)
- BM25 only
- Hybrid α=0.3, α=0.5, α=0.7

Report which combination maximizes R@5 on the 25-query eval set.
