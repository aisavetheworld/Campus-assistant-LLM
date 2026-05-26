# Chunking Ablation Report

## 1. Experiment Setup

**Corpus:** 55 UCSD official source pages, 395 chunks (at 512/50 baseline)
**Eval set:** 25 queries across 5 categories (`data/rag/rag_eval_seed.json`)
**Goal:** Find the best chunk_size / overlap for dense retrieval on UCSD official-page text

## 2. Controlled Variables

| Variable | Value |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Index type | FAISS IndexFlatIP (cosine on normalized vectors) |
| Source dedup | Enabled — each source_id contributes at most 1 chunk to top-k |
| top_k | 5 |
| Eval queries | 25 |

## 3. Result Table

| config (size/overlap) | chunks | avg words/chunk | R@1 | R@3 | R@5 | KW% |
|---|---|---|---|---|---|---|
| 256/50 | 494 | 132 | 0.480 | 0.740 | **0.773** | 0.769 |
| **512/50** (baseline) | 395 | 152 | 0.460 | **0.753** | **0.773** | 0.761 |
| 512/100 | 399 | 158 | **0.480** | **0.753** | **0.773** | **0.769** |
| 1024/100 | 361 | 164 | 0.460 | **0.753** | **0.773** | **0.769** |

## 4. Per-Category R@5

| category | 256/50 | 512/50 | 512/100 | 1024/100 |
|---|---|---|---|---|
| international_students | 1.000 | 1.000 | 1.000 | 1.000 |
| course_enrollment | 0.533 | 0.533 | 0.533 | 0.533 |
| health_insurance | 0.433 | 0.433 | 0.433 | 0.433 |
| student_health | 1.000 | 1.000 | 1.000 | 1.000 |
| housing | 0.900 | 0.900 | 0.900 | 0.900 |

**Per-category R@5 is identical across all 4 configs.**

## 5. Weak Case Analysis

The same 10 queries fail across all configs. Chunking does not resolve any of them.

| id | category | R@5 | root cause |
|---|---|---|---|
| rag_eval_006 | health_insurance | 0.67 | waiver_002 not in top-5 — outranked by coverage/calendar pages |
| rag_eval_008 | course_enrollment | 0.67 | add_drop_002 (enrollment calendar) lacks "W grade" vocabulary |
| rag_eval_011 | course_enrollment | 0.50 | add_drop_002 missing; immunization/OPT pages outrank it |
| rag_eval_013 | course_enrollment | 0.50 | "grade option"/"Pass No Pass" — add_drop_002 not retrieved |
| rag_eval_014 | course_enrollment | 0.00 | "permission code" query retrieves student mail pages; phrase absent from add_drop source |
| rag_eval_015 | health_insurance | 0.50 | "ACA criteria" vocabulary in waiver_002/003 doesn't match query phrasing |
| rag_eval_016 | health_insurance | 0.50 | "late fee" query matches OPT/calendar pages; waiver_004 (137 words) too thin |
| rag_eval_017 | health_insurance | 0.50 | "mental health counseling" retrieves use/maf pages; coverage_002 missing |
| rag_eval_018 | health_insurance | 0.00 | "F-1 visa + insurance required" retrieves ISEO visa pages; waiver_002 missing |
| rag_eval_023 | housing | 0.50 | "mailroom hours" retrieves mailroom pages but student_mail_002 not in top-5 |

### Failure pattern classification

**Vocabulary mismatch (query expansion would fix):**
- rag_eval_014: "permission code" → source uses "authorization code"
- rag_eval_018: "F-1 visa" + "insurance" → retriever routes to ISEO visa pages
- rag_eval_008/011/013: "W grade", "grade option", "authorization" → weak match to enrollment calendar

**Source content too thin or narrow:**
- rag_eval_016: waiver_004 is 137 words; "late fee" content is minimal
- rag_eval_006/015: waiver_002/003 have content but retrieval score is consistently 4th-5th

**Multi-source coverage requirement:**
- rag_eval_006, 008, 011, 013: expected 2–3 sources simultaneously; top-5 can only cover a subset when sources score similarly

## 6. Final Chosen Config

**Keep 512/50 as default** (current baseline).

Rationale:
- R@5 is flat (0.773) across all 4 configs — chunking is not the bottleneck
- 512/100 has the same R@3/R@5 with a marginal R@1 advantage (+0.02), but the difference is not statistically meaningful on 25 queries
- 512/50 produces fewer chunks than 256/50 (395 vs 494), keeping the index lean
- 1024/100 has fewer chunks (361) but loses R@1 with no R@5 gain

The UCSD source material (official HTML pages, accordion sections, short procedural pages) does not benefit from larger chunk windows because the pages themselves are short and mostly already fit within one 512-word chunk.

## 7. Recommendation for Next Step

**Chunking is not the bottleneck. The 10 weak cases are caused by vocabulary mismatch, not chunk granularity.**

Two complementary approaches to address the remaining weak cases:

### Option A — Query Expansion (opt 4)

Maintain a UCSD-specific synonym dictionary:
```json
{
  "permission code": ["authorization code", "instructor approval"],
  "W grade": ["withdrawal", "withdraw", "drop after deadline"],
  "grade option": ["Pass/No Pass", "S/U", "letter grade change"],
  "F-1 visa": ["international student", "F-1 status"],
  "reduced course load": ["RCL", "drop below full-time", "fewer units"],
  "UC SHIP waiver": ["health insurance waiver", "waive insurance"]
}
```

**Expected impact:** rag_eval_014, 018, 008, 013 are direct vocabulary mismatches — expansion should resolve them.

### Option B — BM25 + Hybrid Retrieval (opt 5 + 6)

BM25 rewards exact term matches, which would help:
- "permission code" (exact phrase in source text)
- "W grade", "SEVIS", "I-20", "CPT", "OPT" (policy abbreviations)
- "late fee", "50 dollar" (specific numeric/phrase matches)

Dense retrieval handles semantic matching; BM25 handles exact policy terms. Hybrid alpha ∈ {0.3, 0.5, 0.7} ablation would quantify the gain.

### Recommended order

1. Query expansion first (no index rebuild needed, directly measurable on current eval set)
2. BM25 + hybrid retrieval (requires rank_bm25, score normalization, alpha ablation)
3. Re-run 25-query eval after each step to verify improvement

**Current ceiling with dense-only retrieval: R@5 = 0.773.** The 10 weak cases require either better source coverage or a different retrieval signal. Chunking ablation has confirmed this is a vocabulary/signal problem, not a granularity problem.
