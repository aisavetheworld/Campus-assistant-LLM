# Project 2 RAG — Retrieval Stage Summary

## Status: Complete

Retrieval pipeline finalized. Moving to grounded answer generation.

---

## Final Configuration

| Component | Value |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Index | FAISS IndexFlatIP (cosine, L2-normalized, dim=384) |
| Chunking | 512-word sliding window, 50-word overlap |
| BM25 | rank_bm25 BM25Okapi |
| Hybrid weight | α=0.7 (dense-dominant) |
| Query expansion | Enabled — 15 trigger keys, `configs/rag_query_expansion.json` |
| Source dedup | Enabled — one chunk per source_id in top-k |
| top_k | 5 |
| Corpus | 66 sources, 412 chunks, 5 categories |

---

## Final Eval Results (50-query set, hybrid α=0.7 + QE)

| Metric | Value |
|---|---|
| R@1 | 0.687 |
| R@3 | 0.950 |
| **R@5** | **1.000** |
| Keyword hit rate | 0.888 |
| Weak cases | 0 / 50 |

Per-category R@5: all 5 categories at 1.000 (international_students, course_enrollment, health_insurance, student_health, housing).

Generalization confirmed: config was tuned on 25 queries and held at R@5=1.000 on 25 new held-out queries with diverse styles (casual, formal, abbreviation-heavy, cross-category, paraphrase).

---

## Optimization History

| Stage | R@5 | Notes |
|---|---|---|
| Dense baseline, 10 queries | 0.850 | Overfit — only easy queries |
| Dense + source dedup, 25 queries | 0.773 | True baseline |
| + 7 UC SHIP sources | 0.773 | Better KW%, no R@5 gain |
| + Query expansion | 0.793 | health_insurance: 0.267 → 0.533 |
| + Hybrid α=0.7 | 0.853 | BM25 fixed permission code, criteria queries |
| + 10 registrar grades sources | 0.837 | Regression from BM25 vocab confusion |
| + QE tuning + eval seed audit | 0.877–0.960 | Removed bad expansion terms; corrected 5 wrong expected sources |
| + Final QE additions | 1.000 | "can I waive", "qualify", "authorization code" triggers |
| Expanded to 50 queries | **1.000** | Generalization confirmed |

---

## Why Query Expansion Is Load-Bearing

Without QE, hybrid α=0.7 has 5 failures on 50 queries:

| Query | Failure without QE | QE trigger that fixes it |
|---|---|---|
| rag_eval_006 — SHIP waiver deadline + qualifying insurance | waiver_002 not retrieved | `qualify` → waiver criteria vocabulary |
| rag_eval_013 — P/NP grade change | add_drop_001 missing | `Pass/No Pass` → grade option vocabulary |
| rag_eval_014 — permission code from professor | ISEO/CPT pages returned | `permission code` → instructor/department approval |
| rag_eval_015 — SHIP waiver criteria | waiver_003 not retrieved | `waiver criteria` → comparable coverage terms |
| rag_eval_018 — F-1 student, required to have SHIP? | ISEO visa pages returned instead of waiver pages | `F-1 visa` → UC SHIP/insurance waiver |

QE is not overfit to dev queries — it fires on novel phrasing and prevents cross-category BM25 confusion.

---

## Why No Reranker (opt8)

A cross-encoder reranker would re-score the hybrid top-20 and return top-5.

**Not added because:**

1. **R@5 is already 1.000.** All expected sources appear in top-5 for all 50 queries. A reranker can only improve ranking order within the already-correct set — it cannot raise R@5 above 1.000.

2. **Remaining gap is R@3/R@1, not R@5.** R@3=0.950, R@1=0.687. This means some relevant sources rank 4th or 5th instead of 1st. This affects which source appears most prominently in the generated answer, not whether the answer has factual coverage. This is a generation-quality issue, not a retrieval-coverage issue.

3. **Latency cost is real.** Cross-encoder rerankers (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) add 300–500ms per query on CPU. For Project 3 serving, this is significant.

4. **Source content gaps were the bottleneck, not ranking.** During development, most failures were caused by vocabulary gaps in source pages (e.g., add_drop_002 having no content, waiver_004 being about ACA reporting). A reranker cannot rescue sources that aren't retrieved at all — content quality is the root cause.

**When to revisit:** If answer quality evaluation (Phase 2) shows that the model consistently uses the 4th/5th-ranked source instead of the most relevant one, a reranker would help R@1/R@3. Add it then, not now.

---

## Why No Category Routing (opt7)

Keyword-based category routing would pre-filter the corpus to a predicted category before retrieval (e.g., detect "UC SHIP" → restrict to health_insurance sources only).

**Not added because:**

1. **QE already solves the cross-category confusion that motivated routing.** The original motivation was:
   - F-1 visa queries pulling ISEO pages instead of SHIP waiver pages → fixed by `F-1 visa` QE expansion
   - "late fee" query pulling enrollment calendar → fixed by removing "waiver deadline"/"fee deadline" from QE
   - "authorization code" pulling ISEO authorization pages → fixed by `authorization code` QE trigger

   All 5 cross-category failures in hybrid α=0.7 (no QE) are fixed by QE alone. There is no residual cross-category problem to route around.

2. **Category routing would break legitimate cross-category queries.** Several eval queries intentionally span categories:
   - rag_eval_029: CPT (international) + 12-unit rule (enrollment)
   - rag_eval_035: drop deadline (enrollment) + W grade definition (grades)
   - rag_eval_050: housing eligibility (housing) + full-time enrollment (enrollment/F-1)

   A hard category filter would force a single-category retrieval for these queries, reducing R@5.

3. **R@5=1.000 leaves no problem for routing to solve.** Category routing is an optimization for when dense retrieval fails on cross-category queries. That failure mode no longer exists.

**When to revisit:** If the corpus grows significantly (e.g., 500+ sources) and BM25 begins returning noisy matches from distant categories, routing becomes valuable. At 412 chunks / 66 sources, the signal-to-noise ratio is manageable without it.

---

## Key Lessons from Retrieval Development

1. **Source dedup is essential.** Before dedup, a single high-scoring source (e.g., ucsd_ucship_waiver_001) monopolized 3 of 5 top-k slots. Source dedup improved R@3 by ~0.10.

2. **BM25 alone degrades every category** (R@5=0.710 vs dense 0.937). The semantic gap between student queries and formal policy register is too large for keyword matching alone. Dense retrieval is load-bearing.

3. **BM25 "authorization" is overloaded.** Three different document types use "authorization": ISEO visa pages (CPT/OPT authorization), grade-change pages (Enrollment Authorization System EASy), and enrollment pages. QE must carefully avoid terms that trigger this ambiguity.

4. **Eval seed integrity matters.** 5 of the original 25 expected_source_ids pointed to sources whose content did not actually answer the query (e.g., waiver_004 = ACA reporting, not late fees; coverage_002 = billing notice, not coverage). These were discovered by reading raw source text. Correcting them was necessary before the eval was meaningful.

5. **Chunking is not the bottleneck.** Ablation across 256/50, 512/50, 512/100, 1024/100 showed identical R@5=0.773. The model's embedding quality and hybrid scoring matter far more than chunk size at this corpus scale.

---

## Files

| File | Purpose |
|---|---|
| `scripts/rag/retrieve_hybrid.py` | Production retrieval function |
| `scripts/rag/ablation_hybrid.py` | Eval runner (all configs, all queries) |
| `scripts/rag/query_expansion.py` | QueryExpander class |
| `configs/rag_query_expansion.json` | 15 QE trigger keys |
| `data/rag/rag_eval_seed.json` | 50-query eval set |
| `data/rag/vector_store/` | FAISS index + chunk metadata (412 vectors) |
| `outputs/rag_eval/hybrid_ablation_raw.json` | Latest eval raw results |
| `docs/rag/hybrid_retrieval_report.md` | Full ablation report with alpha trade-offs |
| `docs/rag/retrieval_eval_50q_report.md` | 50-query generalization report |

---

## Next: Grounded Answer Generation

Run `scripts/rag/rag_answer.py` end-to-end. Evaluate:
- Answer groundedness (claims traceable to retrieved chunks)
- Safety adherence (no invented dates/fees/deadlines, appropriate escalation to ISEO/SHS/HDH)
- Citation accuracy (cited source actually contains the stated information)
- Handling of low-confidence retrievals (score < threshold → safe escalation message)
