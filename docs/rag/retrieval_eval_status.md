# Retrieval Evaluation Status

## Metrics Comparison

| Metric | 10-query (v0) | 25-query (v1) |
|---|---|---|
| Queries | 10 | 25 |
| Recall@1 | 0.650 | 0.460 |
| Recall@3 | 0.733 | 0.653 |
| Recall@5 | 0.850 | 0.720 |
| Keyword Hit Rate | 0.833 | 0.737 |
| Total eval time | ~30s | 0.21s |
| Avg per query | ~3000ms | 8.3ms |

The 10-query set was overfit to easy cases — the 25-query results are the reliable baseline.

## Timing Improvement

**Before:** `eval_retrieval.py` called `retrieve()` for each query, which reloaded the embedding model, FAISS index, and chunk metadata on every call. 10 queries ≈ 30 seconds.

**After:** Model, index, and metadata load once. All queries encoded in a single batched call to `model.encode()`. FAISS `index.search()` runs on the full batch. 25 queries ≈ 0.21 seconds (**~360× faster**).

## Per-Category Recall@5

| category | n | R@1 | R@3 | R@5 | KW% |
|---|---|---|---|---|---|
| international_students | 5 | 0.667 | 0.833 | 0.900 | 0.783 |
| course_enrollment | 5 | 0.267 | 0.467 | 0.533 | 0.550 |
| health_insurance | 5 | 0.167 | 0.267 | 0.267 | 0.860 |
| student_health | 5 | 0.800 | 1.000 | 1.000 | 0.750 |
| housing | 5 | 0.400 | 0.700 | 0.900 | 0.743 |

**Strong categories:** student_health (R@5=1.00), housing (0.90), international_students (0.90)

**Weak categories:** health_insurance (R@5=0.27), course_enrollment (0.53)

## Weak Cases (R@5 < 1.0)

### Critical failures (R@5 = 0.00)

| id | query summary | expected | diagnosis |
|---|---|---|---|
| rag_eval_014 | permission code for full class | ucsd_registrar_add_drop_001 | "permission code" vocabulary not in source text |
| rag_eval_015 | UC SHIP waiver criteria | ucsd_ucship_waiver_002/003 | waiver_002/003 pages are short and not matched |
| rag_eval_018 | F-1 student + UC SHIP required? | ucsd_ucship_waiver_001/002 | F-1 / international vocabulary pulls immunization pages |

### Partial failures (0 < R@5 < 1.0)

| id | R@5 | issue |
|---|---|---|
| rag_eval_004 | 0.50 | "reduced course load" query misses visa_status_005; OPT pages outrank it |
| rag_eval_006 | 0.33 | waiver_001 is retrieved but waiver_002/003 are too short to score high |
| rag_eval_008 | 0.67 | enrollment calendar lacks "W grade" vocabulary |
| rag_eval_011 | 0.50 | "add course" query misses registrar_add_drop_002 (enrollment calendar) |
| rag_eval_013 | 0.50 | "grade option" / "Pass/No Pass" vocabulary partially present |
| rag_eval_016 | 0.50 | waiver_004 (late fee page, 137 words) is too short to rank in top-5 |
| rag_eval_017 | 0.50 | coverage_001 retrieved but not coverage_002 |
| rag_eval_023 | 0.50 | "mailroom hours" query misses student_mail_002 |

## Root Cause Analysis

Two distinct failure modes:

**1. Source content is too thin** (most impactful):
- `ucsd_ucship_waiver_002/003`: 639/818 words but content is narrow; waiver criteria details seem missing
- `ucsd_ucship_waiver_004`: only 137 words (late fee page)
- `ucsd_registrar_add_drop_002`: enrollment calendar; lacks "W grade" text
- Fix: source strengthening (manually check raw .txt content; re-scrape or add supplemental content)

**2. Vocabulary mismatch** (retrieval algorithm limitation):
- "permission code" → registrar pages talk about "authorization codes" (partial match)
- "F-1 visa" + "insurance" → immunization pages rank above waiver pages
- "reduced course load" → OPT pages outrank visa_status_005
- Fix: query expansion with UCSD-specific synonyms (opt 4)

## Are We Ready for Chunking Ablation?

**Not yet.** Health_insurance R@5=0.27 is likely a source content problem, not a chunking problem. Running chunking ablation now would not reveal the true chunking signal — the floor is set by missing source content.

**Recommended order before chunking ablation:**

1. Fix weak sources (opt 3): check `ucsd_ucship_waiver_002/003/004` raw text; re-scrape or manually expand `ucsd_registrar_add_drop_002`; check `ucsd_iseo_visa_status_005` content completeness
2. Add query expansion (opt 4): synonym dict for "permission code"/"authorization code", "RCL"/"reduced course load", "F-1"/"international student"
3. Then run chunking ablation (opt 2): with better source content, ablation results will be informative

## Open-Source References

The following projects were reviewed as design references. No code was copied.

**RAGFlow** (InfiniFlow/ragflow)
Useful reference for chunking and retrieval evaluation design, especially the structured approach to document parsing. RAGFlow's DeepDoc handles complex PDFs with layout detection — not central to this project since UCSD sources are mostly plain HTML/text scraped from official webpages. The chunking evaluation methodology informed our ablation script design.

**Langchain-Chatchat** (chatchat-space/Langchain-Chatchat)
Useful reference for BM25 + dense vector hybrid retrieval design (the `hybrid_search` module). Their approach of score normalization before combining BM25 and dense scores informed the planned hybrid retrieval implementation (opt 6).

**QAnything** (netease-youdao/QAnything)
Useful reference for two-stage retrieval: retrieve top-20 with hybrid search, then rerank to top-5 with a cross-encoder. The reranker stage is not implemented yet in this project (planned as opt 8), but the QAnything architecture shows the expected latency vs. recall trade-off clearly.
