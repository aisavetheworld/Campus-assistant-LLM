# Reranker Comparison Report — Project 2 RAG

**Date:** 2026-05-28
**Eval set:** 95 queries, 7 categories
**Decision:** **Do not use a reranker as default. Keep hybrid direct top-5.**

---

## TL;DR

| Config | R@5 | Δ vs baseline | Latency/q | Verdict |
|--------|-----|---------------|-----------|---------|
| **Hybrid direct (baseline)** | **0.837** | — | 32 ms (CPU) | ✓ Keep as default |
| + MiniLM rerank top-20 | 0.708 | −12.9 pp | 298 ms (CPU) | ✗ Hurts |
| + MiniLM rerank top-50 | 0.676 | −16.1 pp | 543 ms (CPU) | ✗ Worse |
| + BGE-v2-m3 rerank top-20 | 0.711 | −12.6 pp | 1054 ms (L4 GPU) | ✗ Hurts |
| + BGE-v2-m3 rerank top-50 | 0.672 | −16.5 pp | 2130 ms (L4 GPU) | ✗ Worse |

All rerankers tested actively **degrade** R@5 on this corpus. The candidate pool from hybrid retrieval is already excellent (cR@20 = 0.958) — the bottleneck is not retrieval recall, it's that the reranker's ranking signal is *worse* than the hybrid score for this domain.

---

## Background

### Bi-encoder vs Cross-encoder

Current retrieval pipeline = **bi-encoder + sparse**:
- Dense: `all-MiniLM-L6-v2` encodes queries and chunks independently → cosine similarity
- Sparse: BM25Okapi
- Hybrid: `alpha · dense + (1 − alpha) · BM25`, alpha = 0.8

Bi-encoders precompute chunk embeddings once. At query time only the query is encoded → fast (O(1) per query for FAISS lookup). The model never sees the query and chunk together.

A **cross-encoder reranker** encodes the (query, chunk) pair *jointly* through a transformer, capturing fine-grained interactions a bi-encoder cannot. The cost: every (query, chunk) pair requires a full transformer forward pass.

Two-stage retrieval = bi-encoder for recall (cheap), cross-encoder for precision (expensive):

```
query → hybrid top-N candidates → cross-encoder rerank → top-5
```

**A reranker can only reorder. It cannot recover sources missing from candidate top-N.**

### Why BGE-reranker-v2-m3 was tested

`cross-encoder/ms-marco-MiniLM-L-6-v2` is trained on **MS MARCO** (short web-search queries, web snippets). UCSD queries differ:
- Institutional terminology: UC SHIP, SEVIS, F-1, CPT, OPT, HDH, ISEO
- Multi-sentence scenario questions: *"I had a family emergency during finals week..."*
- Long policy passages (waiver criteria, enrollment deadlines)

`BAAI/bge-reranker-v2-m3` is multilingual, trained on diverse corpora (BEIR / MTEB), and handles longer formal-register text. It was the strongest candidate for institutional-domain rerank.

---

## Corpus and Eval Setup

| Parameter | Value |
|-----------|-------|
| Corpus | 673 sources, 4098 chunks |
| Chunking | 512 words / 50 word overlap |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse | BM25Okapi |
| Hybrid alpha | 0.8 (tuned via grid search) |
| Query expansion | Enabled (18 triggers, word-boundary regex) |
| Reranker uses | original query (not expanded) for (q, chunk) scoring |
| Eval set | 95 queries, 7 categories |
| Final top-k | 5 |

---

## Candidate Pool Analysis (the upper bound)

Before any reranking, how often does the expected source appear in hybrid top-N?

| Metric | Value |
|--------|-------|
| Candidate R@20 | **0.958** |
| Candidate R@50 | **0.988** |
| Zero-recall @5 | 3 |
| Zero-recall @20 | **0** |
| Zero-recall @50 | **0** |

**All 3 zero-recall@5 cases have their expected source in the top-20 candidates.** A *perfect* reranker on top-20 candidates would achieve R@5 ≤ 0.958. So the theoretical ceiling for rerank is ~+0.12 over baseline (0.837 → 0.958). Instead, every reranker tested moved us *backward* by 0.12–0.17.

---

## Results

| Config | R@1 | R@3 | R@5 | MRR | nDCG@5 | cR@20 | cR@50 | Zero@5 | Time/q |
|--------|-----|-----|-----|-----|--------|-------|-------|--------|--------|
| Hybrid direct (baseline) | 0.266 | 0.688 | **0.837** | **0.621** | **0.626** | 0.958 | 0.988 | **3** | **33 ms** (CPU) |
| + MiniLM rerank top-20 | **0.342** | 0.593 | 0.708 | 0.580 | 0.577 | 0.958 | 0.958 | 15 | 298 ms (CPU) |
| + MiniLM rerank top-50 | 0.325 | 0.591 | 0.676 | 0.557 | 0.553 | 0.958 | 0.988 | 19 | 543 ms (CPU) |
| + BGE-v2-m3 rerank top-20 | 0.272 | 0.566 | 0.711 | 0.534 | 0.544 | 0.958 | 0.958 | 16 | 1054 ms (L4) |
| + BGE-v2-m3 rerank top-50 | 0.312 | 0.562 | 0.672 | 0.535 | 0.535 | 0.958 | 0.988 | 20 | 2130 ms (L4) |

**Best metric in each column bolded.** Baseline wins R@5, R@3, MRR, nDCG@5, zero-recall, and latency. MiniLM wins R@1 only.

### Per-Category R@5

| Category | n | Baseline | MiniLM top-20 | Δ |
|----------|---|---------|---------------|---|
| course_enrollment | 15 | 0.817 | 0.694 | −0.123 |
| financial_aid | 10 | 0.800 | 0.617 | −0.183 |
| graduate_students | 10 | 0.817 | 0.667 | −0.150 |
| health_insurance | 15 | 0.833 | 0.739 | −0.094 |
| housing | 15 | 0.806 | 0.628 | −0.178 |
| international_students | 15 | 0.767 | 0.700 | −0.067 |
| student_health | 15 | **1.000** | 0.867 | −0.133 |

**Every category gets worse**, including the previously-perfect `student_health` (1.00 → 0.87). No category is rescued by reranking.

### Zero-recall cases (the 3 stubborn queries)

The 3 baseline zero-recall@5 queries:
- `rag_eval_006` (course_enrollment): drop without W deadline — academic calendar page not in corpus
- `rag_eval_024` (course_enrollment / housing): dorm cost
- `rag_eval_094` (student_health): SHS cost

All 3 have expected source in candidate top-20. **No reranker recovered any of them.** They remained zero-recall after reranking, while the rerankers pushed *other* correct sources out of top-5 — explaining why zero-recall@5 *grew* from 3 → 15–20.

---

## Why Rerankers Hurt Here

1. **Candidate pool is already saturated.** cR@20 = 0.958 means hybrid retrieval is doing 95.8% of the recall work. There's almost nothing for a reranker to "rescue."
2. **Hybrid ranking signal is strong.** Dense + BM25 + query expansion is well-tuned for this corpus (alpha grid search, hyphen-aware QE). Re-scoring with an off-the-shelf cross-encoder discards this carefully tuned signal.
3. **Domain mismatch dominates.** Both MiniLM (web QA) and BGE (general/multilingual) were trained on text very different from UCSD policy/admin pages. They penalize chunks dense in institutional acronyms and formal policy language because such chunks don't look like their training distribution.
4. **Long chunks confuse the reranker.** 512-word chunks exceed the typical MS MARCO passage length; cross-encoder attention may degrade on longer inputs.
5. **R@1 improves slightly** with MiniLM (0.266 → 0.342). When *one* chunk is overwhelmingly obvious, the reranker finds it. But across positions 2–5, it loses more than it gains.
6. **More candidates → worse, not better.** top-50 underperforms top-20 for both rerankers, because giving a domain-mismatched scorer more candidates amplifies its errors.

---

## Latency Tradeoff

| Stage | Hardware | Time/q |
|-------|----------|--------|
| Hybrid direct top-5 | CPU | 33 ms |
| + MiniLM rerank top-20 | CPU | 298 ms (9× slower) |
| + MiniLM rerank top-50 | CPU | 543 ms (16× slower) |
| + BGE rerank top-20 | L4 GPU | 1054 ms (32× slower) |
| + BGE rerank top-50 | L4 GPU | 2130 ms (65× slower) |

For a serving context (Project 3), even acceptable rerank latency cannot justify a 12 pp R@5 drop.

---

## Recommendation

### Default config (unchanged)

```
hybrid (alpha=0.8) + query expansion + dedup-by-source
```

R@5 = 0.837. ~33 ms/query on CPU. **This is the production default.**

### Do not enable rerank by default

Both MiniLM and BGE rerankers degrade R@5 on this corpus. Off-the-shelf cross-encoders are not the right tool here.

### When a reranker *would* help

A reranker would help if:
- Candidate cR@N were low (e.g. cR@20 < 0.85) — not our case
- We had a domain-fine-tuned cross-encoder (e.g. fine-tuned on UCSD QA pairs) — out of scope
- We wanted R@1 specifically (only one slot returned), and were willing to lose R@5 — not our use case (RAG generation benefits from multiple diverse sources)

### To improve R@5 further (alternatives to rerank)

The 3 zero-recall@5 queries all fail because the corresponding source is *missing or weak in the corpus*, not because retrieval is bad:
- Add UCSD academic calendar pages (drop deadlines, W-grade week boundaries)
- Add HDH housing cost / rate pages (currently weak coverage)
- Add SHS pricing / fee schedule pages

Expanding the corpus to cover these gaps would yield more R@5 gain than any reranker.

---

## Reproducibility

### Local (CPU, MiniLM + baseline)

```bash
# Baseline
python scripts/rag/eval_retrieval.py --hybrid --use_query_expansion --alpha 0.8 \
    --report_suffix baseline_candidate

# MiniLM rerank top-20
python scripts/rag/eval_retrieval.py --hybrid --use_query_expansion --alpha 0.8 \
    --candidate_k 20 --rerank \
    --reranker_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --reranker_backend crossencoder \
    --report_suffix rerank_minilm_top20
```

### Colab (L4 GPU, BGE-v2-m3)

```python
%cd /content/drive/MyDrive/Campus-assistant-LLM
!git pull

!python scripts/rag/eval_retrieval.py --hybrid --use_query_expansion --alpha 0.8 \
    --candidate_k 20 --rerank \
    --reranker_model BAAI/bge-reranker-v2-m3 \
    --reranker_backend crossencoder \
    --report_suffix rerank_bge_top20
```

Note: BGE via `--reranker_backend crossencoder` (sentence-transformers ≥2.7) is more reliable than `flag` (FlagEmbedding hits a tokenizer compatibility issue with newer transformers).

### Reports written to

```
outputs/rag_eval/retrieval_eval_report_baseline_candidate.{md,json}
outputs/rag_eval/retrieval_eval_report_rerank_minilm_top20.{md,json}
outputs/rag_eval/retrieval_eval_report_rerank_minilm_top50.{md,json}
outputs/rag_eval/retrieval_eval_report_rerank_bge_top20.{md,json}
outputs/rag_eval/retrieval_eval_report_rerank_bge_top50.{md,json}
```
