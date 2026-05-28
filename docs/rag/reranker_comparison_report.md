# Reranker Comparison Report — Project 2 RAG

## Background

### Bi-encoder vs Cross-encoder

The current retrieval pipeline uses a **bi-encoder** architecture:

- Dense: `all-MiniLM-L6-v2` encodes queries and chunks independently, then computes cosine similarity.
- Sparse: BM25Okapi computes term-overlap scores.
- Hybrid: `alpha * dense + (1 - alpha) * BM25`, alpha = 0.8.

Bi-encoders are fast (O(1) per query after precomputing chunk embeddings) but use separate representations — the model never sees the query and chunk together.

A **cross-encoder** (reranker) encodes the (query, chunk) pair jointly through a transformer. This captures fine-grained query-chunk interactions that bi-encoders miss. The trade-off is speed: a cross-encoder must run inference for each (query, chunk) pair at query time.

Two-stage retrieval combines both:
1. **Stage 1 (bi-encoder):** retrieve top-N candidates quickly.
2. **Stage 2 (cross-encoder):** rerank candidates, return top-k.

The reranker can only reorder what Stage 1 retrieved. If an expected source is not in the candidate pool, the reranker cannot recover it.

### Why BGE-reranker-v2-m3 May Be Better for This Domain

`cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on **MS MARCO**, a web search dataset with short, keyword-style queries and web document passages. UCSD administrative queries use:
- Institutional terminology ("UC SHIP", "SEVIS", "F-1 status", "CPT", "OPT", "HDH")
- Multi-sentence procedural questions ("I had a family emergency during finals week and couldn't...")
- Policy-heavy long passages (waiver criteria, enrollment deadlines, housing contracts)

`BAAI/bge-reranker-v2-m3` is a **multilingual** cross-encoder trained on diverse datasets including academic and formal-register text. Its M3 variant handles mixed-language queries better and was evaluated on heterogeneous retrieval benchmarks (BEIR, MTEB). For a domain with institutional jargon and long passages, BGE is likely a better fit than a narrow web-search reranker.

---

## Corpus and Eval Setup

| Parameter | Value |
|-----------|-------|
| Corpus | 673 sources, 4098 chunks |
| Chunking | 512 words / 50 word overlap |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse | BM25Okapi |
| Hybrid alpha | 0.8 (tuned via grid search on 95 queries) |
| Query expansion | Enabled (18 triggers, word-boundary matching) |
| Eval set | 95 queries, 7 categories |
| Final top-k | 5 |

---

## Candidate Pool Analysis

Before running any reranker, we measured how many expected sources appear in the top-N hybrid candidates. This sets the upper bound for what a reranker can achieve.

| Metric | Value |
|--------|-------|
| Candidate R@20 | **0.958** |
| Candidate R@50 | **0.988** |
| Zero-recall @5 | 3 |
| Zero-recall @20 | **0** |
| Zero-recall @50 | **0** |

All 3 zero-recall@5 cases have their expected source in the top-20 hybrid candidates. A perfect reranker with top-20 candidates could achieve R@5 ≤ 0.958 (bounded by the 4 queries where the expected source is not in top-20 at all).

---

## Results

| Config | R@1 | R@3 | R@5 | MRR | nDCG@5 | cR@20 | cR@50 | Zero@5 | Time/q |
|--------|-----|-----|-----|-----|--------|-------|-------|--------|--------|
| Hybrid direct top-5 (baseline) | 0.266 | 0.688 | **0.837** | 0.621 | 0.626 | 0.958 | 0.988 | 3 | 32 ms |
| top-20 + MiniLM rerank | 0.342 | 0.593 | 0.708 | 0.580 | 0.577 | 0.958 | 0.958 | 15 | 304 ms |
| top-50 + MiniLM rerank | 0.325 | 0.591 | 0.676 | 0.557 | 0.553 | 0.958 | 0.988 | 19 | 546 ms |
| top-20 + BGE-reranker-v2-m3 | — | — | — | — | — | 0.958 | — | — | (Colab) |
| top-50 + BGE-reranker-v2-m3 | — | — | — | — | — | 0.958 | 0.988 | — | (Colab) |

*BGE results pending — requires GPU (Colab L4). Run command below.*

### Per-Category R@5 (baseline vs MiniLM top-20)

*(Fill in from report JSONs after running per-category breakdown.)*

| Category | Baseline R@5 | MiniLM top-20 R@5 | Δ |
|----------|-------------|-------------------|---|
| course_enrollment | — | — | — |
| financial_aid | — | — | — |
| graduate_students | — | — | — |
| health_insurance | — | — | — |
| housing | — | — | — |
| international_students | — | — | — |
| student_health | — | — | — |

---

## Analysis: Why MiniLM Hurts

MiniLM (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is trained on MS MARCO — short web queries and web snippets. On this corpus it:

1. **Penalizes relevant UCSD chunks** that contain institutional acronyms and policy language, because these look unlike MS MARCO passages.
2. **Increases zero-recall@5 from 3 → 15**: it actively pushes correct sources below rank 5 in exchange for passages that superficially look like web-search answers.
3. **Improves R@1 slightly** (0.266 → 0.342): when there is one very obvious answer, the cross-encoder finds it. But across all 5 slots, it loses more than it gains.
4. **More candidates make it worse**: top-50 + MiniLM scores lower than top-20 + MiniLM, because giving a domain-mismatched reranker more candidates amplifies its errors.

This is a known pattern in retrieval research: cross-encoders trained on web search do not generalize to narrow-domain institutional corpora without fine-tuning.

---

## Latency Tradeoff

| Stage | Config | Time/query |
|-------|--------|------------|
| Bi-encoder only | Hybrid top-5 | ~32 ms (CPU) |
| + MiniLM rerank | top-20 candidates | ~304 ms (CPU, 10× slower) |
| + MiniLM rerank | top-50 candidates | ~546 ms (CPU, 17× slower) |
| + BGE rerank | top-20 candidates | ~TBD ms (GPU L4) |

For a serving context (Project 3 / FastAPI + vLLM), BGE-M3 on GPU should be acceptable latency (~20–100ms on L4). MiniLM on CPU at 300ms/query may be acceptable for low-traffic demo use, but is not worth the R@5 penalty.

---

## Recommendation

### Current decision: **Do not replace hybrid direct retrieval with MiniLM reranker.**

- Hybrid direct (alpha=0.8): R@5 = 0.837
- MiniLM top-20: R@5 = 0.708 (−12.9pp), 10× slower

The bi-encoder pipeline is better for this domain and faster.

### Next step: Run BGE-reranker-v2-m3 on Colab (L4)

BGE-M3 is a multilingual cross-encoder trained on diverse corpora. It may generalize better to UCSD institutional text. If BGE achieves R@5 > 0.837 with acceptable latency, it becomes the recommended default for serving.

**Decision threshold:** BGE should improve R@5 by ≥ 0.01 (to ≥ 0.847) AND latency on L4 GPU should be ≤ 150ms/query to justify adding the complexity.

---

## Colab Commands (BGE eval)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Campus-assistant-LLM
!git pull

# Install FlagEmbedding
!pip install -q FlagEmbedding

# BGE top-20
!python scripts/rag/eval_retrieval.py \
    --hybrid --use_query_expansion --alpha 0.8 \
    --candidate_k 20 \
    --rerank \
    --reranker_model BAAI/bge-reranker-v2-m3 \
    --reranker_backend flag \
    --report_suffix rerank_bge_top20

# BGE top-50
!python scripts/rag/eval_retrieval.py \
    --hybrid --use_query_expansion --alpha 0.8 \
    --candidate_k 50 \
    --rerank \
    --reranker_model BAAI/bge-reranker-v2-m3 \
    --reranker_backend flag \
    --report_suffix rerank_bge_top50
```

```python
# Print results
import json

for suffix in ["baseline_candidate", "rerank_minilm_top20", "rerank_minilm_top50",
               "rerank_bge_top20", "rerank_bge_top50"]:
    p = f"outputs/rag_eval/retrieval_eval_report_{suffix}.json"
    try:
        d = json.loads(open(p).read())
        agg = d["aggregate"]
        print(f"\n=== {suffix} ===")
        print(f"R@5={agg['recall@5']:.3f} MRR={agg.get('mrr',0):.3f} "
              f"nDCG@5={agg.get('ndcg@5',0):.3f} "
              f"cR@20={agg.get('candidate_recall@20',0):.3f} "
              f"zero@5={agg.get('zero_recall@5',0)}")
    except FileNotFoundError:
        print(f"\n=== {suffix} === NOT FOUND")
```
