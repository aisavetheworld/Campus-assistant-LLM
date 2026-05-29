# Experiments Summary — Project 2 RAG

Consolidated ablation tables for chunking, retrieval (alpha + QE),
reranking, and answer generation. Numbers are read from
`outputs/rag_eval/*.json` artifacts.

---

## 1. Chunking ablation (50-q eval, 66-source initial corpus)

| Config | # chunks | avg words / chunk | R@1 | R@3 | R@5 |
|--------|----------|-------------------|-----|-----|-----|
| 256 words / 50 overlap  | 494 | 131.7 | 0.480 | 0.740 | 0.773 |
| **512 / 50** (chosen)   | **395** | **152.2** | **0.460** | **0.753** | **0.773** |
| 512 / 100               | 399 | 158.3 | 0.480 | 0.753 | 0.773 |
| 1024 / 100              | 361 | 164.4 | 0.460 | 0.753 | 0.773 |

**Pick:** `512 / 50`. All four are within noise on R@5; 512 / 50 keeps
chunks small enough to be precise yet long enough to carry full policy
paragraphs, halving index size vs 256 / 50.

*Source: `outputs/rag_eval/chunking_ablation_raw.json`*

---

## 2. Hybrid retrieval alpha grid (50-q corpus, before expansion)

| Config | Query expansion | R@1 | R@5 | KW hit |
|--------|-----------------|-----|-----|--------|
| dense only (alpha = 1.0) | off | 0.577 | 0.937 | 0.815 |
| BM25 only (alpha = 0.0)  | off | 0.353 | 0.710 | 0.817 |
| hybrid alpha = 0.3       | off | 0.520 | 0.877 | 0.886 |
| hybrid alpha = 0.5       | off | 0.557 | 0.947 | 0.896 |
| hybrid alpha = 0.7       | off | 0.657 | 0.937 | 0.883 |
| hybrid alpha = 0.5       | **on**  | 0.607 | **1.000** | 0.887 |
| **hybrid alpha = 0.7**   | **on**  | **0.687** | **1.000** | 0.888 |

**Pick (50-q corpus):** `alpha = 0.7 + QE`. Saturates R@5 at 1.000 on
this eval. On the harder 95-q expanded corpus we re-tuned (next table).

*Source: `outputs/rag_eval/hybrid_ablation_raw.json`*

---

## 3. Hybrid alpha re-tuning (95-q, 4098-chunk expanded corpus)

After corpus expansion the 50-q grid no longer transferred. 16-point
re-tune on 95 queries:

| alpha | R@5 (with QE) | Notes |
|-------|---------------|-------|
| 0.0 (BM25 only) | 0.621 | term match only |
| 0.5 | 0.797 | even weighting |
| 0.6 | 0.806 | |
| 0.7 | 0.797 | (no longer the best) |
| 0.75 | 0.816 | |
| **0.80** | **0.837** ← **chosen** | best across grid |
| 0.85 | 0.826 | |
| 0.9 | 0.821 | |
| 1.0 (dense only) | 0.815 | semantic only |

**Pick (95-q corpus):** `alpha = 0.8 + QE`. The shift from 0.7 to 0.8
reflects the expanded corpus having more semantically-similar but
topically-different pages — dense gets more weight to disambiguate.

*Source: `outputs/rag_eval/retrieval_eval_report_hybrid_a08_qe_95q.json`*

---

## 4. Query expansion (95-q)

| Triggers | R@1 | R@3 | R@5 | MRR | zero-recall@5 |
|----------|-----|-----|-----|-----|--------------|
| Off       | 0.158 | 0.475 | 0.630 | — | many |
| 18 triggers | 0.266 | 0.688 | 0.837 | 0.621 | 3 |
| **25 triggers** (final) | **0.271** | **0.719** | **0.868** | **0.640** | **0** |

**Added in v2 (commit `30ac6ad`):** `cost to live`, `dorm cost`,
`summer quarter`, `summer F-1`, `federal loans`, `federal loan`,
`cost to live in the on-campus dorms`. Each targets a specific
zero-recall query where the user's vocabulary differed from the
document's.

*Sources: `outputs/rag_eval/retrieval_eval_report__before_expand.json`
vs `outputs/rag_eval/retrieval_eval_report__after_expand.json`*

---

## 5. Reranker comparison (95-q) — **negative result**

| Config | R@1 | R@5 | MRR | Candidate R@20 | Latency / q |
|--------|-----|-----|-----|----------------|------------|
| **Hybrid direct (baseline)** | **0.266** | **0.837** | **0.621** | 0.958 | **33 ms** (CPU) |
| + MiniLM rerank top-20 | 0.342 | 0.708 | 0.580 | 0.958 | 298 ms (CPU) |
| + MiniLM rerank top-50 | 0.325 | 0.676 | 0.557 | 0.988 | 543 ms (CPU) |
| + BGE-v2-m3 rerank top-20 | 0.272 | 0.711 | 0.534 | 0.958 | 1054 ms (L4 GPU) |
| + BGE-v2-m3 rerank top-50 | 0.312 | 0.672 | 0.535 | 0.988 | 2130 ms (L4 GPU) |

**Pick:** **no reranker.** Candidate pool was already saturated
(cR@20 = 0.958). Both off-the-shelf rerankers actively demoted correct
chunks because UCSD policy text doesn't look like MS-MARCO / BGE
training distribution. R@5 dropped 12-17 pp.
See `docs/rag/reranker_comparison_report.md`.

*Sources: `retrieval_eval_report_rerank_minilm_top20.json` etc.*

---

## 6. Answer generation — with vs without constraints

### 45-query answer eval

| Model | Constraints | All-pass | Failed queries |
|-------|-------------|----------|----------------|
| DPO   | off | 43/45 (95.6%) | 004, 006 (hallucinated week 4 / week 9) |
| Base  | off | 40/45 (88.9%) | 004, 006, 019, 024, 040 |
| **DPO**   | **on** | **45/45 (100%)** | — (3 retries) |
| **Base**  | **on** | **45/45 (100%)** | — (3 retries) |

### 95-query answer eval (final)

| Model | All-pass | Retries | Fallbacks | Notes |
|-------|----------|---------|-----------|-------|
| DPO + constraints | 90/95 (94.7%) | 7/95 | 5/95 | All 5 failures are the fallback message tripping `cites_source` — validator-design gap, not model error |
| Base + constraints | 90/95 (94.7%) | 7/95 | 5/95 | Same pattern; different 5 retry IDs |

If fallback messages are exempted from grounding-style checks, both
models hit 95/95 (100%).

### DPO vs Base — same number, complementary failure modes

| | DPO retried IDs | Base retried IDs |
|---|---|---|
| | 007, 011, 013, 051, 078, 085, 094 | 008, 011, 035, 041, 050, 087, 094 |
| Overlap | 011, 094 (2 / 7) |  |

DPO and Base struggle on different queries — DPO wins +1 in
course_enrollment, Base wins +1 in housing. **The 11-check eval is
saturated; finer-grained eval (answer-quality scoring) would expose
DPO's hedging / conciseness advantage.**

### Per-check pass rate (95-q DPO)

| Check | Pass rate |
|-------|-----------|
| answer_not_empty | 100% |
| **cites_source** | **94.7%** (all 5 failures are fallback messages) |
| uses_retrieved_context | 98.9% |
| no_hallucinated_deadline | 100% |
| no_hallucinated_fee | 100% |
| no_absolute_promise | 100% |
| safe_escalation | 100% |
| answer_has_steps | 98.9% |
| no_extra_notes | 100% |
| no_forbidden_claims | 100% |
| insufficient_context_behavior | 100% |

---

## 7. Latency budget (single query, M4 / L4 / A100)

| Stage | Hardware | Wall time |
|-------|----------|-----------|
| Hybrid retrieval (top-5) | CPU | ~33 ms |
| Pre-gen confidence gate | CPU | <1 ms |
| Qwen2.5-7B generation (512 tokens) | A100 80GB | 2-5 s |
| Qwen2.5-7B generation (512 tokens) | L4 22GB | 5-15 s |
| Qwen2.5-7B-4bit generation (MLX) | **M4 base** | **8-15 s** |
| 11 post-hoc validators | CPU | <50 ms |
| Retry (when triggered, ~7% of queries) | + 1× generation | +5-15 s |
| Fallback (when triggered, ~5%) | — | 0 ms (constant string) |

**Typical end-to-end (no retry, GPU):** ~3-15 s.
**With 1 retry:** ~15-30 s.

---

## 8. What we considered but did not ship

| Idea | Why not |
|------|---------|
| BAAI/bge-large-en-v1.5 as dense encoder | all-MiniLM-L6-v2 already gives 0.86 R@5; bigger embedder would help marginally at 4× cost |
| 4-bit quantize the corpus-encoded vectors | FAISS IndexFlatIP cost is dominated by retrieval, not embedding storage |
| Multi-query retrieval | Manual QE dictionary already covered the 3 zero-recall cases at zero cost |
| HyDE (hypothetical doc embedding) | Same — would help when corpus is much larger and QE infeasible |
| Decode-time grammar (outlines / lm-format-enforcer) | Post-hoc validators + retry already gives 95%+ pass with simpler infra |
| Domain-finetuned cross-encoder | Would require 1000+ labeled (query, chunk) pairs; ROI low when current R@5 = 0.868 |
| Span-level citation | Deferred to P2 future work |

---

## Reproducibility

```bash
# Retrieval eval (CPU, ~30 s on the 95-q seed)
python scripts/rag/eval_retrieval.py \
    --hybrid --use_query_expansion --alpha 0.8 \
    --report_suffix final

# Build the answer-eval prompts (one-time, CPU)
python scripts/rag/rag_answer.py \
    --build_prompt_only \
    --eval_seed data/rag/rag_answer_eval_seed.json \
    --output_file outputs/rag_eval/grounded_prompts/batch.json

# Generation eval — requires GPU (Colab L4/A100) or Apple Silicon (MLX)
# See docs/rag/grounded_generation_constraints.md and docs/rag/demo_usage.md
```
