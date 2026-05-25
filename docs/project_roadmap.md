# Campus Assistant LLM — Project Roadmap

## Project 1: Model Behavior Alignment — COMPLETE / FROZEN

**Status:** Frozen. Do not modify SFT data, DPO data, or training configs unless explicitly requested.

**Best checkpoint:** `outputs/dpo_7b` (Qwen2.5-7B-Instruct + SFT adapter + DPO adapter)

Completed steps:

- SFT pipeline (data build, training, rule-based eval)
- LoRA rank ablation — rank 32 selected
- Target modules ablation — attn + MLP selected
- DPO preference data construction (151 pairs)
- DPO smoke test (v1–v5, 1.5B)
- DPO beta ablation (beta=0.05/0.10/0.30, 1.5B) — beta=0.10 selected
- Scale-up to Qwen2.5-7B (SFT + DPO)
- Preference eval: 90.00% win rate (7B DPO vs SFT-only 76.67%)
- Rule eval: 97.75% pass rate (304/311 checks)
- `mentions_international_office` resolved at 7B scale

Known limitations (accepted):

- `no_extra_notes` oscillates (5 failures in promoted run)
- `no_absolute_promise` can appear occasionally (1 failure in promoted run)
- 1-epoch LoRA DPO at this data scale is mildly non-deterministic

**Key configs:**

- `configs/sft_7b.yaml`
- `configs/dpo_7b.yaml`

---

## Project 2: RAG with Official UCSD Sources — IN PROGRESS

**Status:** Planning / skeleton phase.

Goal: Ground the model's answers in official UCSD source documents so it does not rely on memorized or potentially hallucinated policy details.

Steps:

1. Official source collection and metadata
2. Source text storage and cleaning
3. Document chunking
4. Embedding
5. Vector index build
6. Retrieval (top-k)
7. Grounded prompt construction
8. Answer generation with current best checkpoint
9. Retrieval and grounded answer evaluation

**Key files:**

- `docs/project2_rag_plan.md`
- `docs/rag/source_collection_guide.md`
- `docs/rag/rag_eval_plan.md`
- `data/rag/ucsd_sources.json`
- `scripts/rag/`

---

## Project 3: Serving / Deployment — NOT STARTED

**Status:** Not started. Do not implement until Project 2 is substantially complete.

Planned steps:

- FastAPI inference endpoint
- vLLM integration
- Request batching
- Quantization (int4/int8)
- Latency measurement and benchmarking
- Deployment documentation

---

## Important Notes

- Project 2 is RAG, not serving. Serving belongs to Project 3.
- The SFT/DPO training stack from Project 1 is frozen.
- Project 2 uses the `outputs/dpo_7b` checkpoint for answer generation.
- Do not hard-code changing UCSD deadlines or fees into SFT/DPO data; those belong in the RAG source documents.
