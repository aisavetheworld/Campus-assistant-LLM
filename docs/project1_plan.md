# Project 1 Plan: SFT/DPO Training Module

Project 1 builds the training layer for the International Student Campus Assistant. The first version is a UCSD-focused prototype with school-agnostic training behavior.

## Phase 1: SFT Minimal Loop

Goals:

- Build seed dataset.
- Process raw JSON into train/eval JSONL.
- Run base model baseline inference.
- Run LoRA SFT.
- Compare before/after outputs.
- Collect bad cases for dataset improvement.

Execution:

1. Write seed samples covering housing, course enrollment, health insurance, and email drafting.
2. Run `scripts/build_sft_data.py`.
3. Run `scripts/infer.py` with the base model on representative prompts.
4. Train with `scripts/train_sft.py`.
5. Run `scripts/infer.py` with the LoRA adapter.
6. Run `scripts/eval_sft.py`.
7. Review failed checks and update seed data.

## Phase 2: LoRA Ablation

Experiments:

- Rank: `4 / 8 / 16 / 32`.
- Target modules:
  - `q_proj + v_proj`.
  - `q_proj + k_proj + v_proj + o_proj`.
  - all linear modules if supported by the base model and PEFT setup.
- LoRA vs QLoRA.

Record metrics:

- Training loss.
- Eval loss.
- GPU memory.
- Training time.
- Email format validity.
- Safe escalation pass rate.
- Forbidden absolute promise rate.

Suggested experiment naming:

```text
outputs/sft_lora_r4_qv/
outputs/sft_lora_r8_qv/
outputs/sft_lora_r16_qkvo/
outputs/sft_qlora_r16_qv/
```

## Phase 3: DPO

Goals:

- Build chosen/rejected preference pairs.
- Train DPO later.
- Compare SFT-only vs SFT+DPO.

Preference principles:

- Chosen answers are more polite.
- Chosen answers are more specific.
- Chosen answers provide steps.
- Chosen answers do not invent policy facts.
- Chosen answers avoid absolute promises.
- Chosen answers refer high-risk cases to official offices.

Metrics:

- Preference win rate.
- Safety violation rate.
- Email quality score.
- Response helpfulness score.

Project 1 only validates and stages DPO data. Full DPO training is reserved for a later phase.

## Phase 4: Bridge to Project 2

Project 2 will add retrieval over official university documents.

Planned scope:

- UCSD official document collection.
- RAG indexing.
- Hybrid retrieval.
- Citation.
- Low-confidence refusal.

Design boundary:

- SFT teaches behavior.
- RAG supplies current policy facts.

## Phase 5: Bridge to Project 3

Project 3 will add production-style serving and benchmarking.

Planned scope:

- vLLM serving.
- FastAPI wrapper.
- Redis cache.
- Rate limiting.
- Latency/QPS benchmark.
- Quantization comparison.
