# International Student Campus Assistant

A UCSD-focused prototype with school-agnostic training behavior.

This repository covers a three-project pipeline for an international student campus affairs assistant. All three projects are complete: Project 1 (SFT/DPO fine-tuning), Project 2 (RAG retrieval pipeline), and Project 3 (FastAPI + vLLM serving).

## Project Overview

The assistant is designed to help international students handle common campus affairs such as housing issues, course enrollment questions, health insurance workflows, and formal English email drafting.

Project 1 focuses only on the training layer:

- SFT for response behavior, structure, email writing, bilingual handling, and safe escalation.
- LoRA/QLoRA for parameter-efficient fine-tuning.
- DPO seed data and a one-epoch smoke-test path for preference alignment.
- No RAG, no vLLM, no FastAPI, and no full web app in this phase.

## Project 1 Status

- SFT pipeline complete.
- LoRA rank ablation complete; rank `32` is the selected rank.
- Target modules ablation complete.
- Final high-quality SFT config: `configs/final_sft_r32_attn_mlp.yaml`.
- Final target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Lightweight backup config: `configs/ablations/sft_r32_qv.yaml`.
- DPO preference data is prepared; one-epoch DPO smoke-test training is now available.

## Why This Project Exists

International students often need help translating vague campus problems into clear next steps and polite English communication. The goal is not to replace official offices, advisors, healthcare providers, or legal professionals. The goal is to teach the model a reliable behavior pattern: clarify, structure, draft, and escalate when risk is high.

## Project 1 Scope

Covered categories:

- `housing`: dorms, mailroom, packages, maintenance, roommates, lease/contract general questions.
- `course_enrollment`: add/drop, waitlist, prerequisites, professor/department contact, missed deadline general guidance.
- `health_insurance`: waiver, immunization, clinic finding, insurance office contact, medical safety boundaries.
- `email_drafting`: professor, housing, department, insurance, and international student office emails.

Not included:

- Retrieval over official documents.
- Campus-specific policy QA with citations.
- vLLM serving.
- FastAPI deployment.
- Large-scale DPO training beyond the smoke test.

## Why SFT Does Not Memorize School Policies

SFT teaches behavior; RAG supplies facts.

The SFT dataset should not train the model to memorize UCSD-specific deadlines, fees, office addresses, or binding policy details. Those facts can change and should be supplied later by Project 2 RAG over official documents. Project 1 trains the model to:

- Identify campus affairs intent.
- Answer with clear steps.
- Give actionable but non-binding guidance.
- Draft formal English emails.
- Handle Chinese and mixed Chinese-English questions.
- Avoid unsafe certainty for visa, legal, medical, insurance claim, and academic misconduct topics.
- Refer students to official offices and official websites when needed.

## Data Strategy

Seed data uses four source types:

- `official_faq`: FAQ-style samples inspired by official university pages, used to teach boundaries and process awareness.
- `template_synthetic`: common international student scenarios expanded into instruction/input/output examples.
- `email_template`: formal email writing samples.
- `safe_escalation`: high-risk scenarios where the model must avoid absolute conclusions and refer to official support.

## Dataset Schema

Raw SFT seed data lives at `data/raw/sft_seed.json` as a JSON list with metadata:

```json
{
  "id": "housing_001",
  "school": "UCSD",
  "category": "housing",
  "task_type": "process_guidance",
  "risk_level": "low",
  "source_type": "template_synthetic",
  "source_url": "",
  "date_collected": "2026-05-12",
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

Processed SFT data is JSONL with an added `text` field:

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

## Setup

```bash
pip install -r requirements.txt
```

For Colab, use a GPU runtime. If you enable QLoRA, keep `bitsandbytes` installed and set `use_4bit: true` in `configs/sft_lora.yaml`.

## Build SFT Data

```bash
python scripts/build_sft_data.py \
  --input_file data/raw/sft_seed.json \
  --train_output data/processed/sft_train.jsonl \
  --eval_output data/processed/sft_eval.jsonl \
  --eval_ratio 0.2 \
  --seed 42
```

This validates required fields, removes duplicates, preserves metadata, creates the SFT prompt text, splits train/eval, and prints dataset statistics.

## Train LoRA SFT

Edit `configs/sft_lora.yaml` if you want to change the model, LoRA rank, target modules, sequence length, or 4-bit loading.

```bash
python scripts/train_sft.py --config configs/sft_lora.yaml
```

Outputs are saved to:

```text
outputs/sft_lora/
```

The training script saves the LoRA adapter, tokenizer files, `resolved_config.yaml`, trainer state, and metrics.

## Run Inference

Base model:

```bash
python scripts/infer.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --instruction "用户是国际学生，想给教授写邮件申请延期。" \
  --input "原因：生病；作业：project report；希望语气礼貌，不要太长。" \
  --max_new_tokens 300
```

With LoRA adapter:

```bash
python scripts/infer.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/sft_lora \
  --instruction "用户是国际学生，想给教授写邮件申请延期。" \
  --input "原因：生病；作业：project report；希望语气礼貌，不要太长。" \
  --max_new_tokens 300
```

## Evaluate

The evaluator uses deterministic rules only. It does not call LLM-as-judge.

```bash
python scripts/eval_sft.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/sft_lora \
  --eval_file data/eval/eval_seed.json \
  --output_json outputs/eval_report.json \
  --output_md outputs/eval_report.md
```

Checks include email formatting, safe escalation language, official office referrals, non-empty responses, step structure, and dangerous absolute-promise detection.

## DPO Smoke Test

Project 1 includes prepared DPO preference data and a one-epoch smoke-test path for `SFT-only` vs `SFT+DPO` comparison.

The current DPO data targets:

- `steps_plus_email` completeness;
- safer CPT/OPT/visa escalation;
- safer medical and insurance escalation;
- housing office / Student Mail escalation;
- concise professional email quality.

```bash
python scripts/build_dpo_data.py \
  --input_file data/dpo/dpo_seed.json \
  --train_output data/dpo/dpo_train.jsonl \
  --eval_output data/dpo/dpo_eval.jsonl \
  --eval_ratio 0.2 \
  --seed 42
```

This validates DPO seed pairs, audits preference quality, creates `formatted_prompt`, and writes an 80/20 train/eval split for TRL `DPOTrainer`.

Run the DPO smoke test:

```bash
python scripts/train_dpo.py --config configs/dpo_lora.yaml
```

Evaluate DPO preference win rate:

```bash
python scripts/eval_dpo_preference.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/dpo_r32_attn_mlp_v1 \
  --eval_file data/dpo/dpo_eval.jsonl \
  --output_json outputs/dpo_r32_attn_mlp_v1/preference_eval.json \
  --output_md outputs/dpo_r32_attn_mlp_v1/preference_eval.md
```

## References / Engineering Inspiration

- LLaMA-Factory: used as engineering reference for data templates and experiment configuration organization.
- TRL: used for actual SFT/DPO training implementation.
- PEFT: used for LoRA/QLoRA parameter-efficient fine-tuning.
- Transformers: used for model/tokenizer loading and inference.

## Final Results

| Project | Status | Key Metrics |
|---------|--------|-------------|
| P1: SFT + DPO | ✅ Complete | Rule pass rate 98.39% (311 checks), preference win rate 90.00%, `mentions_international_office` failures = 0 |
| P2: RAG Pipeline | ✅ Complete | Recall@5 = 0.868, Hit@5 = 1.000, answer pass rate 94.7% (90/95), retrieval latency 33ms (CPU) |
| P3: Serving | ✅ Complete | vLLM + FastAPI + Redis cache; sweet spot u=8: RPS=0.62, P95=18s, zero failures; Redis cache 338× speedup |

## Quick Start — RAG + Serving

**1. Start vLLM (requires GPU with ~15GB VRAM):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name campus-assistant \
  --enable-lora --max-lora-rank 32 \
  --lora-modules campus-assistant=outputs/dpo_7b \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.88
```

**2. Start FastAPI (new terminal):**
```bash
SERVING_BACKEND=vllm uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

**3. Run smoke test:**
```bash
python scripts/deploy/test_chat_api.py --url http://localhost:8080
```

**4. Query the API:**
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I waive UC SHIP health insurance?", "top_k": 5}'
```

Optional: start Redis for query caching (`redis-server --daemonize yes`) before step 2.

## Roadmap

- **Project 1 (complete):** SFT + LoRA ablation + DPO preference alignment. Best checkpoint: `outputs/dpo_7b`. See `docs/experiments/dpo_7b_summary.md`.
- **Project 2 (complete):** RAG over 673 official UCSD source documents, 4098 chunks. See `docs/rag/project_2_final_report.md`.
- **Project 3 (complete):** FastAPI + vLLM serving, load testing, KV cache analysis, Redis caching, failure drills. See `docs/deployment/project3_final_report.md`.
