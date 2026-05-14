# LoRA Rank Ablation Plan

## Purpose

Compare how LoRA rank affects the SFT adapter for the International Student Campus Assistant training module. The goal is to choose a practical rank before doing broader experiments, not to tune on individual eval failures.

Current baseline before ablation:

- SFT data: 220 raw samples
- Rule eval data: 60 samples
- Prompt format: Qwen chat template
- Latest rule eval: 91.00% pass rate
- Raw/final prompt leakage: 0
- Truncation / early truncation: 0

## Controlled Variables

Keep these fixed across all runs:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset files:
  - `data/processed/sft_train.jsonl`
  - `data/processed/sft_eval.jsonl`
  - `data/eval/eval_seed.json`
- Prompt template: `chat`
- Epochs: `1`
- Batch size: `1`
- Gradient accumulation steps: `8`
- Learning rate: `0.0002`
- LoRA dropout: `0.05`
- Target modules:
  - `q_proj`
  - `v_proj`
- Evaluation command and rule checks
- Random seed: `42`

## Rank Comparison

| Config | Rank | Alpha | Output Dir |
|---|---:|---:|---|
| `configs/ablations/sft_lora_r4.yaml` | 4 | 8 | `outputs/ablations/sft_lora_r4` |
| `configs/ablations/sft_lora_r8.yaml` | 8 | 16 | `outputs/ablations/sft_lora_r8` |
| `configs/ablations/sft_lora_r16.yaml` | 16 | 32 | `outputs/ablations/sft_lora_r16` |
| `configs/ablations/sft_lora_r32.yaml` | 32 | 64 | `outputs/ablations/sft_lora_r32` |

## Commands

Print copyable commands:

```bash
python scripts/run_rank_ablation.py --with_eval
```

Run training and rule eval sequentially:

```bash
python scripts/run_rank_ablation.py --with_eval --run
```

Run one config manually:

```bash
python scripts/train_sft.py --config configs/ablations/sft_lora_r16.yaml
python scripts/eval_sft.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/ablations/sft_lora_r16 \
  --eval_file data/eval/eval_seed.json \
  --output_json outputs/ablations/reports/eval_report_sft_lora_r16.json \
  --output_md outputs/ablations/reports/eval_report_sft_lora_r16.md \
  --max_new_tokens 300 \
  --temperature 0 \
  --prompt_template chat \
  --torch_dtype bfloat16
```

## Metrics To Record

For each rank, record:

- Training loss
- Eval loss
- Rule eval pass rate
- Raw prompt leakage count
- Final prompt leakage count
- Truncated count
- Early truncation count
- `not_too_short` failures
- `has_steps` failures
- Official escalation failures:
  - `mentions_official_office`
  - `mentions_international_office`
  - `mentions_academic_office`
  - `mentions_healthcare_provider`
- Training time
- Peak GPU memory if available
- Adapter size

## Result Table

| Rank | Alpha | Train Loss | Eval Loss | Pass Rate | Raw Leakage | Final Leakage | Short Failures | Step Failures | Official Mention Failures | Time | Peak GPU Mem | Adapter Size | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 4 | 8 |  |  |  |  |  |  |  |  |  |  |  |  |
| 8 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |
| 16 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |
| 32 | 64 |  |  |  |  |  |  |  |  |  |  |  |  |

## Decision Rule

Prefer the smallest rank that keeps rule eval quality close to the best run while preserving zero prompt leakage and low short-answer failures. If two ranks are close, prefer the smaller adapter unless the larger rank clearly improves safety or official escalation reliability.
