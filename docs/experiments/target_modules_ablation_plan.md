# LoRA Target Modules Ablation Plan

## Purpose

This experiment compares which LoRA target modules are most useful for the International Student Campus Assistant SFT task after rank ablation selected rank 32 as the best current setting.

The goal is to measure whether adapting more attention projections or adding MLP projections improves:

- structured step-by-step responses;
- safe escalation behavior;
- official-office mentions;
- output boundary control;
- adapter quality relative to adapter size.

## Controlled Variables

Keep these variables fixed across all runs:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Train/eval data: unchanged v7 processed data
- Rule eval set: unchanged 60-sample eval set
- Prompt template: `chat`
- LoRA rank: `32`
- LoRA alpha: `64`
- Learning rate: `0.0002`
- Epochs: `1`
- Train batch size: `8`
- Eval batch size: `8`
- Gradient accumulation steps: `1`
- Seed: `42`
- LoRA dropout: `0.05`

Do not modify the eval set or tune data against this experiment's results.

## Configs

| Config | Output Dir | Target Modules |
|---|---|---|
| `configs/ablations/sft_r32_qv.yaml` | `outputs/ablations/sft_r32_qv` | `q_proj`, `v_proj` |
| `configs/ablations/sft_r32_qkvo.yaml` | `outputs/ablations/sft_r32_qkvo` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `configs/ablations/sft_r32_attn_mlp.yaml` | `outputs/ablations/sft_r32_attn_mlp` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

## Expected Tradeoffs

`q_proj + v_proj` is the current lightweight baseline. It changes fewer modules and keeps the adapter smaller.

`q_proj + k_proj + v_proj + o_proj` adapts the full attention projection stack. It may improve instruction following and structured response consistency while increasing adapter size.

`attention + MLP projections` adapts both attention and feed-forward blocks. It has the highest capacity and largest adapter size, and may help with response style and safe-escalation completeness, but it is also the most likely to overfit on a small dataset.

## Metrics To Record

Record these values for each config:

- Pass rate
- Raw generation prompt leakage count
- Final response prompt leakage count
- Truncated count
- `not_too_short` failures
- `has_steps` failures
- Official escalation failures:
  - `mentions_official_office`
  - `mentions_international_office`
  - `mentions_academic_office`
  - `mentions_healthcare_provider`
- Adapter size
- Training time if available

## Run Commands

Print commands:

```bash
python scripts/run_target_modules_ablation.py --with_eval
```

Run the full target_modules ablation:

```bash
python scripts/run_target_modules_ablation.py --with_eval --eval_batch_size 8 --run
```

Measure adapter sizes after training:

```bash
du -sh outputs/ablations/sft_r32_qv \
  outputs/ablations/sft_r32_qkvo \
  outputs/ablations/sft_r32_attn_mlp
```
