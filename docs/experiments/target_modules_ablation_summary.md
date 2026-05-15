# Target Modules Ablation Summary

## Experiment Purpose

This ablation tested which LoRA target modules produce the strongest SFT behavior for the International Student Campus Assistant after rank ablation fixed the rank at `r=32`.

The goal was to compare quality, output-boundary control, safe escalation behavior, and adapter size.

## Controlled Variables

The following variables were fixed across all target-module runs:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT train/eval data: unchanged
- Prompt template: `chat`
- LoRA rank: `32`
- LoRA alpha: `64`
- Learning rate: `0.0002`
- Epochs: `1`
- Train batch size: `8`
- Eval batch size: `8`
- Gradient accumulation steps: `1`
- Seed: `42`
- Rule eval set: 60 samples, 311 checks

## Results

| Setting | Target Modules | Pass Rate | Prompt Leakage | Truncation | Adapter Size |
|---|---|---:|---:|---:|---:|
| `qv` | `q_proj`, `v_proj` | 93.89% | 0 | 0 | 150M |
| `qkvo` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 90.03% | 0 | 0 | 266M |
| `attn_mlp` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 98.39% | 0 | 0 | 1020M |

## Adapter Size Comparison

```text
qv:       150M
qkvo:     266M
attn_mlp: 1020M
```

`attn_mlp` is about 6.8x larger than `qv`. This is acceptable for the high-quality research checkpoint, but `qv` remains useful when storage, upload, or download cost matters.

## Interpretation

`qv` is lightweight and strong. It preserves clean generation boundaries and achieves a solid pass rate with the smallest adapter.

`qkvo` underperforms despite the larger adapter. In this run, adding `k_proj` and `o_proj` without MLP modules did not improve behavior and produced more short-answer and escalation failures.

`attn_mlp` gives the best behavioral alignment. It substantially improves rule-eval performance, removes the short-answer failure mode on this eval run, and keeps prompt leakage and truncation at zero.

## Final Decision

Final SFT config:

```text
r32 + attn_mlp
configs/final_sft_r32_attn_mlp.yaml
outputs/final_sft_r32_attn_mlp
```

Lightweight backup:

```text
r32 + qv
configs/ablations/sft_r32_qv.yaml
outputs/ablations/sft_r32_qv
```

`qkvo` is not selected because it is larger than `qv` but performs worse.
