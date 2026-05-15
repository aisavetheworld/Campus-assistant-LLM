# LoRA Target Modules Ablation Results

## Summary

This experiment compared three LoRA target-module settings after rank ablation selected rank 32 as the best current rank.

Best quality setting:

```text
Config: configs/ablations/sft_r32_attn_mlp.yaml
Output dir: outputs/ablations/sft_r32_attn_mlp
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Pass rate: 98.39%
Adapter size: 1020M
```

Best lightweight setting:

```text
Config: configs/ablations/sft_r32_qv.yaml
Output dir: outputs/ablations/sft_r32_qv
Target modules: q_proj, v_proj
Pass rate: 93.89%
Adapter size: 150M
```

## Experiment Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Prompt template: `chat`
- SFT raw samples: 220
- Processed train samples: 176
- Processed SFT eval split samples: 44
- Rule eval samples: 60
- Rule checks: 311
- LoRA rank: 32
- LoRA alpha: 64
- Epochs: 1
- Train batch size: 8
- Gradient accumulation steps: 1
- Max new tokens during rule eval: 300
- Temperature: 0
- Evaluation run commit: `32a8c87`

Note: this target-modules ablation was run before the batched evaluation helper commit `306b3d8`. The results are valid, but evaluation was slower because generation was still one sample at a time.

## Results

| Config | Target Modules | Passed Checks | Pass Rate | Raw Leakage | Final Leakage | Truncation | not_too_short | has_steps | Official Mention Failures | Adapter Size | Train Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sft_r32_qv` | `q_proj`, `v_proj` | 292 / 311 | 93.89% | 0 | 0 | 0 | 9 | 0 | 9 | 150M | 10.81s |
| `sft_r32_qkvo` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 280 / 311 | 90.03% | 0 | 0 | 0 | 15 | 1 | 15 | 266M | 12.14s |
| `sft_r32_attn_mlp` | attention + MLP projections | 306 / 311 | 98.39% | 0 | 0 | 0 | 0 | 1 | 3 | 1020M | 17.06s |

Official mention failures combine:

- `mentions_official_office`
- `mentions_international_office`
- `mentions_academic_office`
- `mentions_healthcare_provider`

## Failed Check Counts

### `sft_r32_qv`

| Failed check | Count |
|---|---:|
| `not_too_short` | 9 |
| `mentions_official_office` | 3 |
| `mentions_international_office` | 3 |
| `mentions_academic_office` | 2 |
| `no_extra_notes` | 1 |
| `mentions_healthcare_provider` | 1 |

### `sft_r32_qkvo`

| Failed check | Count |
|---|---:|
| `not_too_short` | 15 |
| `mentions_academic_office` | 6 |
| `mentions_international_office` | 5 |
| `mentions_healthcare_provider` | 2 |
| `mentions_official_office` | 2 |
| `has_steps` | 1 |

### `sft_r32_attn_mlp`

| Failed check | Count |
|---|---:|
| `mentions_official_office` | 2 |
| `no_extra_notes` | 1 |
| `mentions_international_office` | 1 |
| `has_steps` | 1 |

## Interpretation

`sft_r32_attn_mlp` is the clear rule-evaluation winner. It removes the short-answer failure mode on this eval set:

```text
not_too_short failures: 0
raw prompt leakage: 0
final prompt leakage: 0
truncation: 0
```

The tradeoff is adapter size. The attention + MLP adapter is about 6.8x larger than the `q_proj + v_proj` adapter:

```text
qv adapter: 150M
attn_mlp adapter: 1020M
```

`sft_r32_qv` remains the best lightweight option. It is close to the previous formal r32 best run and keeps the adapter small.

`sft_r32_qkvo` is not recommended in the current setup. It is larger than `qv` but performs worse on the rule eval set, mostly due to more short-answer and official-escalation failures.

## Decision

Use `sft_r32_attn_mlp` as the current highest-quality SFT adapter when adapter size is acceptable.

Use `sft_r32_qv` when adapter size or upload/download cost matters more than squeezing out the final few rule-eval points.

Do not use `sft_r32_qkvo` as the default based on the current results.

## Next Step

Before changing data again, run a qualitative base-vs-SFT comparison using the original 8 representative prompts. The comparison should include:

1. Base model output.
2. `sft_r32_qv` output.
3. `sft_r32_attn_mlp` output.
4. Short notes on format, safety, official escalation, and usefulness.

Do not tune directly against the current eval set row by row.
