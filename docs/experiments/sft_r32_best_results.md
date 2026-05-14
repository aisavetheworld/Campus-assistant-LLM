# SFT r32 Best Formal Run Results

## Summary

This run records the formal rank-32 LoRA SFT checkpoint after the rank ablation phase.

Current best checkpoint:

```text
Config: configs/sft_lora.yaml
Output dir: outputs/sft_lora_r32_best
LoRA rank: 32
LoRA alpha: 64
Prompt template: chat
Train batch size: 8
Gradient accumulation steps: 1
```

## Dataset

- Raw SFT samples: 220
- Processed train samples: 176
- Processed SFT eval split samples: 44
- Rule eval samples: 60
- Rule checks: 311

## Result

| Metric | Value |
|---|---:|
| Passed checks | 293 / 311 |
| Pass rate | 94.21% |
| Raw generation prompt leakage | 0 |
| Final response prompt leakage | 0 |
| Truncated outputs | 0 |
| Early truncation | 0 |
| Late truncation | 0 |
| Not-too-short failures caused by stop truncation | 0 |
| Not-too-short failures without stop truncation | 8 |

## Failed Check Counts

| Failed check | Count |
|---|---:|
| `not_too_short` | 8 |
| `mentions_official_office` | 3 |
| `mentions_international_office` | 3 |
| `mentions_academic_office` | 2 |
| `no_extra_notes` | 1 |
| `mentions_healthcare_provider` | 1 |

## Interpretation

The formal r32 run improves over the rank ablation r32 result:

| Run | Pass rate | Raw leakage | Final leakage | Truncation |
|---|---:|---:|---:|---:|
| r32 ablation | 93.57% | 0 | 0 | 0 |
| r32 formal best | 94.21% | 0 | 0 | 0 |

The previous boundary-control problems are no longer the dominant issue:

- No raw prompt leakage.
- No final response prompt leakage.
- No stop-sequence truncation.
- No early truncation.
- Email formatting is mostly stable, with one remaining `no_extra_notes` failure.

The main remaining failure mode is short answers on a small set of prompts. These short answers are not caused by stop-sequence truncation. They likely come from a learned terse refusal or minimal-answer behavior on certain safe-escalation and official-office prompts.

## Recommended Next Step

Do not continue tuning against this eval set row by row.

Recommended next work:

1. Save qualitative base-vs-SFT comparisons on the original 8 baseline prompts.
2. If improving SFT further, expand or refresh the eval set before another targeted data pass.
3. Before DPO, add preference pairs that prefer complete numbered safe-escalation responses over short refusals.
