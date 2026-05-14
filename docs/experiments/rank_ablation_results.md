# LoRA Rank Ablation Results

## Summary

The first LoRA rank ablation compared ranks 4, 8, 16, and 32 using the expanded v7 SFT dataset and 60-sample rule evaluation set.

Best current choice:

```text
LoRA rank: 32
LoRA alpha: 64
Output config: configs/sft_lora.yaml
Best adapter output dir: outputs/sft_lora_r32_best
```

## Experiment Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Prompt template: `chat`
- SFT raw samples: 220
- Processed train samples: 176
- Processed SFT eval samples: 44
- Rule eval samples: 60
- Rule checks: 311
- Epochs: 1
- Target modules:
  - `q_proj`
  - `v_proj`
- LoRA alpha policy: `2 * rank`
- Data and eval set were fixed across ranks.

## Results

| Rank | Alpha | Pass Rate | Raw Leakage | Final Leakage | Truncation | Early Truncation | not_too_short | has_steps | Official Mention Failures | Adapter Size |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 91.00% | 0 | 0 | 0 | 0 | 11 | 6 | 11 | 48M |
| 8 | 16 | 88.75% | 0 | 0 | 0 | 0 | 14 | 6 | 15 | 63M |
| 16 | 32 | 91.64% | 0 | 0 | 0 | 0 | 12 | 4 | 10 | 92M |
| 32 | 64 | 93.57% | 0 | 0 | 0 | 0 | 9 | 0 | 11 | 150M |

Official mention failures combine:

- `mentions_official_office`
- `mentions_international_office`
- `mentions_academic_office`
- `mentions_healthcare_provider`

## Decision

Use rank 32 as the current best SFT LoRA setting.

Reasons:

- Highest pass rate: 93.57%
- Zero raw prompt leakage
- Zero final response prompt leakage
- Zero truncation and early truncation
- Zero `has_steps` failures
- Adapter size is still practical at 150M

Rank 16 remains the lightweight backup:

- Adapter size: 92M
- Pass rate: 91.64%
- Still has 4 `has_steps` failures

## Remaining Failure Pattern

All ranks still show a short-answer failure mode on a small group of prompts. These are not caused by prompt leakage or truncation.

Common affected examples include:

- `eval_v3_housing_001`
- `eval_v3_course_001`
- `eval_v3_course_003`
- `eval_v3_health_002`
- `eval_v3_health_009`

These failures usually produce 6-9 word answers. This suggests a learned short-refusal pattern rather than insufficient LoRA capacity.

## Next Steps

Use `configs/sft_lora.yaml` as the main r32 configuration for future SFT runs.

Before DPO, consider one targeted data pass focused on replacing short refusals with full four-step safe escalation responses. Do not tune directly against individual eval rows unless the eval set is expanded or refreshed.
