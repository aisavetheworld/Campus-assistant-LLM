# DPO v3 Plan

## Why v2 Was Not Enough

DPO v2 achieved 100% preference win rate on the v2 eval set, but two problems remain:

1. **Rule pass rate still 98.07%** — four persistent rule failures carried over from v1 into v2, and a new CPT failure appeared (`v7_eval_course_cpt_001`). Net: 0 change in rule quality.

2. **Eval set no longer discriminative** — the v2 data revision simplified most rejected answers to one-liners, making the eval set too easy. SFT-only also scores 100% on the v2 eval set, so win rate cannot measure DPO improvement.

## What v3 Changes

### Eval set repair (6 pairs revised)

The 6 eval pairs with obviously-wrong rejected answers have been revised to near-miss responses — structured, plausible, but with a clear targeted flaw:

| Pair | Flaw in revised rejected |
|---|---|
| `dpo_housing_safe_001` | Has steps, contacts carrier/seller but not mailroom/housing office |
| `dpo_medical_safe_001` | Has steps, mentions health center for tomorrow — unsafe delay for urgent symptoms |
| `dpo_medical_safe_004` | Has steps, recommends antihistamine + wait — dangerous delay for swelling/breathing |
| `dpo_visa_safe_002` | Has steps, mentions ISO but defers contact and says "can make plans" — overconfident |
| `dpo_visa_safe_007` | Has steps, mentions ISO but says check "after you start" — wrong order |
| `dpo_housing_safe_006` | Has steps, says "usually no immediate consequences" — overconfident for high-risk |

This restores the eval set's ability to distinguish models.

### New targeted train pairs (5 pairs)

| Pair ID | Target failure | Flaw in rejected |
|---|---|---|
| `dpo_visa_safe_013` | `v7_eval_course_cpt_001` `mentions_international_office` | Has steps, uses portal/department but never contacts ISO |
| `dpo_visa_safe_014` | `eval_v3_course_003` `mentions_international_office` | Has steps, contacts academic advisor only, skips ISO entirely |
| `dpo_medical_safe_012` | `eval_v3_health_009` `mentions_official_office` | Has steps, says "relevant office" without naming health center/immunization office |
| `dpo_medical_safe_013` | `v7_eval_health_immunization_001` `mentions_official_office` | Has steps, says "contact the school" — vague, not specific office name |
| `dpo_steps_email_012` | `eval_v3_health_006` `has_steps` | Paragraph form only, no numbered steps, no email draft |

### Dataset size after v3

| | v1 | v2 | v3 |
|---|---:|---:|---:|
| Total pairs | 50 | 56 | 61 |
| Train | 40 | 45 | 49 |
| Eval | 10 | 11 | 12 |
| Avg rejected word count | 38.32 | 16.64 | 22.46 |

### Prefix distribution

| Prefix | v2 | v3 |
|---|---:|---:|
| `dpo_email_quality` | 12 | 12 |
| `dpo_housing_safe` | 10 | 10 |
| `dpo_medical_safe` | 11 | 13 |
| `dpo_steps_email` | 11 | 12 |
| `dpo_visa_safe` | 12 | 14 |

## Expected Improvements

### Rule pass rate

| Failure | v2 pairs targeting it | v3 pairs added |
|---|---|---|
| `mentions_international_office` (course drop) | `dpo_visa_safe_005/006/011` | `dpo_visa_safe_014` |
| `mentions_international_office` (CPT) | `dpo_visa_safe_003/004/013` | `dpo_visa_safe_013` |
| `mentions_official_office` (immunization ×2) | `dpo_medical_safe_005/011` | `dpo_medical_safe_012/013` |
| `has_steps` (insurance appeal) | `dpo_medical_safe_006`, `dpo_steps_email_011` | `dpo_steps_email_012` |
| `no_extra_notes` | `dpo_email_quality_006/012` | — |

Target: rule pass rate ≥ 98.39% (recover SFT-only baseline).

### Preference win rate on rebuilt eval set

The eval set now has near-miss rejected answers. SFT-only will likely score below 100% on this harder set, restoring the eval's discriminative power. DPO v3 should maintain or exceed the v2 score margin (+0.20 over SFT-only).

## Metrics to Compare

| Metric | SFT-only | DPO v2 | DPO v3 (target) |
|---|---:|---:|---:|
| Preference win rate (v3 eval set) | ? | ? | ≥ DPO v3 > SFT-only |
| Score margin | ? | ? | wider than v2 |
| Rule pass rate | 98.39% | 98.07% | ≥ 98.39% |
| Prompt leakage | 0 | 0 | 0 |
| Truncation | 0 | 0 | 0 |
| `mentions_international_office` failures | — | 2 | 0 |
| `mentions_official_office` failures | — | 2 | 0 |
| `has_steps` failures | — | 1 | 0 |
| `no_extra_notes` failures | — | 1 | 0 |

## Training Configuration

Use `configs/dpo_lora_v2.yaml` with output changed to `outputs/dpo_r32_attn_mlp_v3`:

```yaml
training:
  output_dir: "outputs/dpo_r32_attn_mlp_v3"
```

Or create `configs/dpo_lora_v3.yaml` pointing to v3 output.

## Expansion Criteria

After v3 results:
- If rule pass rate ≥ 98.39% AND preference win rate on harder eval set is higher than SFT-only → expand to 150–200 pairs.
- If rule pass rate still below baseline → one more targeted pass before expansion.
- Do not expand until both conditions are met.
