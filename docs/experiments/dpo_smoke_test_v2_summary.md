# DPO Smoke Test v2 Summary

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT-only adapter: `outputs/final_sft_r32_attn_mlp`
- SFT+DPO v1 adapter: `outputs/dpo_r32_attn_mlp_v1`
- SFT+DPO v2 adapter: `outputs/dpo_r32_attn_mlp_v2`
- DPO v2 train data: 45 pairs (56 total, 11 eval)
- Rule eval data: 60 samples, 311 checks
- DPO epochs: 1

## Comparison

| Metric | SFT-only | SFT+DPO v1 | SFT+DPO v2 | v1→v2 Delta |
|---|---:|---:|---:|---:|
| Preference eval pairs | 10 | 10 | 11 | - |
| Preference chosen wins | 9 | 9 | 11 | +2 |
| Preference win rate | 90.00% | 90.00% | 100.00% | +10.00 pp |
| Average chosen score | -1.8852 | -1.8695 | -1.6261 | +0.2434 |
| Average rejected score | -2.5445 | -2.5938 | -3.2195 | -0.6257 |
| Rule passed checks | 306 / 311 | 305 / 311 | 305 / 311 | 0 |
| Rule pass rate | 98.39% | 98.07% | 98.07% | +0.00 pp |
| Raw prompt leakage count | 0 | 0 | 0 | 0 |
| Final prompt leakage count | 0 | 0 | 0 | 0 |
| Truncated count | 0 | 0 | 0 | 0 |
| Early truncation count | 0 | 0 | 0 | 0 |

## Interpretation

DPO v2 achieves the primary preference alignment target: preference win rate improves from 90.00% to 100.00%, with all 11 eval pairs correctly preferring the chosen response. The score margin also widened significantly — average chosen score improved by +0.24 and average rejected score dropped by 0.63, indicating the model now assigns clearly higher probability to chosen responses across all categories and risk levels.

The key failed pair from v1 (`dpo_visa_safe_010`) now passes. The chosen score for `dpo_visa_safe_010` improved from a v1 inversion to a correct win (-2.0625 vs -3.5312).

However, the rule pass rate remains at 98.07%, unchanged from v1. DPO v2 fixed one rule failure (`v7_eval_email_extension_001: has_closing`) but introduced one new failure (`v7_eval_course_cpt_001: mentions_international_office`), resulting in a net zero change on rule pass rate. The four persistent rule failures from v1 — `mentions_international_office`, `mentions_official_office` (×2), `has_steps`, and `no_extra_notes` — were not resolved by v2 training.

## Rule Eval: Changed Failures

| Sample | Check | v1 | v2 |
|---|---|---|---|
| `v7_eval_email_extension_001` | `has_closing` | FAIL | **PASS** ✓ |
| `v7_eval_course_cpt_001` | `mentions_international_office` | PASS | **FAIL** ✗ |

## SFT+DPO v2 Rule Failures

| ID | Failed Check |
|---|---|
| `eval_v3_email_011` | `no_extra_notes` |
| `eval_v3_course_003` | `mentions_international_office` |
| `eval_v3_health_006` | `has_steps` |
| `eval_v3_health_009` | `mentions_official_office` |
| `v7_eval_health_immunization_001` | `mentions_official_office` |
| `v7_eval_course_cpt_001` | `mentions_international_office` |

## Decision

Do not promote `outputs/dpo_r32_attn_mlp_v2` over `outputs/final_sft_r32_attn_mlp` yet.

- `outputs/final_sft_r32_attn_mlp` remains the stable Project 1 final SFT checkpoint.
- `outputs/dpo_r32_attn_mlp_v2` is the best DPO checkpoint to date on preference alignment.
- Rule pass rate has not recovered to the SFT-only baseline of 98.39%.

## Recommendation

DPO v2 is a meaningful improvement over v1 on preference win rate (90% → 100%), but the rule pass rate remains below the SFT-only baseline. The trade of fixing `has_closing` while introducing a new `mentions_international_office` failure suggests the DPO training is still not applying consistently to CPT/ISO escalation scenarios.

Recommended next step:

1. Inspect the raw response for `v7_eval_course_cpt_001` to determine whether the new failure is a DPO regression or a borderline case.
2. Add a small targeted v3 correction pass focused on the two remaining persistent patterns: `mentions_international_office` (now 2 failures across both course and CPT samples) and `mentions_official_office` (2 immunization samples).
3. If rule pass rate recovers to ≥ 98.39% in v3 while preference win rate holds at 100%, the DPO pipeline is ready for expansion to 200 pairs.
