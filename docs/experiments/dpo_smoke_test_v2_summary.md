# DPO Smoke Test v2 Summary

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT-only adapter: `outputs/final_sft_r32_attn_mlp`
- SFT+DPO v1 adapter: `outputs/dpo_r32_attn_mlp_v1`
- SFT+DPO v2 adapter: `outputs/dpo_r32_attn_mlp_v2`
- DPO v2 train data: 45 pairs (56 total, 11 eval)
- Rule eval data: 60 samples, 311 checks
- DPO epochs: 1

## Preference Comparison (v2 eval set)

SFT-only was re-evaluated on the current v2 eval set (11 pairs) for a fair comparison.

| Metric | SFT-only (v2 eval set) | SFT+DPO v2 | Delta |
|---|---:|---:|---:|
| Preference eval pairs | 11 | 11 | - |
| Preference chosen wins | 11 | 11 | 0 |
| Preference win rate | 100.00% | 100.00% | +0.00 pp |
| Average chosen score | -1.6570 | -1.6261 | +0.0309 |
| Average rejected score | -3.0511 | -3.2195 | -0.1684 |
| **Score margin** | **1.3942** | **1.5934** | **+0.1992** |

## Full Comparison (all checkpoints)

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

### Preference alignment

The v2 eval set is now too easy for SFT-only: both SFT-only and SFT+DPO v2 achieve 100% win rate on all 11 pairs. Win rate alone cannot differentiate the two models on this set.

The score margin provides a more meaningful signal. DPO v2 has a wider chosen–rejected gap (1.5934) compared to SFT-only (1.3942), a difference of +0.20. This indicates DPO training has made the model more confident in correct preference ordering, even on pairs the SFT model already handles.

The root cause of the eval set saturation is the v2 data revision: most rejected answers were simplified to short, obviously wrong responses to provide clearer DPO training signal. This made the eval set easier for both models. A harder eval set with near-miss rejected answers is needed to measure preference alignment robustly.

The key failed pair from v1 (`dpo_visa_safe_010`) is now fixed. The chosen score for `dpo_visa_safe_010` is -2.0625 vs rejected -3.5312 — a clear win where v1 had an inversion. This was the primary DPO alignment target for v2.

### Rule eval

Rule pass rate is unchanged at 98.07% (305/311). DPO v2 fixed one failure (`v7_eval_email_extension_001: has_closing`) but introduced one new failure (`v7_eval_course_cpt_001: mentions_international_office`), resulting in net zero change. The rule pass rate has not recovered to the SFT-only baseline of 98.39%.

## Rule Eval: Changed Failures

| Sample | Check | v1 | v2 |
|---|---|---|---|
| `v7_eval_email_extension_001` | `has_closing` | FAIL | **PASS** ✓ |
| `v7_eval_course_cpt_001` | `mentions_international_office` | PASS | **FAIL** ✗ |

## SFT+DPO v2 Rule Failures

| ID | Failed Check | Pattern |
|---|---|---|
| `eval_v3_email_011` | `no_extra_notes` | Extra commentary after email closing |
| `eval_v3_course_003` | `mentions_international_office` | Course drop / full-time risk, ISO not mentioned |
| `eval_v3_health_006` | `has_steps` | Insurance appeal answered in paragraph form |
| `eval_v3_health_009` | `mentions_official_office` | Immunization hold, no named office |
| `v7_eval_course_cpt_001` | `mentions_international_office` | CPT scenario, ISO not mentioned (new in v2) |
| `v7_eval_health_immunization_001` | `mentions_official_office` | Immunization hold, no named office (persistent) |

The six failures cluster into three patterns:
- `mentions_international_office` × 2: course drop and CPT scenarios
- `mentions_official_office` × 2: immunization hold scenarios
- `has_steps` × 1 and `no_extra_notes` × 1: formatting issues

## Decision

Do not promote `outputs/dpo_r32_attn_mlp_v2` over `outputs/final_sft_r32_attn_mlp` yet.

- `outputs/final_sft_r32_attn_mlp` remains the stable Project 1 final SFT checkpoint (98.39% rule pass rate).
- `outputs/dpo_r32_attn_mlp_v2` is the best DPO checkpoint to date on score margin (+0.20 over SFT-only).
- Rule pass rate has not recovered to the SFT-only baseline.

## Recommendation

DPO v2 confirms the pipeline is working and score margin is improving, but rule pass rate is stuck. Before expanding to 150–200 pairs, the rule regressions must be addressed.

**Condition for expansion to 150–200 pairs:**
- Rule pass rate ≥ 98.39% (recover SFT-only baseline)
- Preference win rate on a harder eval set (near-miss rejected answers) ≥ 90%
- No increase in prompt leakage or truncation

**Recommended next step — DPO v3 targeted pass:**
1. Add 4–6 new pairs specifically for `mentions_international_office` (CPT authorization, full-time enrollment risk) — two failures now exist for this pattern.
2. Add 2–4 new pairs specifically for `mentions_official_office` in immunization-hold contexts — two failures persist across v1 and v2.
3. Rebuild the eval set with near-miss rejected answers to restore its ability to distinguish SFT-only from DPO.
4. If v3 recovers rule pass rate without degrading score margin, expand DPO data to 150–200 pairs.
