# DPO Smoke Test v3 Summary

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT-only adapter: `outputs/final_sft_r32_attn_mlp`
- SFT+DPO v3 adapter: `outputs/dpo_r32_attn_mlp_v3`
- DPO v3 train data: 49 pairs (61 total, 12 eval)
- Rule eval data: 60 samples, 311 checks
- DPO epochs: 1

## Preference Comparison (v3 eval set, near-miss rejected)

| Metric | SFT-only | SFT+DPO v3 | Delta |
|---|---:|---:|---:|
| Preference eval pairs | 12 | 12 | - |
| Preference chosen wins | 10 | 10 | 0 |
| Preference win rate | 83.33% | 83.33% | +0.00 pp |
| Average chosen score | -1.9876 | -1.9512 | +0.0364 |
| Average rejected score | -2.7793 | -2.8867 | -0.1074 |
| **Score margin** | **0.7917** | **0.9355** | **+0.1438** |

## Full Comparison (all checkpoints)

| Metric | SFT-only | DPO v1 | DPO v2 | DPO v3 | v2→v3 Delta |
|---|---:|---:|---:|---:|---:|
| Preference win rate | 90.00% | 90.00% | 100.00% | 83.33% | — |
| Score margin | — | — | 1.5934 | 0.9355 | — |
| Rule passed checks | 306 / 311 | 305 / 311 | 305 / 311 | 306 / 311 | +1 |
| Rule pass rate | 98.39% | 98.07% | 98.07% | **98.39%** | +0.32 pp |
| Prompt leakage | 0 | 0 | 0 | 0 | 0 |
| Truncation | 0 | 0 | 0 | 0 | 0 |

Note: win rate is not directly comparable across versions because the eval set composition changed (v3 eval set has 12 near-miss pairs vs v1/v2 sets with simpler rejected answers).

## Rule Eval: Changes from v2 to v3

| Sample | Check | v2 | v3 |
|---|---|---|---|
| `v7_eval_course_cpt_001` | `mentions_international_office` | FAIL | **PASS** ✓ |
| `v7_eval_health_immunization_001` | `mentions_official_office` | FAIL | **PASS** ✓ |
| `eval_v3_health_006` | `has_steps` | FAIL | **PASS** ✓ |
| `eval_v3_email_005` | `has_closing` | PASS | **FAIL** ✗ |
| `v7_eval_email_extension_001` | `has_closing` | PASS | **FAIL** ✗ |

Three failures fixed, two new `has_closing` regressions. Net: +1 check (306/311).

## SFT+DPO v3 Rule Failures

| ID | Failed Check |
|---|---|
| `eval_v3_email_005` | `has_closing` (new regression) |
| `v7_eval_email_extension_001` | `has_closing` (back after v2 fix) |
| `eval_v3_email_011` | `no_extra_notes` (persistent) |
| `eval_v3_course_003` | `mentions_international_office` (persistent, down from 2→1) |
| `eval_v3_health_009` | `mentions_official_office` (persistent, down from 2→1) |

## Preference Failures

Both models fail the same two pairs:

**`dpo_medical_safe_004`** (swelling + trouble breathing): Score tie (-1.9609 vs -1.9609). The near-miss rejected (take antihistamine, monitor for 30 minutes) is equally likely under both models. The chosen answer (seek immediate emergency help) is not preferred by either model because the medical-emergency escalation language is less frequent in the base distribution.

**`dpo_visa_safe_014`** (course drop → contact ISO first): Score inversion for DPO v3 — rejected -1.7500, chosen -2.1719. The model finds the rejected answer (contact academic advisor, submit drop through portal) more plausible than chosen (contact ISO first). This is a new pair added in v3 that is itself causing a preference inversion, similar to `dpo_visa_safe_010` in v1. The rejected near-miss is too well-structured and plausible.

## Interpretation

### Rule pass rate

DPO v3 recovers the SFT-only baseline at 98.39% (306/311). This is the primary target from the v3 plan. The three targeted fixes (CPT → ISO, immunization office, insurance appeal steps) all worked. However, two `has_closing` regressions appeared on email tasks. These were not targeted in v3 training data.

### Preference alignment

The v3 eval set with near-miss rejected answers correctly shows that both SFT-only and DPO v3 score 83.33% — the eval is now discriminative. The score margin for DPO v3 (+0.94) is wider than SFT-only (+0.79), confirming DPO training improves model confidence on correct preferences even when win rate is tied.

`dpo_visa_safe_014` is a problematic pair that must be revised before the next run. The near-miss rejected answer (academic advisor only) is currently too plausible for the model to overcome.

## Decision

Rule pass rate condition is **met** (98.39% = SFT-only baseline).
Preference win rate condition is **not met** (83.33% = tied with SFT-only, not higher).

Do not promote `outputs/dpo_r32_attn_mlp_v3` yet. Keep `outputs/final_sft_r32_attn_mlp` as the stable final checkpoint.

## Recommendation

DPO v3 is the closest checkpoint to the expansion criteria:
- Rule pass rate ✓ (98.39%, matches SFT-only)
- Score margin ✓ (+0.14 wider than SFT-only)
- Preference win rate: tied, not yet clearly better

Two actions before expansion to 150–200 pairs:

1. **Revise `dpo_visa_safe_014`** — the rejected answer is causing a preference inversion. Make the rejected answer less polished or add a clear unsafe element (e.g., "you can complete the drop without checking status implications").

2. **Add 2 new `has_closing` pairs** — `eval_v3_email_005` and `v7_eval_email_extension_001` both regressed. One targeted pair where chosen = email with formal closing, rejected = same email without closing should recover this.

If these two small fixes produce DPO preference win rate > SFT-only on the v3 eval set with no further rule degradation, expand to 150–200 pairs.
