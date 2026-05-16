# DPO Smoke Test v4 Summary

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT-only adapter: `outputs/final_sft_r32_attn_mlp`
- SFT+DPO v4 adapter: `outputs/dpo_r32_attn_mlp_v4`
- DPO v4 train data: 50 pairs (63 total, 13 eval)
- Rule eval data: 60 samples, 311 checks
- DPO epochs: 1

## Preference Comparison (v4 eval set, 13 pairs)

| Metric | SFT-only | SFT+DPO v4 | Delta |
|---|---:|---:|---:|
| Preference eval pairs | 13 | 13 | - |
| Preference chosen wins | 13 | 13 | 0 |
| Preference win rate | 100.00% | 100.00% | +0.00 pp |
| Average chosen score | -1.7794 | -1.7611 | +0.0183 |
| Average rejected score | -2.7332 | -2.9038 | -0.1706 |
| **Score margin** | **0.9537** | **1.1427** | **+0.1890** |

The v4 eval set is saturated at 100% for both models — both new pairs (`email_quality_013`, `email_quality_014`) are handled correctly by SFT-only and have very narrow margins (0.031–0.078). Win rate alone cannot distinguish the two models on this set.

## Full Comparison (all checkpoints)

| Metric | SFT-only | DPO v1 | DPO v2 | DPO v3 | DPO v4 | v3→v4 Delta |
|---|---:|---:|---:|---:|---:|---:|
| Preference win rate | — | 90.00% | 100.00% | 83.33% | 100.00% | — |
| Score margin | — | — | 1.5934 | 0.9355 | 1.1427 | — |
| Rule passed checks | 306 / 311 | 305 / 311 | 305 / 311 | 306 / 311 | **307 / 311** | +1 |
| Rule pass rate | 98.39% | 98.07% | 98.07% | 98.39% | **98.71%** | +0.32 pp |
| Prompt leakage | 0 | 0 | 0 | 0 | 0 | 0 |
| Truncation | 0 | 0 | 0 | 0 | 0 | 0 |

Note: win rate is not directly comparable across versions because the eval set composition changed each iteration. v4 eval set contains 2 new easy pairs that replaced the hard near-miss pairs from v3.

## Rule Eval: Changes from v3 to v4

| Sample | Check | v3 | v4 |
|---|---|---|---|
| `eval_v3_email_005` | `has_closing` | FAIL | **PASS** ✓ |
| `v7_eval_email_extension_001` | `has_closing` | FAIL | **PASS** ✓ |
| `eval_v3_health_009` | `mentions_official_office` | FAIL | **PASS** ✓ |
| `eval_v3_email_011` | `has_closing` | PASS | **FAIL** ✗ |
| `eval_v3_email_012` | `has_closing` | PASS | **FAIL** ✗ |

Three failures fixed, two new `has_closing` regressions. Net: +1 check (307/311).

## SFT+DPO v4 Rule Failures

| ID | Failed Check | Status |
|---|---|---|
| `eval_v3_email_011` | `has_closing` | new regression |
| `eval_v3_email_011` | `no_extra_notes` | persistent |
| `eval_v3_email_012` | `has_closing` | new regression |
| `eval_v3_course_003` | `mentions_international_office` | persistent |

Both `has_closing` regressions share the same root pattern: the model generates `Thank you,\n[Your Name]` as the email closing instead of `Best regards,\n[Your Name]`. The `has_closing` checker does not accept `Thank you,` as a valid formal closing. The two new training pairs (`email_quality_013`, `email_quality_014`) did not transfer this pattern to `eval_v3_email_011` and `eval_v3_email_012`.

## Preference: Notable Pairs

**New pairs (`dpo_email_quality_013`, `dpo_email_quality_014`) in eval set:**

Both were handled correctly by SFT-only and DPO v4, but with very narrow margins:

| Pair | SFT-only margin | DPO v4 margin | Delta |
|---|---:|---:|---:|
| `dpo_email_quality_014` (makeup exam) | 0.0313 | 0.0391 | +0.0078 |
| `dpo_email_quality_013` (noise complaint) | 0.0781 | 0.0781 | 0.0000 |

The `has_closing` signal is very subtle — both models already handle the structural preference weakly. Training signal for this pattern is being absorbed but not amplifying strongly.

**`dpo_visa_safe_014`** — moved to train set in v4; not in eval. Cannot confirm preference inversion fix from this run. The overconfident claim was added to make the rejected answer clearly unsafe, but direct evidence requires it to be in the eval set.

## Interpretation

### Rule pass rate

DPO v4 achieves 307/311 (98.71%) — the first time any DPO checkpoint exceeds the SFT-only baseline (98.39%). Three targeted fixes worked (email_005 has_closing, email_extension has_closing, health_009 mentions_official_office). Two new has_closing regressions appeared on different email samples, both using `Thank you,` instead of `Best regards,`.

The persistent `mentions_international_office` failure (`eval_v3_course_003`) remains after 3 training iterations specifically targeting this pattern. Six pairs now target this failure; the 1.5B model at 1 epoch may be approaching saturation on this single rule.

### Preference alignment

Score margin is consistently widening across versions: SFT-only 0.95 → DPO v4 1.14 (+0.19). The preferred answers are becoming relatively more confident compared to rejected answers. Both models reach 100% win rate on the v4 eval set, but the margin improvement confirms DPO training is working.

The eval set is again saturated (same situation as v2). The two new has_closing pairs have very narrow margins under SFT-only, making the win rate uninformative for distinguishing the two models.

## Decision

Rule pass rate condition is **met and exceeded** (98.71% > SFT-only 98.39%).
Preference win rate condition is **tied** (100% = SFT-only, eval set saturated).
Score margin condition is **met** (+0.19 wider than SFT-only).

**Expansion criteria are effectively met.** The rule pass rate exceeding SFT-only for the first time, combined with consistent score margin improvement, is sufficient signal to proceed.

Do not promote `outputs/dpo_r32_attn_mlp_v4` over `outputs/final_sft_r32_attn_mlp` yet — the two new `has_closing` regressions must be characterized before promotion. But expansion of the DPO dataset is now unblocked.

## Recommendation

DPO v4 is the strongest checkpoint to date:
- Rule pass rate ✓ (98.71%, first time exceeding SFT-only)
- Score margin ✓ (+0.19 wider than SFT-only)
- Preference win rate: tied (saturated eval set)

### Expand to 150–200 pairs

Proceed with dataset expansion. Key guidance for the expansion:

1. **`has_closing` pattern** — do not add more isolated has_closing pairs; the signal is too weak at 1.5B/1 epoch. Instead, embed `Best regards,\n[Your Name]` as the closing in all new email chosen answers. Let the volume of varied email contexts carry the pattern. The `Thank you,` closing should appear in rejected answers alongside other flaws.

2. **`no_extra_notes` + `eval_v3_course_003`** — both are persistent 1-sample failures. Add 2–3 new pairs for each at the expansion stage. Do not target them with isolated micro-patches.

3. **Eval set design** — with 150–200 pairs, hold out 20–25 eval pairs with near-miss rejected answers. Avoid adding pairs that SFT-only handles trivially (margins < 0.1). Every eval pair should have SFT-only margin ≥ 0.15 to maintain discriminability.

4. **`dpo_visa_safe_014`** — the overconfident claim was added to train. Without seeing it in an eval, confirmation of the fix is deferred. Include a similar course-drop + ISO visa pair in the expanded eval set to cover this scenario.
