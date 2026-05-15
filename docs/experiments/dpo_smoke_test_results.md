# DPO Smoke Test Results

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT starting adapter: `outputs/final_sft_r32_attn_mlp`
- DPO output adapter: `outputs/dpo_r32_attn_mlp_v1`
- DPO train data: 40 pairs
- DPO eval data: 10 pairs
- DPO epochs: 1
- Rule eval set: 60 samples, 311 checks

## Preference Evaluation

| Metric | Value |
|---|---:|
| Total pairs | 10 |
| Chosen wins | 9 |
| Chosen win rate | 90.00% |
| Average chosen score | -1.8695 |
| Average rejected score | -2.5938 |

### Per-Category Win Rate

| Category | Wins | Total | Win Rate |
|---|---:|---:|---:|
| `course_enrollment` | 2 | 3 | 66.67% |
| `email_drafting` | 1 | 1 | 100.00% |
| `health_insurance` | 5 | 5 | 100.00% |
| `housing` | 1 | 1 | 100.00% |

### Preference Eval Failure

| ID | Category | Risk | Chosen Score | Rejected Score |
|---|---|---|---:|---:|
| `dpo_visa_safe_010` | `course_enrollment` | `high` | -2.0938 | -1.8906 |

The only preference loss is a high-risk visa/status email scenario. This should be inspected before expanding DPO data or running a larger DPO experiment.

## Rule-Based Generation Evaluation

| Metric | SFT+DPO |
|---|---:|
| Total samples | 60 |
| Total checks | 311 |
| Passed checks | 305 |
| Pass rate | 98.07% |
| Raw prompt leakage | 0 |
| Final prompt leakage | 0 |
| Truncated count | 0 |
| Early truncation count | 0 |
| Not-too-short failures | 0 |

For comparison, the selected final SFT adapter previously reached 98.39% on the same rule-eval setup. The DPO smoke test is therefore roughly neutral on rule-based generation quality, with a small drop of 0.32 percentage points.

## Failed Rule Checks

| Check | Count |
|---|---:|
| `mentions_official_office` | 2 |
| `no_extra_notes` | 1 |
| `mentions_international_office` | 1 |
| `has_steps` | 1 |
| `has_closing` | 1 |

## Interpretation

The DPO smoke test is successful as a pipeline validation:

- DPO training completed for one epoch.
- Preference win rate is high at 90.00%.
- Prompt leakage remains 0.
- Truncation remains 0.
- Rule-eval quality remains close to the final SFT baseline.
- No not-too-short regression appears after DPO.

However, the DPO adapter should be treated as an experimental SFT+DPO checkpoint rather than replacing the final SFT adapter immediately, because the final SFT adapter has a slightly higher rule pass rate and the DPO eval set is still small.

## Next Recommended Checks

Before larger DPO training:

1. Inspect `dpo_visa_safe_010`, the only preference-eval loss.
2. Inspect the six rule-eval failures, especially `no_extra_notes`, `has_closing`, and `mentions_international_office`.
3. Run a small qualitative SFT-only vs SFT+DPO comparison on the same 8 typical prompts used before.
4. If DPO improves preference and qualitative behavior without harming safety, expand DPO data to 100-200 pairs from observed SFT/DPO failure cases.

## Decision

- Keep `outputs/final_sft_r32_attn_mlp` as the final stable Project 1 SFT checkpoint.
- Keep `outputs/dpo_r32_attn_mlp_v1` as the first DPO smoke-test checkpoint.
- Do not start larger DPO training until the failed cases are reviewed.
