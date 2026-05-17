# DPO Beta Ablation Plan

## Objective

Find the best beta value for DPO training on the campus assistant dataset.
Beta controls the KL-divergence penalty between the policy and the frozen SFT reference model.

- **Smaller beta (0.05):** more conservative; stays closer to SFT behavior; lower risk of rule regression
- **Current beta (0.10):** DPO v5 baseline
- **Larger beta (0.30):** pushes harder toward chosen responses; higher preference signal but may degrade rule consistency

## Hypothesis

DPO v5 at beta=0.10 achieves 86.67% preference win rate with rule pass rate tied at SFT-only (98.39%). The oscillating rule failures (`has_closing`, `mentions_official_office`, `mentions_international_office`) may be sensitive to beta:

- beta=0.05 may reduce oscillation by staying closer to SFT, at the cost of weaker preference signal
- beta=0.30 may improve preference win rate further but risk amplifying rule regressions

## Controlled Variables

All three runs are identical except for beta:

| Variable | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| SFT adapter | `outputs/final_sft_r32_attn_mlp` |
| DPO train file | `data/dpo/dpo_train.jsonl` (121 pairs) |
| DPO eval file | `data/dpo/dpo_eval.jsonl` (30 pairs) |
| Epochs | 1 |
| Learning rate | 5e-6 |
| Per-device train batch | 4 |
| Gradient accumulation | 2 (effective batch = 8) |
| LoRA r / alpha | 32 / 64 |
| Target modules | q/k/v/o + gate/up/down |
| Seed | 42 |
| Prompt template | chat |

## Beta Variants

| Config | Beta | Output dir |
|---|---|---|
| `configs/dpo_beta_005.yaml` | 0.05 | `outputs/dpo_beta_005` |
| `configs/dpo_beta_010.yaml` | 0.10 | `outputs/dpo_beta_010` |
| `configs/dpo_beta_030.yaml` | 0.30 | `outputs/dpo_beta_030` |

Note: `dpo_beta_010` reproduces DPO v5 exactly (same config, different output dir). Used as the ablation's internal baseline.

## Evaluation Protocol

For each adapter, run:

1. **Preference eval** — `scripts/eval_dpo_preference.py` on `data/dpo/dpo_eval.jsonl`
2. **Rule eval** — `scripts/eval_sft.py` on `data/eval/eval_seed.json` with `--eval_batch_size 16`

Also run preference eval on SFT-only adapter as the external baseline.

## Decision Criteria

| Metric | Threshold |
|---|---|
| Preference win rate | Must exceed SFT-only 83.33% |
| Score margin | Must exceed SFT-only 0.5936 |
| Rule pass rate | Must be ≥ 98.39% (SFT-only baseline) |
| Prompt leakage | Must be 0 |
| Truncation | Must be 0 |

Best beta = highest preference win rate among configs that meet all thresholds.

## Metrics to Record

- Preference win rate (per category + overall)
- Score margin (avg chosen − avg rejected)
- Rule pass rate (X / 311)
- Prompt leakage count
- Truncation count
- `no_extra_notes` failure count
- Official escalation failure count (`mentions_international_office` + `mentions_official_office`)
- `has_closing` failure count
- Qualitative notes on any new failure patterns

## Future Work

Preference data quality / label noise robustness can be tested later using 10% and 30% chosen-rejected swap noise.
