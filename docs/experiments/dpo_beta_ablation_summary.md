# DPO Beta Ablation Summary

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT adapter: `outputs/final_sft_r32_attn_mlp`
- DPO data: 151 pairs, 121 train / 30 eval
- Rule eval: 60 samples, 311 checks
- Epochs: 1 | LR: 5e-6 | LoRA r=32 | Seed: 42

## Preference Eval (30-pair eval set)

| Beta | Wins | Total | Win Rate | Chosen Score | Rejected Score | Margin |
|---|---:|---:|---:|---:|---:|---:|
| SFT-only (baseline) | 25 | 30 | 83.33% | -1.6777 | -2.2695 | 0.5918 |
| 0.05 | 25 | 30 | 83.33% | -1.6486 | -2.3736 | 0.7250 |
| 0.10 (baseline) | 26 | 30 | 86.67% | -1.6492 | -2.3746 | **0.7254** |
| 0.30 | 27 | 30 | **90.00%** | -1.6499 | -2.3725 | 0.7227 |

## Rule Eval (311 checks)

| Beta | Passed | Total | Pass Rate | Leakage | Truncation | no_extra_notes | intl_office | official_office | has_closing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 306 | 311 | 98.39% | 0 | 0 | 0 | 2 | 2 | 0 |
| 0.10 (baseline) | **309** | 311 | **99.36%** | 0 | 0 | 0 | 2 | 0 | **0** |
| 0.30 | 308 | 311 | 99.04% | 0 | 0 | 0 | 1 | 2 | 0 |

## Decision

**Best beta: 0.10**

Beta=0.10 achieves the best rule pass rate (309/311 = 99.36%) — the highest across all DPO checkpoints to date — while maintaining a clear preference win rate advantage over SFT-only (86.67% vs 83.33%) and the widest score margin (0.7254).

Beta=0.30 achieves a higher preference win rate (90.00%) but regresses on rule pass rate (308/311 = 99.04%) and reintroduces `mentions_official_office` failures that beta=0.10 eliminated. The +3.33 pp win rate gain does not justify the rule regression for a safety-relevant campus assistant.

Beta=0.05 fails to improve over SFT-only on preference win rate (both 83.33%), making it strictly dominated by beta=0.10.

**Confirmed best config for 1.5B:** `configs/dpo_beta_010.yaml` (identical to `configs/dpo_lora_v5.yaml`). DPO v5 remains the promoted checkpoint.

## Qualitative Notes

- `has_closing` failures are **0 across all betas** — the 151-pair dataset expansion resolved this persistent oscillating failure from v1–v4. Volume of varied email contexts with `Best regards,` closing carried the pattern as intended.
- `no_extra_notes` failures are **0 across all betas** — also resolved by expansion.
- `mentions_international_office` persists at 1–2 failures across all betas. This is confirmed as a capacity ceiling at 1.5B/1 epoch, independent of beta. It will be revisited at 7B scale.
- Score margins for all three betas are nearly identical (~0.725) and all substantially better than SFT-only (0.592). Beta has little effect on how far chosen/rejected scores separate; its main effect is on win rate and rule stability.
- The beta=0.10 run on A100 achieves 309/311 vs the earlier v5 run on L4 which achieved 306/311, with identical config and seed. This suggests minor non-determinism from hardware/CUDA differences, not a systematic improvement.

## Future Work

Preference data quality / label noise robustness can be tested later using 10% and 30% chosen-rejected swap noise.
