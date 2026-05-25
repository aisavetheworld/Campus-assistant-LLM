# DPO 7B Summary

## Setup

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- SFT adapter: `outputs/final_sft_7b`
- DPO adapter: `outputs/dpo_7b`
- SFT data: 176 train / 44 eval
- DPO data: 151 pairs, 121 train / 30 eval (same as 1.5B v5)
- Rule eval: 60 samples, 311 checks
- SFT: 1 epoch, lr=2e-4, LoRA r=32, batch=2, accum=4
- DPO: 1 epoch, lr=5e-6, beta=0.1, LoRA r=32, batch=1, accum=8

## Full Comparison (1.5B vs 7B)

| Metric | 1.5B SFT | 1.5B DPO v5 | 7B SFT | 7B DPO |
|---|---:|---:|---:|---:|
| Rule passed checks | 306 / 311 | 306 / 311 | 305 / 311 | 306 / 311 |
| Rule pass rate | 98.39% | 98.39% | 98.07% | **98.39%** |
| Preference win rate | 83.33% | 86.67% | 76.67% | **90.00%** |
| Score margin | 0.5918 | 0.7254 | 0.7276 | **1.0522** |
| Prompt leakage | 0 | 0 | 0 | 0 |
| Truncation | 0 | 0 | 0 | 0 |

## Rule Eval: 7B SFT vs 7B DPO

| Check | 7B SFT | 7B DPO | Delta |
|---|---:|---:|---:|
| `no_absolute_promise` | 2 | **4** | +2 (regression) |
| `no_extra_notes` | 3 | 1 | -2 (fixed) |
| `mentions_international_office` | 1 | **0** | -1 (fixed) |
| `mentions_official_office` | 0 | 0 | — |
| `has_closing` | 0 | 0 | — |

## Preference Eval: Per-Category

| Category | 7B SFT | 7B DPO |
|---|---:|---:|
| `academic` | 6/6 (100%) | 6/6 (100%) |
| `course_enrollment` | 6/7 (85.71%) | **7/7 (100%)** |
| `health_insurance` | 6/9 (66.67%) | 7/9 (77.78%) |
| `housing` | 5/8 (62.50%) | 7/8 (87.50%) |

## Training Metrics

| Metric | 1.5B DPO v5 | 7B DPO |
|---|---:|---:|
| train_loss | 0.5571 | **0.4398** |
| eval_loss | — | **0.3265** |
| Final rewards/margins | ~0.61 | **~1.36** |
| logits/chosen (final) | negative (~-0.6) | **positive (~+0.5)** |
| train_runtime | 57s (L4) | 68s (A100) |

## Key Findings

### 1. `mentions_international_office` resolved at 7B scale
The persistent 1-2 failure across all 1.5B versions (v1–v5, all beta values) disappears completely at 7B. Confirms this was a model capacity limitation, not a data coverage issue. No additional DPO pairs were needed.

### 2. DPO improvement is much larger at 7B
- 1.5B: preference win rate +3.34 pp (83.33% → 86.67%), margin +0.13
- 7B: preference win rate **+13.33 pp** (76.67% → 90.00%), margin **+0.32**

The 7B model has sufficient capacity to internalize the DPO preference signal more effectively. Score margin of 1.052 (vs 0.725 for 1.5B DPO) indicates much stronger separation between chosen and rejected responses.

### 3. `no_absolute_promise` is the new failure pattern at 7B
7B SFT already exhibits 2 failures (vs 0 in 1.5B SFT). DPO amplifies this to 4. The 7B model's higher confidence (positive logits, larger rewards/margins) appears to cause overconfident phrasing in some responses. This is a new failure mode not seen in 1.5B.

### 4. 7B SFT preference win rate is lower than 1.5B SFT
7B SFT: 76.67% vs 1.5B SFT: 83.33%. The 7B base model has different prior preferences that diverge more from the DPO eval set design. DPO corrects this strongly (+13.33 pp), ending at 90.00%.

## Decision

**Promote `outputs/dpo_7b` as the new serving checkpoint**, replacing `outputs/dpo_r32_attn_mlp_v5`.

- Rule pass rate: 98.39% (meets threshold, same as 1.5B DPO v5) ✓
- Preference win rate: 90.00% > 7B SFT-only 76.67% ✓
- Score margin: 1.0522 > 0.7276 ✓
- `mentions_international_office` resolved ✓

## Outstanding Issues

**`no_absolute_promise`: 4 failures in DPO 7B**

The model generates overconfident statements in 4 samples. This is the primary remaining issue. Options:
1. Add DPO pairs that penalize absolute/overconfident claims in chosen answers (rejected: overconfident, chosen: appropriately hedged)
2. Run a beta ablation at 7B scale — lower beta (0.05) may reduce overconfidence amplification
3. Accept as-is if overconfidence is minor and not safety-critical in context

Investigation of which samples fail `no_absolute_promise` is recommended before deciding on a fix.

## Next Steps

- Investigate `no_absolute_promise` failure samples
- Consider targeted DPO correction pairs for overconfident phrasing
- Proceed to Project 2 (serving infrastructure)
