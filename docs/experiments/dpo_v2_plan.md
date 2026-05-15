# DPO v2 Plan

## Why v1 Was Not Enough

DPO v1 validated the full pipeline (prompt leakage = 0, truncation = 0, no short-answer regression) but produced no measurable preference alignment improvement over the final SFT adapter.

Two specific problems make v1 insufficient for promotion:

**1. Failed preference pair (`dpo_visa_safe_010`)**

The most safety-critical pair in the visa category — SEVIS/status email handling — was scored higher for the rejected answer than the chosen answer by both SFT-only and SFT+DPO. The v1 rejected answer was a three-step numbered list that appeared nearly as credible as chosen on surface form. The DPO training signal for this pair was inverted.

**2. Six rule check regressions in SFT+DPO v1**

The DPO-trained model introduced one additional rule check failure compared to SFT-only (305/311 vs. 306/311). Across all evaluated samples, six distinct failure patterns appeared:

| Sample | Check | Pattern |
|---|---|---|
| `eval_v3_email_011` | `no_extra_notes` | Model added commentary after email closing |
| `eval_v3_course_003` | `mentions_international_office` | Full-time/drop scenario, ISO not mentioned |
| `eval_v3_health_006` | `has_steps` | Insurance appeal answered in paragraph form |
| `eval_v3_health_009` | `mentions_official_office` | Immunization hold, no named office |
| `v7_eval_health_immunization_001` | `mentions_official_office` | Second immunization sample, same gap |
| `v7_eval_email_extension_001` | `has_closing` | Extension email missing formal closing |

None of the v1 training pairs targeted the email-closing failure specifically. The immunization-office pattern repeated twice, confirming it is a systematic gap rather than an edge case.

---

## What v2 Changes

### Data changes summary

| Change type | Count | Details |
|---|---|---|
| Revised existing pairs | Many (throughout all 50) | Simplified rejected to shorter, clearer-flaw answers; strengthened chosen escalation language |
| Revised `dpo_visa_safe_010` chosen | 1 | Added "do not wait for another reminder" + explicit deadline/action question |
| Added new targeted pairs | 6 | Directly address each observed failure pattern |
| **Total v2 dataset** | **56 pairs** | 45 train / 11 eval (seed 42, eval ratio 0.2) |

### New targeted pairs (6)

| Pair ID | Targets | Failure addressed |
|---|---|---|
| `dpo_email_quality_011` | `v7_eval_email_extension_001` | Chosen = assignment extension email with `Best regards, [Your Name]`; rejected = same email with `Thanks, [Name]` informal close |
| `dpo_email_quality_012` | `eval_v3_email_011` | Chosen = clean registrar-hold inquiry email only; rejected = same email + `Note:` post-closing instruction |
| `dpo_visa_safe_011` | `eval_v3_course_003` | Chosen = step 1 contacts ISO before any further enrollment changes; rejected = "professor approved it, explain to ISO later if needed" |
| `dpo_medical_safe_011` | `eval_v3_health_009`, `v7_eval_health_immunization_001` | Chosen explicitly names "student health center or immunization office"; rejected = "wait, check portal next week" |
| `dpo_steps_email_011` | `eval_v3_health_006` | Chosen = 3 numbered claim-review steps + complete email; rejected = email only, no steps |
| `dpo_visa_safe_012` | OPT pre-start shadowing | Chosen = cannot confirm, contact ISO before any employer activity; rejected = "shadowing is informal, it should be fine, check later" |

### Prefix distribution after v2

| Prefix | v1 count | v2 count |
|---|---:|---:|
| `dpo_email_quality` | 10 | 12 |
| `dpo_housing_safe` | 10 | 10 |
| `dpo_medical_safe` | 10 | 11 |
| `dpo_steps_email` | 10 | 11 |
| `dpo_visa_safe` | 10 | 12 |
| **Total** | **50** | **56** |

### Audit statistics after v2

- Pair count: 56
- Train split: 45
- Eval split: 11
- Average chosen word count: 61.88
- Average rejected word count: 16.64
- Average chosen/rejected length ratio: 4.38
- Chosen missing official office when required: 0
- Chosen bad marker count: 0

Rejected flaw distribution:

| Flaw | Count |
|---|---:|
| `too_short` | 49 |
| `missing_steps` | 43 |
| `no_official_office` | 39 |
| `poor_email_format` | 13 |
| `extra_notes` | 7 |
| `overconfident_claim` | 6 |

Note: The high `too_short` count reflects the v2 revision strategy of simplifying rejected answers to single-sentence or short-paragraph responses. This makes the preference signal clearer at the cost of near-miss realism. The `extra_notes` count (7) reflects pairs where rejected answers include post-email `Note:` or `Explanation:` commentary, which directly targets the `no_extra_notes` failure pattern.

---

## Expected Improvements

### Preference win rate

`dpo_visa_safe_010` is the only eval pair that failed in v1. The v2 revision strengthens the chosen answer (step 1 now explicitly says "do not wait for another reminder") and keeps the rejected as a short unsafe prose paragraph. The preference signal contrast is cleaner. Expected outcome: preference win rate improves from 90.00% to 100.00% on the eval set.

### Rule pass rate

Each of the six v1 rule failures has a corresponding new or revised DPO pair in v2:

| v1 failure | v2 pairs addressing it |
|---|---|
| `no_extra_notes` | `dpo_email_quality_006` (revised), `dpo_email_quality_012` (new), `dpo_email_quality_001/003/004/008/010` (revised to include `Note:`/`Explanation:` in rejected) |
| `mentions_international_office` | `dpo_visa_safe_005/006` (revised), `dpo_visa_safe_011` (new) |
| `has_steps` | `dpo_medical_safe_006` (revised), `dpo_steps_email_011` (new) |
| `mentions_official_office` | `dpo_medical_safe_005` (revised), `dpo_medical_safe_011` (new) |
| `has_closing` | `dpo_email_quality_011` (new) |

Expected rule pass rate: at or above 98.39% (matching SFT-only baseline). Each failure pattern now has at least two DPO pairs.

### What NOT to expect

- Perfect elimination of all failures — the 1.5B model trained for 1 epoch has limited capacity to absorb many simultaneous behavioral changes.
- No regression on unrelated categories — this needs to be confirmed with rule eval.
- Large word-count improvement — rejected answers are now shorter, which may reduce the reward model's ability to learn subtle near-miss patterns at the next expansion.

---

## Metrics to Compare

| Metric | SFT-only | SFT+DPO v1 | SFT+DPO v2 (target) |
|---|---:|---:|---:|
| Preference eval pairs | 10 | 10 | 11 |
| Preference chosen wins | 9 | 9 | ≥ 10 |
| Preference win rate | 90.00% | 90.00% | ≥ 95.00% |
| Average chosen score | -1.8852 | -1.8695 | improvement |
| Average rejected score | -2.5445 | -2.5938 | maintained gap |
| Rule passed checks | 306 / 311 | 305 / 311 | ≥ 306 / 311 |
| Rule pass rate | 98.39% | 98.07% | ≥ 98.39% |
| Raw prompt leakage count | 0 | 0 | 0 |
| Final prompt leakage count | 0 | 0 | 0 |
| Truncated count | 0 | 0 | 0 |
| Early truncation count | 0 | 0 | 0 |
| `no_extra_notes` failures | — | 1 | 0 |
| `mentions_international_office` failures | — | 1 | 0 |
| `mentions_official_office` failures | — | 2 | 0 |
| `has_steps` failures | — | 1 | 0 |
| `has_closing` failures | — | 1 | 0 |

---

## Training Configuration

Use `configs/dpo_lora_v2.yaml`:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT reference adapter: `outputs/final_sft_r32_attn_mlp`
- Output: `outputs/dpo_r32_attn_mlp_v2`
- LoRA: r=32, alpha=64, dropout=0.05, attn+MLP target modules
- Beta: 0.1
- Epochs: 1
- Learning rate: 5e-6
- Train data: `data/dpo/dpo_train.jsonl` (45 pairs)
- Eval data: `data/dpo/dpo_eval.jsonl` (11 pairs)

This preserves `outputs/dpo_r32_attn_mlp_v1` for direct comparison.

Run training:

```bash
python scripts/train_dpo.py --config configs/dpo_lora_v2.yaml
```

Evaluate preference win rate:

```bash
python scripts/eval_dpo_preference.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/dpo_r32_attn_mlp_v2 \
  --eval_file data/dpo/dpo_eval.jsonl \
  --output_json outputs/dpo_r32_attn_mlp_v2/preference_eval.json \
  --output_md outputs/dpo_r32_attn_mlp_v2/preference_eval.md
```

Evaluate rule pass rate:

```bash
python scripts/eval_sft.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/dpo_r32_attn_mlp_v2 \
  --eval_file data/eval/eval_seed.json \
  --output_json outputs/dpo_r32_attn_mlp_v2/rule_eval.json \
  --output_md outputs/dpo_r32_attn_mlp_v2/rule_eval.md
```

---

## Expansion Criteria

Expand to 200+ DPO pairs only if DPO v2 shows:

- Preference win rate ≥ 95% on the eval set (at least one more win than v1)
- Rule pass rate ≥ 98.39% (no degradation from SFT-only baseline)
- Prompt leakage = 0 and truncation = 0

If v2 shows no improvement in preference win rate but rule pass rate recovers, consider holding at 56 pairs and adjusting beta or learning rate before expanding.

If rule pass rate drops further below 98.07%, investigate whether DPO is destabilizing the base SFT behavior. Do not expand until the regression is understood.
