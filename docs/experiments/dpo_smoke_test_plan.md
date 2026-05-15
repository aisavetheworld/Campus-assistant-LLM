# DPO Smoke Test Plan

## Purpose

This smoke test checks whether a small DPO run can improve preference alignment on top of the final SFT adapter without changing the Project 1 SFT data or final SFT artifact.

DPO is not expected to be a final alignment run yet. The goal is to verify:

- the DPO data format works with TRL `DPOTrainer`;
- the final SFT adapter can be used as both the policy initialization and frozen reference model;
- DPO training completes for one epoch on the 40-pair train split;
- preference win rate and rule-based behavior can be compared against SFT-only.

## Starting Point

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT adapter: `outputs/final_sft_r32_attn_mlp`
- DPO train data: `data/dpo/dpo_train.jsonl` with 40 pairs
- DPO eval data: `data/dpo/dpo_eval.jsonl` with 10 pairs
- DPO output: `outputs/dpo_r32_attn_mlp_v1`
- Training length: 1 epoch only

## Model Setup

The policy model is initialized from the final SFT adapter:

```text
base model + outputs/final_sft_r32_attn_mlp
```

The reference model is a frozen copy of the same SFT model:

```text
frozen(base model + outputs/final_sft_r32_attn_mlp)
```

This makes the smoke test compare whether DPO improves the SFT policy relative to the SFT reference, not relative to the raw base model.

## L4 Efficiency Settings

The first smoke-test config is tuned for a Colab L4-style GPU:

- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 4`
- effective train batch size: 8
- `per_device_eval_batch_size: 2`
- `max_prompt_length: 768`
- `max_length: 1280`
- `gradient_checkpointing: true`
- `torch_dtype: bfloat16`

If memory pressure occurs, reduce `per_device_train_batch_size` to `1` and keep `gradient_accumulation_steps: 4`.

The default reference setup is:

```yaml
model:
  reference_model_mode: "separate"
```

This loads a separate frozen SFT reference model. If L4 memory pressure occurs before training starts, use this fallback only for the smoke test:

```yaml
model:
  reference_model_mode: "trainer_managed"
```

That lets the installed TRL version manage the reference behavior instead of explicitly loading the second SFT copy.

## Commands

Build or refresh DPO data:

```bash
python scripts/build_dpo_data.py \
  --input_file data/dpo/dpo_seed.json \
  --train_output data/dpo/dpo_train.jsonl \
  --eval_output data/dpo/dpo_eval.jsonl \
  --audit_md docs/experiments/dpo_data_audit.md \
  --eval_ratio 0.2 \
  --seed 42
```

Run the DPO smoke test:

```bash
python scripts/train_dpo.py --config configs/dpo_lora.yaml
```

Evaluate DPO preference win rate on the held-out DPO eval split:

```bash
python scripts/eval_dpo_preference.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/dpo_r32_attn_mlp_v1 \
  --eval_file data/dpo/dpo_eval.jsonl \
  --output_json outputs/dpo_r32_attn_mlp_v1/preference_eval.json \
  --output_md outputs/dpo_r32_attn_mlp_v1/preference_eval.md \
  --batch_size 2 \
  --torch_dtype bfloat16
```

Run the existing rule-based generation evaluation for SFT+DPO:

```bash
python scripts/eval_sft.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/dpo_r32_attn_mlp_v1 \
  --eval_file data/eval/eval_seed.json \
  --output_json outputs/dpo_r32_attn_mlp_v1/rule_eval.json \
  --output_md outputs/dpo_r32_attn_mlp_v1/rule_eval.md \
  --max_new_tokens 220 \
  --temperature 0 \
  --prompt_template chat \
  --eval_batch_size 4 \
  --torch_dtype bfloat16
```

## Expected Comparison

Compare SFT-only against SFT+DPO on:

- SFT-only rule pass rate;
- SFT+DPO rule pass rate;
- DPO preference win rate;
- raw/final prompt leakage;
- truncation count;
- `no_extra_notes` failures;
- safe escalation behavior;
- `not_too_short`, `has_steps`, and official-office mention failures.

## Success Criteria

The smoke test is useful if:

- training completes for 1 epoch;
- preference eval runs on 10 held-out pairs;
- rule eval runs on the same 60-sample SFT eval set;
- prompt leakage remains 0 or does not regress materially;
- DPO does not harm email formatting or safety escalation.

If DPO improves preference win rate but hurts rule eval, keep the final SFT adapter as the Project 1 best model and treat DPO as an experimental branch.
