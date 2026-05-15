# DPO v2 Data Revision Notes

## Scope

This revision keeps the DPO dataset size unchanged at 50 preference pairs. It does not modify SFT data, RAG, vLLM, FastAPI, or deployment code.

The purpose is to target the failure modes observed in DPO smoke test v1 before running another smoke test.

## Revised Pairs

| Pair ID | Split | Targeted Issue | Change |
|---|---|---|---|
| `dpo_visa_safe_010` | eval | Failed DPO preference pair for SEVIS/status email | Strengthened the chosen answer with `do not wait for another reminder`, explicit deadline/action question, and international student office escalation. Made the rejected answer less polished and clearly unsafe by recommending waiting. |
| `dpo_visa_safe_005` | train | Full-time/drop international-office escalation | Added that professor approval alone may not resolve full-time enrollment risk. Rejected answer now over-relies on professor approval and later explanation. |
| `dpo_visa_safe_006` | train | Under full-time/random class issue | Strengthened international student office timing/authorization language and made rejected answer a non-step paragraph that recommends adding any easy class later. |
| `dpo_medical_safe_005` | train | Immunization hold official-office escalation | Added `student health center`, `immunization office`, and `relevant official office` language to chosen answer. Rejected answer recommends waiting without official escalation. |
| `dpo_medical_safe_006` | eval | Insurance waiver/claim appeal steps | Revised chosen answer into concrete numbered appeal/correction steps with insurance office/provider escalation and no approval guarantee. Rejected answer implies limited options and lacks steps. |
| `dpo_email_quality_006` | eval | Follow-up email no-extra-notes and closing control | Repurposed the pair from recommendation-letter request to short professor follow-up email. Chosen stops after `Best regards`; rejected adds after-email instructions and uses a weaker closing. |

## Audit After Revision

- Pair count: 50
- Train split: 40
- Eval split: 10
- Average chosen word count: 62.08
- Average rejected word count: 38.32
- Average chosen/rejected length ratio: 1.69
- Chosen missing official office when required: 0
- Chosen bad marker count: 0

Rejected flaw counts:

| Flaw | Count |
|---|---:|
| `no_official_office` | 34 |
| `missing_steps` | 12 |
| `poor_email_format` | 9 |
| `overconfident_claim` | 4 |

## Next Run

Run DPO smoke test v2 with the same training settings as v1 but a separate output directory:

```bash
python scripts/train_dpo.py --config configs/dpo_lora_v2.yaml
```

Then evaluate:

```bash
python scripts/eval_dpo_preference.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/dpo_r32_attn_mlp_v2 \
  --eval_file data/dpo/dpo_eval.jsonl \
  --output_json outputs/dpo_r32_attn_mlp_v2/preference_eval.json \
  --output_md outputs/dpo_r32_attn_mlp_v2/preference_eval.md \
  --batch_size 8 \
  --torch_dtype bfloat16
```

```bash
python scripts/eval_sft.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/dpo_r32_attn_mlp_v2 \
  --eval_file data/eval/eval_seed.json \
  --output_json outputs/dpo_r32_attn_mlp_v2/rule_eval.json \
  --output_md outputs/dpo_r32_attn_mlp_v2/rule_eval.md \
  --max_new_tokens 220 \
  --temperature 0 \
  --prompt_template chat \
  --eval_batch_size 8 \
  --torch_dtype bfloat16
```

This preserves `outputs/dpo_r32_attn_mlp_v1` for comparison.
