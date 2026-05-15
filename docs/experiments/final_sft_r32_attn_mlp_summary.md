# Final SFT r32 attn_mlp Summary

## Selected Configuration

- Selected model: `Qwen/Qwen2.5-1.5B-Instruct`
- Method: LoRA SFT
- Final config: `configs/final_sft_r32_attn_mlp.yaml`
- Output dir: `outputs/final_sft_r32_attn_mlp`
- Rank: `32`
- LoRA alpha: `64`
- Target modules: attention + MLP
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

## Evaluation

- Eval set: 60 samples
- Total checks: 311
- Pass rate: 98.39%
- Prompt leakage: 0
- Truncation: 0

## Main Conclusion

The final SFT model has stable email formatting, safe escalation behavior, and a clean generation boundary. It is the best Project 1 SFT checkpoint based on the current rule evaluation.

The remaining failed checks are small compared with earlier runs and are no longer dominated by prompt leakage, stop-sequence truncation, or very short responses.

## Adapter Registration Notes

If the trained ablation adapter already exists in Colab at `outputs/ablations/sft_r32_attn_mlp`, register it as the final adapter with:

```bash
mkdir -p outputs/final_sft_r32_attn_mlp
cp -r outputs/ablations/sft_r32_attn_mlp/* outputs/final_sft_r32_attn_mlp/
cp outputs/ablations/target_modules_reports/eval_report_sft_r32_attn_mlp.md outputs/final_sft_r32_attn_mlp/eval_report.md
cp outputs/ablations/target_modules_reports/eval_report_sft_r32_attn_mlp.json outputs/final_sft_r32_attn_mlp/eval_report.json
cp configs/final_sft_r32_attn_mlp.yaml outputs/final_sft_r32_attn_mlp/final_config.yaml
```

The final adapter is large. The `qv` adapter is kept as a lightweight alternative when adapter size or transfer cost matters.
