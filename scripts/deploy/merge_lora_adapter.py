"""
merge_lora_adapter.py
---------------------
Merge a PEFT/LoRA adapter into the base model weights and save the fully merged
model so that vLLM (and other inference servers) can load it directly.

Usage
-----
python scripts/deploy/merge_lora_adapter.py \
    --base_model  Qwen/Qwen2.5-7B-Instruct \
    --adapter_path outputs/dpo_7b \
    --output_dir  outputs/deploy/dpo_7b_merged \
    --torch_dtype bfloat16
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a base model and save the "
                    "merged model for direct loading by vLLM or HuggingFace."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID or local path to the base model "
             "(default: Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/dpo_7b",
        help="Path to the PEFT/LoRA adapter directory produced by training "
             "(default: outputs/dpo_7b).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/deploy/dpo_7b_merged",
        help="Directory where the merged model will be saved "
             "(default: outputs/deploy/dpo_7b_merged).",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype to use when loading and saving the model "
             "(default: bfloat16).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float16": None,   # filled in after import
    "bfloat16": None,
    "float32": None,
}


def _resolve_dtype(dtype_str: str):
    """Return the torch dtype object for a string name."""
    import torch
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def _is_lora_adapter(path: str) -> bool:
    """Return True if *path* looks like a PEFT LoRA adapter directory."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _warn_if_exists(output_dir: str) -> None:
    if os.path.exists(output_dir):
        print(
            f"[WARNING] Output directory already exists: {output_dir}\n"
            "          Existing files may be overwritten.",
            file=sys.stderr,
        )


def _verify_output(output_dir: str) -> None:
    """Check that at least one model shard was written."""
    if not os.path.isdir(output_dir):
        print(
            f"[ERROR] Output directory was not created: {output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    model_files = [
        f for f in os.listdir(output_dir)
        if f.endswith((".bin", ".safetensors", ".json"))
    ]
    if not model_files:
        print(
            f"[ERROR] Output directory exists but contains no model files: "
            f"{output_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n[OK] {len(model_files)} file(s) written to: {output_dir}")
    for fname in sorted(model_files):
        fpath = os.path.join(output_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 ** 2)
        print(f"     {fname}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 0. Validate inputs                                                   #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  LoRA Merge → Merged Model")
    print("=" * 60)
    print(f"  base_model   : {args.base_model}")
    print(f"  adapter_path : {args.adapter_path}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  torch_dtype  : {args.torch_dtype}")
    print("=" * 60)

    if not os.path.exists(args.adapter_path):
        print(
            f"[ERROR] adapter_path does not exist: {args.adapter_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    _warn_if_exists(args.output_dir)

    # ------------------------------------------------------------------ #
    # 1. Lazy imports (heavy; only after arg validation)                   #
    # ------------------------------------------------------------------ #
    print("\n[1/5] Importing torch, transformers, peft …")
    import torch  # noqa: F401  (needed for dtype resolution)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = _resolve_dtype(args.torch_dtype)
    print(f"      Using dtype: {torch_dtype}")

    # ------------------------------------------------------------------ #
    # 2. Load base model                                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[2/5] Loading base model from: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    print(f"      Base model loaded  (dtype={base_model.dtype})")

    # ------------------------------------------------------------------ #
    # 3. Load tokenizer                                                    #
    # ------------------------------------------------------------------ #
    print(f"\n[3/5] Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print("      Tokenizer loaded.")

    # ------------------------------------------------------------------ #
    # 4. Merge (or pass-through) adapter                                  #
    # ------------------------------------------------------------------ #
    if _is_lora_adapter(args.adapter_path):
        print(
            f"\n[4/5] LoRA adapter detected at: {args.adapter_path}\n"
            "      Loading PEFT model and merging weights …"
        )
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            torch_dtype=torch_dtype,
        )
        print("      Calling merge_and_unload() …")
        merged_model = peft_model.merge_and_unload()
        print("      Merge complete.")
    else:
        print(
            f"\n[4/5] No adapter_config.json found in: {args.adapter_path}\n"
            "      Treating adapter_path as a pre-merged model — "
            "loading directly …"
        )
        merged_model = AutoModelForCausalLM.from_pretrained(
            args.adapter_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        print("      Pre-merged model loaded.")

    # ------------------------------------------------------------------ #
    # 5. Save merged model + tokenizer                                     #
    # ------------------------------------------------------------------ #
    print(f"\n[5/5] Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("      Model and tokenizer saved.")

    # ------------------------------------------------------------------ #
    # 6. Verify output                                                     #
    # ------------------------------------------------------------------ #
    _verify_output(args.output_dir)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  base_model  : {args.base_model}")
    print(f"  adapter     : {args.adapter_path}")
    print(f"  output_dir  : {os.path.abspath(args.output_dir)}")
    print(f"  dtype       : {args.torch_dtype}")
    print("  Status      : SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
