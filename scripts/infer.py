"""Run inference with a base causal LM and optional LoRA adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(instruction: str, input_text: str) -> str:
    """Create the inference prompt matching the SFT training template."""
    return f"### Instruction:\n{instruction.strip()}\n\n### Input:\n{input_text.strip()}\n\n### Response:\n"


def parse_torch_dtype(dtype_name: str) -> torch.dtype | str:
    """Convert dtype name to a torch dtype or auto."""
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    """Load tokenizer, base model, and optional LoRA adapter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": parse_torch_dtype(args.torch_dtype),
    }

    if args.use_4bit:
        if device.type != "cuda":
            raise RuntimeError("--use_4bit requires a CUDA GPU with bitsandbytes support.")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"
    elif device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    if device.type != "cuda":
        model.to(device)

    model.eval()
    return tokenizer, model, device


def first_parameter_device(model: Any, fallback: torch.device) -> torch.device:
    """Find the device used by model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def generate_response(
    tokenizer: Any,
    model: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate and return only the response portion."""
    input_device = first_parameter_device(model, device)
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    if "### Response:" in decoded:
        return decoded.split("### Response:", maxsplit=1)[-1].strip()
    return decoded.strip()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run base or LoRA-adapter inference.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one inference request."""
    args = parse_args()
    tokenizer, model, device = load_model_and_tokenizer(args)
    prompt = build_prompt(args.instruction, args.input)
    response = generate_response(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(response)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
