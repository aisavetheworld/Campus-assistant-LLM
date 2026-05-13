"""Train a LoRA/QLoRA SFT adapter with Transformers, TRL, PEFT, and datasets."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:  # pragma: no cover - depends on installed TRL version
    from trl import SFTTrainer

    SFTConfig = None  # type: ignore[assignment]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def parse_torch_dtype(dtype_name: str | None) -> torch.dtype | str | None:
    """Convert a config dtype string to a torch dtype."""
    if dtype_name is None or dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[dtype_name]


def require_file(path: str, label: str) -> None:
    """Raise a clear error if a required file does not exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}. Run scripts/build_sft_data.py first.")


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keyword arguments supported by a callable/class signature."""
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_training_args(config: dict[str, Any]) -> TrainingArguments:
    """Create TrainingArguments or TRL SFTConfig, depending on installed TRL."""
    training_cfg = dict(config.get("training", {}))
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    signature = inspect.signature(args_cls)

    if "eval_strategy" in training_cfg and "eval_strategy" not in signature.parameters:
        if "evaluation_strategy" in signature.parameters:
            training_cfg["evaluation_strategy"] = training_cfg.pop("eval_strategy")

    if "evaluation_strategy" in training_cfg and "evaluation_strategy" not in signature.parameters:
        if "eval_strategy" in signature.parameters:
            training_cfg["eval_strategy"] = training_cfg.pop("evaluation_strategy")

    dtype_name = model_cfg.get("torch_dtype")
    cuda_available = torch.cuda.is_available()
    if cuda_available and dtype_name in {"bfloat16", "bf16"} and "bf16" in signature.parameters:
        training_cfg.setdefault("bf16", True)
    if cuda_available and dtype_name in {"float16", "fp16"} and "fp16" in signature.parameters:
        training_cfg.setdefault("fp16", True)

    if args_cls is SFTConfig:
        sft_fields = {
            "dataset_text_field": data_cfg.get("text_field", "text"),
            "packing": data_cfg.get("packing", False),
            "max_seq_length": data_cfg.get("max_seq_length", 1024),
            "max_length": data_cfg.get("max_seq_length", 1024),
        }
        for key, value in sft_fields.items():
            if key in signature.parameters:
                training_cfg.setdefault(key, value)

    filtered = filter_kwargs(args_cls, training_cfg)
    ignored = sorted(set(training_cfg) - set(filtered))
    if ignored:
        print(f"Warning: ignored unsupported training config fields: {ignored}")
    return args_cls(**filtered)


def load_tokenizer(model_cfg: dict[str, Any]) -> AutoTokenizer:
    """Load tokenizer and ensure a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_cfg: dict[str, Any]) -> AutoModelForCausalLM:
    """Load a causal LM, optionally with 4-bit quantization."""
    use_4bit = bool(model_cfg.get("use_4bit", False))
    torch_dtype = parse_torch_dtype(model_cfg.get("torch_dtype", "auto"))
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("use_4bit: true requires a CUDA GPU with bitsandbytes support.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("bitsandbytes 4-bit loading requires a recent transformers install.") from exc

        compute_dtype = torch.bfloat16 if torch_dtype in {"auto", torch.bfloat16} else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        **model_kwargs,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    """Create a PEFT LoRA configuration from YAML."""
    required = ["r", "lora_alpha", "lora_dropout", "bias", "task_type", "target_modules"]
    missing = [field for field in required if field not in lora_cfg]
    if missing:
        raise ValueError(f"Missing LoRA config fields: {missing}")

    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=list(lora_cfg["target_modules"]),
    )


def build_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
    train_dataset: Any,
    eval_dataset: Any,
    data_cfg: dict[str, Any],
) -> SFTTrainer:
    """Create SFTTrainer while handling TRL API differences."""
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora_config,
    }
    signature = inspect.signature(SFTTrainer.__init__)

    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    old_style_fields = {
        "dataset_text_field": data_cfg.get("text_field", "text"),
        "max_seq_length": data_cfg.get("max_seq_length", 1024),
        "packing": data_cfg.get("packing", False),
    }
    for key, value in old_style_fields.items():
        if key in signature.parameters:
            trainer_kwargs[key] = value

    return SFTTrainer(**trainer_kwargs)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a LoRA SFT adapter.")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    return parser.parse_args()


def main() -> None:
    """Run SFT training."""
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    if "model_name_or_path" not in model_cfg:
        raise ValueError("Config missing model.model_name_or_path")

    require_file(data_cfg.get("train_file", ""), "Train file")
    require_file(data_cfg.get("eval_file", ""), "Eval file")

    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg)
    lora_config = build_lora_config(lora_cfg)
    training_args = build_training_args(config)

    train_dataset = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    eval_dataset = load_dataset("json", data_files=data_cfg["eval_file"], split="train")

    text_field = data_cfg.get("text_field", "text")
    for label, dataset in [("train", train_dataset), ("eval", eval_dataset)]:
        if text_field not in dataset.column_names:
            raise ValueError(f"{label} dataset does not contain text field '{text_field}'")

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        lora_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_cfg=data_cfg,
    )

    output_dir = Path(training_cfg.get("output_dir", "outputs/sft_lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting SFT training with {len(train_dataset)} train and {len(eval_dataset)} eval samples.")
    train_result = trainer.train()
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    save_json(output_dir / "metrics.json", {"train": train_metrics, "eval": eval_metrics})

    print(f"Saved LoRA adapter, tokenizer files, resolved config, and metrics to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
