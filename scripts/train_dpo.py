"""Train a one-epoch DPO smoke-test adapter starting from the final SFT adapter."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


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
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: str, label: str) -> None:
    """Raise a clear error if a required directory does not exist."""
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only keyword arguments supported by a callable/class signature."""
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def import_trl_dpo() -> tuple[Any, Any | None]:
    """Import TRL DPO classes with a clear error if TRL is missing."""
    try:
        from trl import DPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "TRL is required for DPO training. Install project requirements first: "
            "pip install -r requirements.txt"
        ) from exc

    try:
        from trl import DPOConfig
    except ImportError:  # pragma: no cover - older TRL fallback
        DPOConfig = None
    return DPOTrainer, DPOConfig


def load_tokenizer(model_cfg: dict[str, Any]) -> AutoTokenizer:
    """Load tokenizer and ensure a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build model loading kwargs, including optional 4-bit quantization."""
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
            raise RuntimeError("4-bit loading requires a recent transformers install.") from exc

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
    return model_kwargs


def load_sft_policy_model(
    model_cfg: dict[str, Any],
    *,
    is_trainable: bool,
    gradient_checkpointing: bool = False,
) -> PeftModel:
    """Load base model and attach the final SFT adapter as policy or reference."""
    adapter_path = model_cfg.get("sft_adapter_path")
    require_dir(adapter_path, "Final SFT adapter")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        **build_model_kwargs(model_cfg),
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=is_trainable)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if is_trainable and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if is_trainable and gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if not is_trainable:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def build_training_args(config: dict[str, Any], dpo_config_cls: Any | None) -> TrainingArguments:
    """Create DPOConfig when available, otherwise TrainingArguments."""
    training_cfg = dict(config.get("training", {}))
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    args_cls = dpo_config_cls if dpo_config_cls is not None else TrainingArguments
    signature = inspect.signature(args_cls)

    if "eval_strategy" in training_cfg and "eval_strategy" not in signature.parameters:
        if "evaluation_strategy" in signature.parameters:
            training_cfg["evaluation_strategy"] = training_cfg.pop("eval_strategy")
    if "evaluation_strategy" in training_cfg and "evaluation_strategy" not in signature.parameters:
        if "eval_strategy" in signature.parameters:
            training_cfg["eval_strategy"] = training_cfg.pop("evaluation_strategy")

    dtype_name = model_cfg.get("torch_dtype")
    if torch.cuda.is_available() and dtype_name in {"bfloat16", "bf16"} and "bf16" in signature.parameters:
        training_cfg.setdefault("bf16", True)
    if torch.cuda.is_available() and dtype_name in {"float16", "fp16"} and "fp16" in signature.parameters:
        training_cfg.setdefault("fp16", True)

    dpo_fields = {
        "beta": training_cfg.get("beta", 0.1),
        "max_prompt_length": data_cfg.get("max_prompt_length", 768),
        "max_length": data_cfg.get("max_length", 1280),
        "remove_unused_columns": False,
    }
    for key, value in dpo_fields.items():
        if key in signature.parameters:
            training_cfg.setdefault(key, value)

    filtered = filter_kwargs(args_cls, training_cfg)
    ignored = sorted(set(training_cfg) - set(filtered))
    if ignored:
        print(f"Warning: ignored unsupported training config fields: {ignored}")
    return args_cls(**filtered)


def add_eos(text: str, eos_token: str | None) -> str:
    """Append EOS to a response if configured and available."""
    stripped = text.strip()
    if not eos_token or stripped.endswith(eos_token):
        return stripped
    return f"{stripped}{eos_token}"


def prepare_dpo_dataset(dataset: Dataset, data_cfg: dict[str, Any], tokenizer: AutoTokenizer) -> Dataset:
    """Convert project DPO JSONL rows into TRL DPOTrainer columns."""
    prompt_field = data_cfg.get("prompt_field", "formatted_prompt")
    chosen_field = data_cfg.get("chosen_field", "chosen")
    rejected_field = data_cfg.get("rejected_field", "rejected")
    add_response_eos = bool(data_cfg.get("add_eos_to_responses", True))

    missing = [
        field
        for field in [prompt_field, chosen_field, rejected_field]
        if field not in dataset.column_names
    ]
    if missing:
        raise ValueError(f"DPO dataset missing required fields: {missing}")

    eos_token = tokenizer.eos_token if add_response_eos else None

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": row[prompt_field],
            "chosen": add_eos(row[chosen_field], eos_token),
            "rejected": add_eos(row[rejected_field], eos_token),
            "id": row.get("id", ""),
            "category": row.get("category", ""),
            "risk_level": row.get("risk_level", ""),
        }

    keep_columns = ["prompt", "chosen", "rejected", "id", "category", "risk_level"]
    converted = dataset.map(convert, remove_columns=dataset.column_names)
    return converted.select_columns([column for column in keep_columns if column in converted.column_names])


def build_dpo_trainer(
    *,
    trainer_cls: Any,
    policy_model: PeftModel,
    ref_model: PeftModel | None,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: dict[str, Any],
) -> Any:
    """Create DPOTrainer while handling TRL API differences."""
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    signature = inspect.signature(trainer_cls.__init__)

    trainer_kwargs: dict[str, Any] = {
        "model": policy_model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if "ref_model" in signature.parameters:
        trainer_kwargs["ref_model"] = ref_model
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    optional_fields = {
        "beta": training_cfg.get("beta", 0.1),
        "max_prompt_length": data_cfg.get("max_prompt_length", 768),
        "max_length": data_cfg.get("max_length", 1280),
        "force_use_ref_model": ref_model is not None,
    }
    for key, value in optional_fields.items():
        if key in signature.parameters:
            trainer_kwargs[key] = value

    return trainer_cls(**trainer_kwargs)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train DPO smoke-test adapter from final SFT.")
    parser.add_argument("--config", default="configs/dpo_lora.yaml")
    return parser.parse_args()


def main() -> None:
    """Run DPO smoke-test training."""
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    trainer_cls, dpo_config_cls = import_trl_dpo()

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    if "model_name_or_path" not in model_cfg:
        raise ValueError("Config missing model.model_name_or_path")

    require_file(data_cfg.get("train_file", ""), "DPO train file")
    require_file(data_cfg.get("eval_file", ""), "DPO eval file")

    tokenizer = load_tokenizer(model_cfg)
    gradient_checkpointing = bool(training_cfg.get("gradient_checkpointing", False))
    print("Loading trainable policy model from final SFT adapter.")
    policy_model = load_sft_policy_model(
        model_cfg,
        is_trainable=True,
        gradient_checkpointing=gradient_checkpointing,
    )

    reference_model_mode = model_cfg.get("reference_model_mode", "separate")
    if reference_model_mode == "separate":
        print("Loading frozen reference model from the same final SFT adapter.")
        try:
            ref_model = load_sft_policy_model(model_cfg, is_trainable=False)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA OOM while loading the separate frozen reference model. "
                    "For the smoke test, reduce per_device_train_batch_size to 1 or set "
                    "model.reference_model_mode: 'trainer_managed' in configs/dpo_lora.yaml "
                    "if your installed TRL version supports PEFT reference handling."
                ) from exc
            raise
    elif reference_model_mode == "trainer_managed":
        print(
            "Using TRL-managed reference handling. This is a memory fallback; "
            "use 'separate' for the clean frozen-SFT reference setup when memory allows."
        )
        ref_model = None
    else:
        raise ValueError("model.reference_model_mode must be 'separate' or 'trainer_managed'")

    training_args = build_training_args(config, dpo_config_cls)

    raw_train = load_dataset("json", data_files=data_cfg["train_file"], split="train")
    raw_eval = load_dataset("json", data_files=data_cfg["eval_file"], split="train")
    train_dataset = prepare_dpo_dataset(raw_train, data_cfg, tokenizer)
    eval_dataset = prepare_dpo_dataset(raw_eval, data_cfg, tokenizer)

    output_dir = Path(training_cfg.get("output_dir", "outputs/dpo_r32_attn_mlp_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Starting DPO smoke-test training with "
        f"{len(train_dataset)} train and {len(eval_dataset)} eval pairs."
    )
    trainer = build_dpo_trainer(
        trainer_cls=trainer_cls,
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )

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

    print(f"Saved DPO adapter, tokenizer files, resolved config, and metrics to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command-line error path
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
