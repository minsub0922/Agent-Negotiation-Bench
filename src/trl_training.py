from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json_dataset(path: Path):
    from datasets import load_dataset

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return load_dataset("json", data_files=str(path), split="train")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _make_training_args(
    config_cls: type,
    output_dir: Path,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    warmup_ratio: float,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    has_eval: bool,
    seed: int,
    bf16: bool,
    fp16: bool,
    extra_kwargs: dict[str, Any] | None = None,
):
    sig = inspect.signature(config_cls.__init__)
    kwargs: dict[str, Any] = {}

    def put(value: Any, *names: str) -> None:
        for name in names:
            if name in sig.parameters:
                kwargs[name] = value
                return

    put(str(output_dir), "output_dir")
    put(per_device_train_batch_size, "per_device_train_batch_size")
    put(per_device_eval_batch_size, "per_device_eval_batch_size")
    put(gradient_accumulation_steps, "gradient_accumulation_steps")
    put(learning_rate, "learning_rate")
    put(num_train_epochs, "num_train_epochs")
    put(warmup_ratio, "warmup_ratio")
    put(logging_steps, "logging_steps")
    put(save_steps, "save_steps")
    put(seed, "seed")
    put(bf16, "bf16")
    put(fp16, "fp16")
    put([], "report_to")

    eval_strategy_value = "steps" if has_eval else "no"
    put(eval_strategy_value, "evaluation_strategy", "eval_strategy")
    put("steps", "save_strategy")
    put("steps", "logging_strategy")

    if has_eval:
        put(eval_steps, "eval_steps")

    put(False, "remove_unused_columns")

    for key, value in (extra_kwargs or {}).items():
        if key in sig.parameters:
            kwargs[key] = value

    return config_cls(**kwargs)


def _resolve_torch_dtype(bf16: bool, fp16: bool):
    import torch

    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def _apply_lora_if_requested(
    model,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str] | None,
):
    if not use_lora:
        return model

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, config)


def _trainer_tokenizer_kwargs(trainer_cls: type, tokenizer) -> dict[str, Any]:
    sig = inspect.signature(trainer_cls.__init__)
    if "tokenizer" in sig.parameters:
        return {"tokenizer": tokenizer}
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    return {}


def train_sft(
    model_path: str,
    train_file: Path,
    eval_file: Path | None,
    output_dir: Path,
    max_seq_length: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    warmup_ratio: float,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    seed: int,
    bf16: bool,
    fp16: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str] | None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    try:
        from trl import SFTConfig  # type: ignore
    except Exception:
        SFTConfig = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_json_dataset(Path(train_file))
    eval_dataset = _load_json_dataset(Path(eval_file)) if eval_file else None

    torch_dtype = _resolve_torch_dtype(bf16=bf16, fp16=fp16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = _apply_lora_if_requested(
        model=model,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    args_cls = SFTConfig or TrainingArguments
    args = _make_training_args(
        config_cls=args_cls,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        has_eval=eval_dataset is not None,
        seed=seed,
        bf16=bf16,
        fp16=fp16,
        extra_kwargs={
            "max_seq_length": max_seq_length,
        },
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer_kwargs.update(_trainer_tokenizer_kwargs(SFTTrainer, tokenizer))

    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "dataset_text_field" in trainer_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in trainer_sig.parameters:
        trainer_kwargs["formatting_func"] = lambda sample: sample["text"]

    if "max_seq_length" in trainer_sig.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()

    eval_metrics = trainer.evaluate() if eval_dataset is not None else None

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    summary = {
        "stage": "sft",
        "generated_at": datetime.now().isoformat(),
        "model_path": model_path,
        "train_file": str(train_file),
        "eval_file": str(eval_file) if eval_file else None,
        "output_dir": str(output_dir),
        "final_model_dir": str(final_dir),
        "num_train_rows": len(train_dataset),
        "num_eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    _json_dump(output_dir / "training_summary.json", summary)
    return summary


def train_dpo(
    model_path: str,
    ref_model_path: str | None,
    train_file: Path,
    eval_file: Path | None,
    output_dir: Path,
    max_length: int,
    max_prompt_length: int,
    beta: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    warmup_ratio: float,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    seed: int,
    bf16: bool,
    fp16: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str] | None,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer

    try:
        from trl import DPOConfig  # type: ignore
    except Exception:
        DPOConfig = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_json_dataset(Path(train_file))
    eval_dataset = _load_json_dataset(Path(eval_file)) if eval_file else None

    torch_dtype = _resolve_torch_dtype(bf16=bf16, fp16=fp16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model = _apply_lora_if_requested(
        model=model,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    ref_base_path = ref_model_path or model_path
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_base_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    args_cls = DPOConfig or TrainingArguments
    args = _make_training_args(
        config_cls=args_cls,
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        has_eval=eval_dataset is not None,
        seed=seed,
        bf16=bf16,
        fp16=fp16,
        extra_kwargs={
            "beta": beta,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "remove_unused_columns": False,
        },
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer_kwargs.update(_trainer_tokenizer_kwargs(DPOTrainer, tokenizer))

    trainer_sig = inspect.signature(DPOTrainer.__init__)
    if "beta" in trainer_sig.parameters:
        trainer_kwargs["beta"] = beta
    if "max_length" in trainer_sig.parameters:
        trainer_kwargs["max_length"] = max_length
    if "max_prompt_length" in trainer_sig.parameters:
        trainer_kwargs["max_prompt_length"] = max_prompt_length

    trainer = DPOTrainer(**trainer_kwargs)
    train_result = trainer.train()

    eval_metrics = trainer.evaluate() if eval_dataset is not None else None

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    summary = {
        "stage": "dpo",
        "generated_at": datetime.now().isoformat(),
        "model_path": model_path,
        "ref_model_path": ref_model_path,
        "train_file": str(train_file),
        "eval_file": str(eval_file) if eval_file else None,
        "output_dir": str(output_dir),
        "final_model_dir": str(final_dir),
        "num_train_rows": len(train_dataset),
        "num_eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    _json_dump(output_dir / "training_summary.json", summary)
    return summary
