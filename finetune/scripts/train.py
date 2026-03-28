#!/usr/bin/env python3
"""
QLoRA fine-tuning script using Hugging Face transformers + peft + trl.

Supports all training domains: code, math, science, finance, general.
Designed for single A100/H100 80GB GPU with 4-bit quantization.

Usage:
    python train.py --config configs/code.yaml
    python train.py --config configs/math.yaml --resume_from_checkpoint ./outputs/checkpoint-500
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict (override wins on conflicts)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> dict:
    """
    Load a training config YAML, merging with base.yaml if it exists alongside.

    The base config (configs/base.yaml) provides default values for all fields.
    Domain-specific configs override only the fields they specify. This means
    domain configs can be minimal — only listing what differs from base.yaml.
    """
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    base_path = os.path.join(config_dir, "base.yaml")

    with open(config_path, "r") as f:
        domain_cfg = yaml.safe_load(f) or {}

    if os.path.exists(base_path) and os.path.abspath(base_path) != config_path:
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(base_cfg, domain_cfg)
        print(f"Config loaded: {config_path} (merged with base.yaml)")
    else:
        cfg = domain_cfg
        print(f"Config loaded: {config_path}")

    return cfg


def setup_model_and_tokenizer(cfg: dict):
    """Load base model with 4-bit quantization and prepare for QLoRA."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    model_name = cfg["model"]["name"]
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Configuring 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True),
    )

    lora_cfg = cfg["lora"]
    print(f"Applying LoRA (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_dataset_from_config(cfg: dict, tokenizer):
    """Load and tokenize dataset from JSONL file(s)."""
    import json
    from datasets import Dataset

    data_files = cfg["data"]["train_files"]
    if isinstance(data_files, str):
        data_files = [data_files]

    all_records = []
    for path in data_files:
        if not os.path.exists(path):
            print(f"WARNING: data file not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    if not all_records:
        raise RuntimeError(
            f"No training data found. Expected files: {data_files}\n"
            "Run download_datasets.py first."
        )

    print(f"Loaded {len(all_records):,} training examples")

    max_seq_len = cfg["training"].get("max_seq_length", 4096)
    chat_template = cfg["data"].get("chat_template", "chatml")

    def format_conversations(record: dict) -> str:
        """Convert conversation record to tokenizer chat format."""
        convs = record.get("conversations", [])
        if not convs:
            return ""
        if chat_template == "chatml":
            parts = []
            for msg in convs:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(parts) + "\n"
        else:
            # ShareGPT format fallback
            text = ""
            for msg in convs:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"System: {content}\n\n"
                elif role == "user":
                    text += f"Human: {content}\n\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n\n"
            return text.strip()

    def tokenize(record: dict) -> dict:
        text = format_conversations(record)
        if not text:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        encoded = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = Dataset.from_list(all_records)
    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=cfg["data"].get("num_proc", 4),
        desc="Tokenizing",
    )
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

    # Optional validation split
    val_size = cfg["data"].get("val_size", 0.02)
    split = tokenized.train_test_split(test_size=val_size, seed=42)
    return split["train"], split["test"]


def build_trainer(cfg: dict, model, tokenizer, train_dataset, eval_dataset):
    """Build SFTTrainer with config-driven hyperparameters."""
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    training_cfg = cfg["training"]
    output_dir = cfg["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("num_epochs", 3),
        per_device_train_batch_size=training_cfg.get("batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("eval_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=training_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        bf16=training_cfg.get("bf16", True),
        fp16=training_cfg.get("fp16", False),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.3),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 200),
        eval_steps=training_cfg.get("eval_steps", 200),
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=training_cfg.get("report_to", ["tensorboard"]),
        run_name=cfg.get("run_name", "qlora-finetune"),
        dataloader_num_workers=training_cfg.get("dataloader_workers", 4),
        group_by_length=True,
        optim=training_cfg.get("optimizer", "paged_adamw_32bit"),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        ddp_find_unused_parameters=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=training_cfg.get("max_seq_length", 4096),
        dataset_num_proc=cfg["data"].get("num_proc", 4),
        packing=training_cfg.get("packing", True),
    )
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen2.5-72B")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training config (e.g. configs/math.yaml)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from",
    )
    args = parser.parse_args()

    # Check dependencies
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print("Run: pip install -r ../requirements_finetune.txt")
        sys.exit(1)

    cfg = load_config(args.config)
    print(f"\nTraining config: {args.config}")
    print(f"  Model:  {cfg['model']['name']}")
    print(f"  Domain: {cfg.get('domain', 'mixed')}")
    print(f"  Output: {cfg['output']['dir']}\n")

    model, tokenizer = setup_model_and_tokenizer(cfg)
    train_ds, eval_ds = load_dataset_from_config(cfg, tokenizer)

    print(f"Train examples: {len(train_ds):,}")
    print(f"Eval examples:  {len(eval_ds):,}\n")

    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    output_dir = cfg["output"]["dir"]
    adapter_dir = os.path.join(output_dir, "final_adapter")
    print(f"\nSaving LoRA adapter → {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("\nTraining complete.")
    print(f"LoRA adapter saved to: {adapter_dir}")
    print("Next step: run merge_and_export.py to merge into base model.")


if __name__ == "__main__":
    main()
