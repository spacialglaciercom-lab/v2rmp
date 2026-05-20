#!/usr/bin/env python3
"""
v2rmp Agent SFT Training Script
================================
Fine-tunes a base LLM on the v2rmp route optimization agent dataset.

Supports:
  - QLoRA (4-bit quantization + LoRA adapters) for single-GPU training
  - Full LoRA (bf16 + LoRA) for multi-GPU
  - Trackio experiment tracking
  - Push to Hugging Face Hub

Usage:
  # Generate dataset first
  python v2rmp_dataset_builder.py --output dataset.jsonl --stats

  # Train with QLoRA on a single GPU (recommended)
  python train_v2rmp_agent.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset dataset.jsonl \
    --output_dir ./v2rmp-agent-sft \
    --qlora \
    --push_to_hub \
    --hub_model_id your-username/v2rmp-agent-7b

  # Train with LoRA (bf16, no quantization)
  python train_v2rmp_agent.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset dataset.jsonl \
    --output_dir ./v2rmp-agent-sft \
    --push_to_hub \
    --hub_model_id your-username/v2rmp-agent-7b

Hardware recommendations:
  - QLoRA: 1x A100 (40GB) or 1x A10G (24GB) for 7B
  - LoRA:  1x A100 (80GB) for 7B
  - Full:  2-4x A100 (80GB) for 7B
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="v2rmp Agent SFT Training")

    # Model
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--dataset", default="dataset.jsonl",
        help="Path to training dataset JSONL (messages format)"
    )
    parser.add_argument(
        "--dataset_split", default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.05,
        help="Fraction of data to use for validation (default: 0.05)"
    )

    # Output
    parser.add_argument(
        "--output_dir", default="./v2rmp-agent-sft",
        help="Output directory for checkpoints and final model"
    )
    parser.add_argument(
        "--run_name", default=None,
        help="Trackio/W&B run name (default: auto-generated)"
    )

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (default: 2e-4 for LoRA)")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length (default: 4096)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--lr_scheduler", default="cosine", help="LR scheduler (default: cosine)")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")

    # Quantization
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA quantization")
    parser.add_argument("--no_qlora", action="store_true", help="Disable QLoRA (use bf16 LoRA)")

    # Memory
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no_packing", action="store_true", help="Disable sequence packing")

    # Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HF Hub after training")
    parser.add_argument("--hub_model_id", default=None, help="HF Hub model ID for push")
    parser.add_argument("--hub_token", default=None, help="HF Hub token (default: auto from env)")

    # Tracking
    parser.add_argument("--report_to", default="none", help="Tracking backend: none|wandb|trackio (default: none)")
    parser.add_argument("--trackio_space_id", default=None, help="Trackio Space ID for dashboard")
    parser.add_argument("--trackio_project", default=None, help="Trackio project name")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps (default: 100)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps (default: 10)")
    parser.add_argument("--disable_tqdm", action="store_true", default=True, help="Disable tqdm (default: True)")

    return parser.parse_args()


def get_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    """Configure LoRA adapters targeting all linear projections."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def get_qlora_config() -> BitsAndBytesConfig:
    """Configure 4-bit QLoRA quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_run_name(args) -> str:
    """Build a descriptive run name for tracking."""
    model_name = args.model.split("/")[-1].lower()
    method = "qlora" if args.qlora else "lora"
    lr = f"lr{args.learning_rate:.0e}".replace("+", "").replace("-0", "-")
    bs = args.batch_size * args.gradient_accumulation
    return f"sft_{model_name}_{method}_{lr}_bs{bs}"


def main():
    args = parse_args()

    # ── Validate dataset exists ────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Generate it first: python v2rmp_dataset_builder.py --output {args.dataset} --stats")
        sys.exit(1)

    # ── Load dataset ───────────────────────────────────────────────────────
    print(f"📂 Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"   Loaded {len(dataset)} examples")

    # Split into train/validation
    if args.validation_split > 0 and len(dataset) > 20:
        split = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"   Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"   Train: {len(train_dataset)} | Eval: skipped (too few examples)")

    # ── Configure LoRA ─────────────────────────────────────────────────────
    peft_config = get_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    print(f"🔧 LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"   Target modules: {peft_config.target_modules}")

    # ── Configure quantization ─────────────────────────────────────────────
    quantization_config = None
    if args.qlora:
        quantization_config = get_qlora_config()
        print("⚡ QLoRA enabled: 4-bit NF4 quantization with double quantization")
    else:
        print("📊 Training in bf16 (no quantization)")

    # ── Build run name ─────────────────────────────────────────────────────
    run_name = args.run_name or build_run_name(args)

    # ── Configure tracking ─────────────────────────────────────────────────
    report_to = args.report_to
    trackio_kwargs = {}
    if report_to == "trackio":
        if args.trackio_space_id:
            trackio_kwargs["trackio_space_id"] = args.trackio_space_id
        if args.trackio_project:
            trackio_kwargs["trackio_project"] = args.trackio_project

    # ── Training config ────────────────────────────────────────────────────
    effective_batch = args.batch_size * args.gradient_accumulation
    print(f"\n{'='*60}")
    print("v2rmp Agent SFT Training")
    print(f"{'='*60}")
    print(f"  Model:              {args.model}")
    print(f"  Dataset:            {len(train_dataset)} train examples")
    print(f"  Method:             {'QLoRA (4-bit)' if args.qlora else 'LoRA (bf16)'}")
    print(f"  LoRA rank:          {args.lora_r}")
    print(f"  Learning rate:      {args.learning_rate:.1e}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Effective batch:    {effective_batch}")
    print(f"  Max sequence length:{args.max_length}")
    print(f"  Packing:            {not args.no_packing}")
    print(f"  Run name:           {run_name}")
    print(f"  Report to:          {report_to}")
    print(f"  Output:             {args.output_dir}")
    if args.push_to_hub and args.hub_model_id:
        print(f"  Hub:                {args.hub_model_id}")
    print(f"{'='*60}\n")

    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Core hyperparameters
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,

        # Sequence handling
        max_length=args.max_length,
        packing=not args.no_packing,

        # Precision & memory — auto-detect bf16 support
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=not args.no_gradient_checkpointing,

        # Logging
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        disable_tqdm=args.disable_tqdm,

        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,

        # Tracking
        report_to=report_to,
        run_name=run_name,

        # Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,

        # Seed
        seed=args.seed,

        # Trackio
        **trackio_kwargs,
    )

    # ── Create trainer ─────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("🚀 Starting training...")
    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────
    print(f"\n💾 Saving model to {args.output_dir}...")
    trainer.save_model()

    if args.push_to_hub and args.hub_model_id:
        print(f"📤 Pushing to Hugging Face Hub: {args.hub_model_id}")
        trainer.push_to_hub()
        print(f"✅ Model available at: https://huggingface.co/{args.hub_model_id}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"  Output: {args.output_dir}")
    print(f"  LoRA adapters saved to: {args.output_dir}")
    print("\n  To use the fine-tuned model:")
    print("  ```python")
    print("  from peft import PeftModel")
    print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("  ")
    print(f"  base = AutoModelForCausalLM.from_pretrained('{args.model}')")
    print(f"  model = PeftModel.from_pretrained(base, '{args.output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.model}')")
    print("  ```")
    if args.push_to_hub and args.hub_model_id:
        print("\n  Or load directly:")
        print(f"  model = PeftModel.from_pretrained(base, '{args.hub_model_id}')")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
