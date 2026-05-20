#!/usr/bin/env python3
"""
v2rmp Agent SFT Training — Self-contained for HF Jobs
======================================================
Fine-tunes Qwen2.5-7B-Instruct on the v2rmp route optimization agent dataset
using QLoRA (4-bit + LoRA adapters).

The dataset is embedded directly in this script so it works in HF Jobs
which has no access to local files.
"""

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from v2rmp_dataset_builder import V2RMPDatasetBuilder

# ─── Build dataset ────────────────────────────────────────────────────────────

print("📦 Building v2rmp training dataset...")
builder = V2RMPDatasetBuilder(seed=42)
examples = builder.build()

# Convert to HF Dataset format
data = {"messages": [ex.messages for ex in examples]}
dataset = Dataset.from_dict(data)
print(f"   {len(dataset)} examples loaded")

# Train/validation split
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"   Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ─── LoRA config ──────────────────────────────────────────────────────────────

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
print(f"🔧 LoRA: r=16, alpha=32, targets={peft_config.target_modules}")

# ─── QLoRA 4-bit quantization ────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print("⚡ QLoRA: 4-bit NF4 with double quantization")

# ─── Training config ─────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./v2rmp-agent-sft-1.5b"
HUB_MODEL_ID = "aerialblancaservices/v2rmp-agent-1.5b"

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Core hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,        # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Sequence handling
    max_length=4096,
    packing=True,

    # Precision & memory
    bf16=True,
    gradient_checkpointing=True,

    # Logging
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    disable_tqdm=True,

    # Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,

    # Evaluation
    eval_strategy="steps",
    eval_steps=50,

    # Hub
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,

    # Seed
    seed=42,
)

# ─── Summary ──────────────────────────────────────────────────────────────────

effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
print(f"\n{'='*60}")
print("v2rmp Agent SFT Training")
print(f"{'='*60}")
print(f"  Model:              {MODEL_ID}")
print("  Method:             QLoRA (4-bit NF4 + LoRA r=16)")
print(f"  Dataset:            {len(train_dataset)} train / {len(eval_dataset)} eval")
print(f"  Learning rate:      {training_args.learning_rate:.1e}")
print(f"  Epochs:             {training_args.num_train_epochs}")
print(f"  Effective batch:    {effective_batch}")
print(f"  Max seq length:     {training_args.max_length}")
print(f"  Packing:            {training_args.packing}")
print(f"  Output:             {OUTPUT_DIR}")
print(f"  Hub:                {HUB_MODEL_ID}")
print(f"{'='*60}\n")

# ─── Train ────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=MODEL_ID,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    quantization_config=bnb_config,
)

print("🚀 Starting training...")
trainer.train()

# ── Save & push ───────────────────────────────────────────────────────────
print(f"\n💾 Saving model to {OUTPUT_DIR}...")
trainer.save_model()

print(f"📤 Pushing to Hub: {HUB_MODEL_ID}")
trainer.push_to_hub()

print("\n✅ Training complete!")
print(f"   Model: https://huggingface.co/{HUB_MODEL_ID}")
