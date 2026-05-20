#!/usr/bin/env python3
import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from huggingface_hub import hf_hub_download
"""v2rmp Agent SFT Training — QLoRA on Qwen2.5-1.5B-Instruct.

Embeds dataset inline for HF Jobs compatibility.
Pushes final adapter to Hub: aerialblancaservices/v2rmp-agent-1.5b
"""

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./v2rmp-agent-sft-1.5b"
HUB_MODEL_ID = "aerialblancaservices/v2rmp-agent-1.5b"
# ─── Load dataset from Hub ──────────────────────────────────────────────────

DATASET_REPO = "aerialblancaservices/v2rmp-sft-data"
DATASET_FILE = "dataset.jsonl"

print("Downloading v2rmp training dataset from Hub...")
dataset_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename=DATASET_FILE,
    repo_type="dataset",
)
print(f"  Downloaded to: {dataset_path}")

examples = []
with open(dataset_path, "r") as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

messages = [ex["messages"] for ex in examples]
dataset = Dataset.from_dict({"messages": messages})
print(f"  {len(dataset)} examples, columns: {dataset.column_names}")

split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ─── LoRA config ─────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ─── QLoRA 4-bit quantization ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ─── Training config ────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    max_length=4096,
    packing=True,
    bf16=True,
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    disable_tqdm=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=50,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    report_to="trackio",
    run_name="sft_v2rmp_qwen1.5b_qlora",
    seed=42,
)

eb = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
print(f"\n{'='*60}")
print("v2rmp Agent SFT Training — 1.5B")
print(f"{'='*60}")
print(f"  Model:       {MODEL_ID}")
print("  Method:      QLoRA (4-bit + LoRA r=16)")
print(f"  Dataset:     {len(train_dataset)} train / {len(eval_dataset)} eval")
print(f"  LR:          {training_args.learning_rate:.1e}")
print(f"  Epochs:      {training_args.num_train_epochs}")
print(f"  Eff. batch:  {eb}")
print(f"  Hub:         {HUB_MODEL_ID}")
print(f"{'='*60}\n")

# ─── Load model ──────────────────────────────────────────────────────────────
print("Loading base model with QLoRA...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
print(f"  Model loaded: {MODEL_ID}")

# ─── Train ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

print("Starting training...")
trainer.train()

print("\nTraining complete!")
print(f"Model pushed to: https://huggingface.co/{HUB_MODEL_ID}")
