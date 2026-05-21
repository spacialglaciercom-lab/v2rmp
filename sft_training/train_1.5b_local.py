# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "peft",
#     "trl",
#     "transformers",
#     "huggingface_hub",
#     "bitsandbytes",
#     "accelerate"
# ]
# ///

import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./v2rmp-agent-sft-1.5b-local"

dataset_path = "dataset_final.jsonl"
print(f"Loading local dataset from: {dataset_path}")

examples = []
with open(dataset_path, "r", encoding="utf-8") as f:
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

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

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
    push_to_hub=False,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=MODEL_ID,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Training complete. Model saved to {OUTPUT_DIR}")