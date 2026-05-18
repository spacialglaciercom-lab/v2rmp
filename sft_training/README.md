# v2rmp SFT Training

Fine-tune an LLM to act as a v2rmp route optimization agent.

## Files

| File | Description |
|------|-------------|
| `v2rmp_dataset_builder.py` | Generates the training dataset from the v2rmp codebase |
| `dataset.jsonl` | Generated ChatML dataset (58 examples) |
| `train_v2rmp_agent.py` | SFT training script using TRL + LoRA/QLoRA |

## Dataset Coverage

The dataset covers 4 key areas:

1. **System prompt** — Defines the v2rmp ecosystem role, capabilities, and behavior
2. **MCP tool calling** — Proper JSON-RPC 2.0 format for all 20+ MCP tools
3. **CLI commands** — All `rmpca` subcommands (extract, compile, clean, optimize, vrp, pipeline, elevation, etc.)
4. **Agent task execution** — Multi-step task plan JSON format for `rmpca agent --plan`

Plus: multi-turn conversations, troubleshooting, conceptual questions, and short-form variations.

## Quick Start

```bash
# 1. Generate the dataset
python v2rmp_dataset_builder.py --output dataset.jsonl --stats

# 2. Train with QLoRA (single GPU)
python train_v2rmp_agent.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset dataset.jsonl \
  --output_dir ./v2rmp-agent-sft \
  --qlora \
  --push_to_hub \
  --hub_model_id your-username/v2rmp-agent-7b

# 3. Train on HF Jobs (A100 recommended for 7B)
# Submit via HF Jobs with the training script
```

## Hardware

| Model Size | Method | GPU | Time (est.) |
|-----------|--------|-----|-------------|
| 7B | QLoRA | 1x A10G (24GB) | ~30 min |
| 7B | LoRA | 1x A100 (40GB) | ~20 min |
| 7B | Full | 4x A100 (80GB) | ~1 hour |

## Using the Fine-tuned Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "your-username/v2rmp-agent-7b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

messages = [
    {"role": "system", "content": "You are an expert route optimization agent..."},
    {"role": "user", "content": "How do I optimize a route on my Montreal map?"},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(output[0]))
```
