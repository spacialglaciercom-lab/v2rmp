#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "numpy",
#     "gguf",
# ]
# ///

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
import subprocess

def run_command(cmd, description):
    """Run a command with error handling."""
    print(f"   {description}...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"   {result.stdout[:200]}")  # Show first 200 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Command failed: {' '.join(cmd)}")
        if e.stdout:
            print(f"   STDOUT: {e.stdout[:500]}")
        if e.stderr:
            print(f"   STDERR: {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        print(f"   ❌ Command not found: {cmd[0]}")
        return False

# CRITICAL FIX: Ensure cmake is installed on the Hugging Face Jobs instance
print("🔍 Ensuring CMake is installed...")
subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake"], check=False)

# Configuration from environment variables
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL", "evalstate/qwen-capybara-medium")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "evalstate/qwen-capybara-medium-gguf")
username = os.environ.get("HF_USERNAME", ADAPTER_MODEL.split('/')[0])

print(f"\n📦 Configuration:")
print(f"   Base model: {BASE_MODEL}")
print(f"   Adapter model: {ADAPTER_MODEL}")
print(f"   Output repo: {OUTPUT_REPO}")

# Step 1: Load base model and adapter
print("\n🔧 Step 1: Loading base model and LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
merged_model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL, trust_remote_code=True)

# Step 2: Save merged model temporarily
print("\n💾 Step 2: Saving merged model...")
merged_dir = "/tmp/merged_model"
merged_model.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)

# Step 3: Install llama.cpp for conversion
print("\n📥 Step 3: Setting up llama.cpp for GGUF conversion...")
run_command(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"], "Cloning llama.cpp repository")
run_command(["pip", "install", "-r", "/tmp/llama.cpp/requirements.txt"], "Installing llama.cpp requirements")

# Step 4: Convert to GGUF (FP16)
print("\n🔄 Step 4: Converting to GGUF format (FP16)...")
gguf_output_dir = "/tmp/gguf_output"
os.makedirs(gguf_output_dir, exist_ok=True)
convert_script = "/tmp/llama.cpp/convert_hf_to_gguf.py"
model_name = ADAPTER_MODEL.split('/')[-1]
gguf_file = f"{gguf_output_dir}/{model_name}-f16.gguf"
run_command([sys.executable, convert_script, merged_dir, "--outfile", gguf_file, "--outtype", "f16"], f"Converting to FP16")

# Step 5: Quantize to different formats
print("\n⚙️  Step 5: Creating quantized versions...")
os.makedirs("/tmp/llama.cpp/build", exist_ok=True)
run_command(["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp", "-DGGML_CUDA=OFF"], "Configuring with CMake")
run_command(["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"], "Building llama-quantize")

quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"
quant_formats = [("Q4_K_M", "4-bit, medium quality"), ("Q8_0", "8-bit, very high quality")]
quantized_files = []
for quant_type, description in quant_formats:
    quant_file = f"{gguf_output_dir}/{model_name}-{quant_type.lower()}.gguf"
    if run_command([quantize_bin, gguf_file, quant_file, quant_type], f"Quantizing to {quant_type}"):
        quantized_files.append((quant_file, quant_type))

# Step 6: Upload to Hub
print("\n☁️  Step 6: Uploading to Hugging Face Hub...")
api = HfApi()
api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=gguf_file, path_in_repo=f"{model_name}-f16.gguf", repo_id=OUTPUT_REPO)
for quant_file, quant_type in quantized_files:
    api.upload_file(path_or_fileobj=quant_file, path_in_repo=f"{model_name}-{quant_type.lower()}.gguf", repo_id=OUTPUT_REPO)

print("\n✅ GGUF Conversion Complete!")
print(f"📦 Repository: https://huggingface.co/{OUTPUT_REPO}")
