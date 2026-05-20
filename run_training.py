#!/usr/bin/env python3
"""HF Job runner for drone CVRP neural solver training."""

import os
import torch

# Import the training module
from train_job import Config, train

# ── Config from env or defaults ──
cfg = Config(
    problem_size=int(os.environ.get("PROBLEM_SIZE", "50")),
    epochs=int(os.environ.get("EPOCHS", "500")),
    batch_size=int(os.environ.get("BATCH_SIZE", "512")),
    lr=float(os.environ.get("LR", "1e-4")),
    embedding_dim=int(os.environ.get("EMBEDDING_DIM", "128")),
    num_encoder_layers=int(os.environ.get("NUM_LAYERS", "3")),
    capacity=float(os.environ.get("CAPACITY", "40.0")),
    max_demand=int(os.environ.get("MAX_DEMAND", "9")),
    hub_model_id=os.environ.get("HUB_MODEL_ID", "aerialblancaservices/drone-cvrp-neural-solver"),
    push_to_hub=os.environ.get("PUSH_TO_HUB", "1") == "1",
    trackio_project=os.environ.get("TRACKIO_PROJECT", "drone-cvrp-training"),
)

print("=== Drone CVRP Neural Solver Training ===")
print(f"Config: {cfg}")
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

train(cfg)
