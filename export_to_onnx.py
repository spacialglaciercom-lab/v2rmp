#!/usr/bin/env python3
"""
Export a v2rmp CVRP checkpoint (.pt from train_job.py) to ONNX.

The ONNX model matches the v2rmp MCP server interface:
  inputs:  locs [1, N, 2], demand [1, N, 1], capacity [1, 1]
  outputs: actions [1, steps], log_p [1, steps]

Usage:
  python export_to_onnx.py --repo aerialblancaservices/v2rmp-routing-ml --ckpt cvrp50_best.pt --out model.onnx
  python export_to_onnx.py --local ./checkpoints/cvrp50_best.pt --out model.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


# ═══════════════════════════════════════════════════════════════════════════
# Model layers (same as train_job.py — needed to reconstruct from checkpoint)
# ═══════════════════════════════════════════════════════════════════════════

class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.hd = dim // heads
        self.scale = self.hd ** -0.5
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, q, kv, mask=None):
        B = q.shape[0]
        Q = self.Wq(q).view(B, -1, self.heads, self.hd).transpose(1, 2)
        K = self.Wk(kv).view(B, -1, self.heads, self.hd).transpose(1, 2)
        V = self.Wv(kv).view(B, -1, self.heads, self.hd).transpose(1, 2)
        a = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(1), float('-inf'))
        a = F.softmax(a, dim=-1)
        o = torch.matmul(a, V).transpose(1, 2).contiguous().view(B, -1, self.dim)
        return self.Wo(o)


class EncLayer(nn.Module):
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.mha = MHA(dim, heads)
        self.n1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.n1(x + self.mha(x, x, mask))
        return self.n2(x + self.ff(x))


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, ff_dim):
        super().__init__()
        self.proj = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([
            EncLayer(embedding_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

    def forward(self, locs, demands):
        x = torch.cat([locs, demands], dim=-1)
        x = self.proj(x)
        for l in self.layers:
            x = l(x)
        return x, x.mean(dim=1, keepdim=True)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        d = embedding_dim
        self.scale = (d // num_heads) ** -0.5
        self.clip = 10.0
        self.ctx_proj = nn.Linear(d * 3, d)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)

    def forward(self, node_emb, graph_emb, demands, capacity, max_steps):
        B, Np1, _ = node_emb.shape
        device = node_emb.device
        keys = self.Wk(node_emb)
        first = node_emb[:, 0:1, :]

        visited = torch.zeros(B, Np1, dtype=torch.bool, device=device)
        cur_loc = torch.zeros(B, dtype=torch.long, device=device)
        rem_cap = capacity.clone()

        actions_list, logp_list = [], []

        for _ in range(max_steps):
            cur_emb = node_emb[torch.arange(B), cur_loc].unsqueeze(1)
            ctx = self.ctx_proj(torch.cat([graph_emb, first, cur_emb], -1)).squeeze(1)
            q = self.Wq(ctx)

            compat = (q.unsqueeze(1) * keys).sum(-1) * self.scale
            compat = self.clip * torch.tanh(compat / self.clip)
            compat[torch.arange(B), cur_loc] = float('-inf')
            compat = compat.masked_fill(visited, float('-inf'))
            compat = compat.masked_fill(demands > rem_cap, float('-inf'))
            compat[:, 0] = torch.where(
                compat[:, 1:].ne(float('-inf')).any(dim=1),
                compat[:, 0],
                torch.tensor(1.0, device=device)
            )

            probs = F.softmax(compat, dim=-1)
            action = torch.argmax(probs, dim=-1)
            lp = torch.log(probs[torch.arange(B), action] + 1e-20)

            actions_list.append(action)
            logp_list.append(lp)

            not_depot = action != 0
            idxs = torch.arange(Np1, device=device).unsqueeze(0)
            visited = visited | ((idxs == action.unsqueeze(1)) & not_depot.unsqueeze(1))
            cur_loc = action

            at_depot = (action == 0).unsqueeze(1)
            consumed = demands[torch.arange(B), action].unsqueeze(1)
            rem_cap = torch.where(at_depot, capacity, rem_cap - consumed)

        return torch.stack(actions_list, -1), torch.stack(logp_list, -1)


class CVRPModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, ff_dim):
        super().__init__()
        self.encoder = Encoder(embedding_dim, num_heads, num_encoder_layers, ff_dim)
        self.decoder = Decoder(embedding_dim, num_heads)

    def forward(self, locs, demands, capacity, max_steps):
        dem_flat = demands.squeeze(-1) if demands.dim() == 3 else demands
        node_emb, graph_emb = self.encoder(locs, demands)
        return self.decoder(node_emb, graph_emb, dem_flat, capacity, max_steps)


class ONNXExportWrapper(nn.Module):
    """Wrapper that prepends depot internally, matching v2rmp MCP interface."""
    def __init__(self, model, max_steps):
        super().__init__()
        self.model = model
        self.max_steps = max_steps

    def forward(self, locs, demand, capacity):
        B, N = locs.shape[0], locs.shape[1]
        depot_l = torch.full((B, 1, 2), 0.5, device=locs.device, dtype=locs.dtype)
        full_l = torch.cat([depot_l, locs], 1)
        depot_d = torch.zeros(B, 1, 1, device=demand.device, dtype=demand.dtype)
        full_d = torch.cat([depot_d, demand], 1)
        return self.model(full_l, full_d, capacity, self.max_steps)


# ═══════════════════════════════════════════════════════════════════════════
# Export logic
# ═══════════════════════════════════════════════════════════════════════════

def load_checkpoint(path_or_repo: str, ckpt_name: str = None) -> dict:
    """Load a checkpoint dict from local path or HF Hub."""
    if ckpt_name:
        print(f"Downloading {ckpt_name} from {path_or_repo}...")
        ckpt_path = hf_hub_download(repo_id=path_or_repo, filename=ckpt_name)
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        print(f"Loading local checkpoint: {path_or_repo}")
        return torch.load(path_or_repo, map_location="cpu", weights_only=False)


def export_to_onnx(ckpt: dict, output_path: str, problem_size: int = 50,
                   max_steps: int = None):
    """Export a loaded checkpoint to ONNX."""
    
    # Extract config from checkpoint
    cfg = ckpt.get("cfg", None)
    state = ckpt.get("state", ckpt.get("model_state_dict", None))
    
    if cfg is None:
        print("No config in checkpoint, using defaults (dim=128, heads=8, layers=6)")
        embedding_dim, num_heads, num_layers, ff_dim = 128, 8, 6, 512
    elif hasattr(cfg, 'embedding_dim'):
        embedding_dim = cfg.embedding_dim
        num_heads = cfg.num_heads
        num_layers = cfg.num_encoder_layers
        ff_dim = cfg.ff_hidden_dim
    else:
        embedding_dim = cfg.get("embedding_dim", 128)
        num_heads = cfg.get("num_heads", 8)
        num_layers = cfg.get("num_encoder_layers", 6)
        ff_dim = cfg.get("ff_hidden_dim", 512)
    
    print(f"Model config: dim={embedding_dim}, heads={num_heads}, layers={num_layers}, ff={ff_dim}")
    
    # Build model
    model = CVRPModel(embedding_dim, num_heads, num_layers, ff_dim)
    model.load_state_dict(state)
    model.eval()
    
    if max_steps is None:
        max_steps = problem_size * 3
    
    wrapper = ONNXExportWrapper(model, max_steps=max_steps)
    wrapper.eval()
    
    dummy_locs = torch.randn(1, problem_size, 2)
    dummy_demand = torch.ones(1, problem_size, 1)
    dummy_capacity = torch.tensor([[1.0]])
    
    print(f"Exporting ONNX model to {output_path} (problem_size={problem_size}, max_steps={max_steps})...")
    
    from torch.export import Dim
    batch_dim = Dim("batch", min=1, max=64)
    nodes_dim = Dim("nodes", min=2, max=200)
    
    torch.onnx.export(
        wrapper,
        (dummy_locs, dummy_demand, dummy_capacity),
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['locs', 'demand', 'capacity'],
        output_names=['actions', 'log_p'],
        dynamic_shapes={
            'locs': {0: batch_dim, 1: nodes_dim},
            'demand': {0: batch_dim, 1: nodes_dim},
            'capacity': {0: batch_dim},
        },
    )
    
    # Validate
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Runtime test
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(output_path)
    r = sess.run(None, {
        'locs': dummy_locs.numpy().astype(np.float32),
        'demand': dummy_demand.numpy().astype(np.float32),
        'capacity': dummy_capacity.numpy().astype(np.float32),
    })
    print(f"✅ ONNX model exported: {output_path}")
    print(f"   actions shape: {r[0].shape}, log_p shape: {r[1].shape}")
    print(f"   First 20 actions: {r[0][0, :20]}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export v2rmp CVRP checkpoint to ONNX")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", type=str, help="HF Hub repo ID")
    src.add_argument("--local", type=str, help="Local checkpoint path")
    parser.add_argument("--ckpt", type=str, default="cvrp50_best.pt",
                        help="Checkpoint filename in repo (default: cvrp50_best.pt)")
    parser.add_argument("--out", type=str, default="model.onnx",
                        help="Output ONNX path (default: model.onnx)")
    parser.add_argument("--problem_size", type=int, default=50,
                        help="Problem size for ONNX export (default: 50)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max decoding steps (default: problem_size * 3)")
    
    args = parser.parse_args()
    
    if args.repo:
        ckpt = load_checkpoint(args.repo, args.ckpt)
    else:
        ckpt = load_checkpoint(args.local)
    
    export_to_onnx(ckpt, args.out, args.problem_size, args.max_steps)
