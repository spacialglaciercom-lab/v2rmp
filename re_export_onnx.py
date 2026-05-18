#!/usr/bin/env python3
"""Download the v2rmp CVRP checkpoint and re-export ONNX with embedded weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from huggingface_hub import hf_hub_download, HfApi
import numpy as np
import os
import sys

# Must match train_job.py exactly
@dataclass
class Config:
    problem_size: int = 50
    embedding_dim: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 3
    ff_hidden_dim: int = 512
    batch_size: int = 512
    epochs: int = 500
    lr: float = 1e-4
    weight_decay: float = 1e-6
    max_demand: int = 9
    capacity: float = 40.0
    pomo_aug: int = 8
    log_every: int = 10
    save_every: int = 100
    hub_model_id: str = ""
    push_to_hub: bool = False


class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.hd = dim // heads
        self.scale = self.hd ** -0.5
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, q, kv, mask=None):
        B, S, _ = q.shape
        _, T, _ = kv.shape
        Q = self.Wq(q).view(B, S, self.heads, self.hd).transpose(1, 2)
        K = self.Wk(kv).view(B, T, self.heads, self.hd).transpose(1, 2)
        V = self.Wv(kv).view(B, T, self.heads, self.hd).transpose(1, 2)
        a = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(1).expand_as(a), float('-inf'))
        a = F.softmax(a, dim=-1)
        o = torch.matmul(a, V).transpose(1, 2).contiguous().view(B, S, -1)
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
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Linear(4, cfg.embedding_dim)  # [x, y, z, demand]
        self.layers = nn.ModuleList([
            EncLayer(cfg.embedding_dim, cfg.num_heads, cfg.ff_hidden_dim)
            for _ in range(cfg.num_encoder_layers)
        ])

    def forward(self, locs, demands):
        dem_exp = demands.unsqueeze(-1)  # [B, N+1, 1]
        x = torch.cat([locs, dem_exp], dim=-1)  # [B, N+1, 4]
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x, x.mean(dim=1, keepdim=True)


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.embedding_dim
        self.clip = 10.0
        self.scale = (d // cfg.num_heads) ** -0.5
        self.ctx_proj = nn.Sequential(nn.Linear(d * 3, d), nn.ReLU())
        self.glimpse = MHA(d, cfg.num_heads)
        self.Wref = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)

    def forward(self, node_emb, graph_emb, demands, capacity, greedy=True, max_steps=200):
        B, Np1, d = node_emb.shape
        device = node_emb.device
        ref = self.Wref(node_emb)
        visited = torch.zeros(B, Np1, dtype=torch.bool, device=device)
        cur_loc = torch.zeros(B, dtype=torch.long, device=device)
        rem_cap = capacity.expand(B, 1).clone()
        first_node = node_emb[:, 0:1, :]
        actions_list, logp_list = [], []

        for _ in range(max_steps):
            cur_emb = node_emb[torch.arange(B, device=device), cur_loc].unsqueeze(1)
            ctx = self.ctx_proj(torch.cat([graph_emb.expand(B, 1, d), first_node, cur_emb], dim=-1))
            glimpse_out = self.glimpse(ctx, node_emb)
            q = self.Wq(glimpse_out.squeeze(1))
            compat = (q.unsqueeze(1) * ref).sum(-1) * self.scale
            compat = self.clip * torch.tanh(compat / self.clip)
            compat[torch.arange(B, device=device), cur_loc] = float('-inf')
            compat = compat.masked_fill(visited, float('-inf'))
            cap_mask = demands > rem_cap
            compat = compat.masked_fill(cap_mask, float('-inf'))
            all_inf = compat[:, 1:].isinf().all(dim=1)
            compat[:, 0] = torch.where(all_inf, torch.zeros_like(compat[:, 0]), compat[:, 0])
            probs = F.softmax(compat, dim=-1)
            probs = torch.nan_to_num(probs, nan=1.0 / Np1)
            if greedy:
                action = probs.argmax(dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)
            log_prob = torch.log(probs[torch.arange(B, device=device), action] + 1e-20)
            actions_list.append(action)
            logp_list.append(log_prob)
            is_depot = (action == 0)
            idxs = torch.arange(Np1, device=device).unsqueeze(0)
            visited = visited | ((idxs == action.unsqueeze(1)) & ~is_depot.unsqueeze(1))
            cur_loc = action
            consumed = demands[torch.arange(B, device=device), action].unsqueeze(1)
            rem_cap = torch.where(is_depot.unsqueeze(1), capacity.expand(B, 1), rem_cap - consumed)

        return torch.stack(actions_list, dim=-1), torch.stack(logp_list, dim=-1)


class CVRPModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, locs, demands, capacity, greedy=True, max_steps=None):
        if max_steps is None:
            max_steps = (locs.shape[1] - 1) * 3
        node_emb, graph_emb = self.encoder(locs, demands)
        return self.decoder(node_emb, graph_emb, demands, capacity, greedy, max_steps)


class ONNXModel(nn.Module):
    def __init__(self, model: CVRPModel, max_steps: int = 200):
        super().__init__()
        self.model = model
        self.max_steps = max_steps

    def forward(self, locs, demand, capacity):
        B, N = locs.shape[0], locs.shape[1]
        depot_l = torch.zeros(B, 1, 3, device=locs.device, dtype=locs.dtype)
        depot_l[:, :, :2] = 0.5  # depot at (0.5, 0.5, 0.0)
        full_l = torch.cat([depot_l, locs], 1)
        depot_d = torch.zeros(B, 1, 1, device=demand.device, dtype=demand.dtype)
        full_d = torch.cat([depot_d, demand], 1).squeeze(-1)
        return self.model(full_l, full_d, capacity, greedy=True, max_steps=self.max_steps)


if __name__ == "__main__":
    print("Downloading checkpoint...")
    ckpt_path = hf_hub_download(
        repo_id='aerialblancaservices/v2rmp-routing-ml',
        filename='cvrp50_best.pt'
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    print(f"Checkpoint: epoch={ckpt['epoch']}, size={cfg.problem_size}, dim={cfg.embedding_dim}, "
          f"layers={cfg.num_encoder_layers}, capacity={cfg.capacity}")

    model = CVRPModel(cfg)
    model.load_state_dict(ckpt['state'])
    model.eval()

    # Quick eval
    torch.manual_seed(42)
    customers = torch.rand(128, cfg.problem_size, 2)
    depot = torch.full((128, 1, 2), 0.5)
    locs = torch.cat([depot, customers], 1)
    demands = torch.randint(1, cfg.max_demand + 1, (128, cfg.problem_size)).float()
    demands = torch.cat([torch.zeros(128, 1), demands], 1)
    # For re-export, locs should be [B, N+1, 3]
    # If checkpoint was trained with 2D locs, pad z=0
    if locs.shape[-1] == 2:
        z_pad = torch.zeros(*locs.shape[:-1], 1)
        locs = torch.cat([locs, z_pad], dim=-1)
    capacity = torch.full((128, 1), cfg.capacity)

    with torch.no_grad():
        a, _ = model(locs, demands, capacity, greedy=True, max_steps=cfg.problem_size * 3)

    # Compute costs manually
    costs = []
    N = cfg.problem_size
    for b in range(128):
        visited = set()
        cost = 0.0
        prev = 0
        for t in range(a.shape[1]):
            node = a[b, t].item()
            cost += torch.norm(locs[b, prev] - locs[b, node]).item()
            if node != 0:
                visited.add(node)
            prev = node
            if len(visited) == N and node == 0:
                break
        costs.append(cost)
    print(f"Greedy: {np.mean(costs):.4f} ± {np.std(costs):.4f}")

    # ONNX export
    max_steps = cfg.problem_size * 3
    wrapper = ONNXModel(model, max_steps=max_steps).cpu().eval()
    dummy_l = torch.randn(1, cfg.problem_size, 3)  # [x, y, z]
    dummy_d = torch.ones(1, cfg.problem_size, 1)
    dummy_c = torch.tensor([[1.0]])

    onnx_path = "/tmp/cvrp50_model_v2.onnx"
    print("Exporting ONNX...")

    from torch.export import Dim
    batch_dim = Dim("batch", min=1, max=64)
    nodes_dim = Dim("nodes", min=2, max=200)

    torch.onnx.export(
        wrapper, (dummy_l, dummy_d, dummy_c),
        onnx_path,
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

    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.save_model(model_onnx, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"OK: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    r = sess.run(None, {
        'locs': dummy_l.numpy().astype(np.float32),
        'demand': dummy_d.numpy().astype(np.float32),
        'capacity': dummy_c.numpy().astype(np.float32),
    })
    print(f"Verified: actions={r[0].shape}, log_p={r[1].shape}")
    print(f"First 30 actions: {r[0][0, :30]}")

    # Variable-size test
    for n in [10, 20]:
        l2 = np.random.rand(1, n, 2).astype(np.float32)
        d2 = np.ones((1, n, 1), dtype=np.float32) * 5
        c2 = np.full((1, 1), 20.0, dtype=np.float32)
        r2 = sess.run(None, {'locs': l2, 'demand': d2, 'capacity': c2})
        unique = set(int(a) for a in r2[0][0]) - {0}
        print(f"N={n}: visited {len(unique)}/{n}")

    # Upload
    print("\nUploading consolidated ONNX to Hub...")
    api = HfApi(token=open("/home/rmp/.cache/huggingface/token").read().strip())
    api.delete_file('cvrp50_model.onnx', repo_id="aerialblancaservices/v2rmp-routing-ml", repo_type="model")
    api.upload_file(
        path_or_fileobj=onnx_path,
        path_in_repo="cvrp50_model.onnx",
        repo_id="aerialblancaservices/v2rmp-routing-ml",
        repo_type="model",
    )
    print("✅ Uploaded!")
    print("DONE")
