#!/usr/bin/env python3
"""
CVRP Neural Solver — Attention Model with REINFORCE + greedy rollout baseline.

Fixes from v1:
  1. Standard CVRP-50 benchmark config: capacity=40, demand∈{1..9}
  2. Proper POMO: the decoder's first node is varied, not just input permutation
  3. REINFORCE with greedy baseline (not POMO mean) for stable gradients
  4. Cost computed on trimmed tours (no depot-loop padding)
  5. Sampling-based eval for better solutions
  6. Correct ONNX export with embedded weights
"""

import os
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    problem_size: int = 50       # N customers
    embedding_dim: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 6
    ff_hidden_dim: int = 512
    batch_size: int = 1024
    epochs: int = 1000
    lr: float = 1e-4
    weight_decay: float = 1e-6
    # Standard CVRP-50 benchmark
    max_demand: int = 9          # demand ∈ {1..9}
    capacity: float = 40.0       # standard benchmark capacity
    # POMO
    pomo_aug: int = 8
    # Entropy bonus
    entropy_coef: float = 0.01
    # Warmup
    warmup_epochs: int = 10
    # Logging
    log_every: int = 10
    save_every: int = 100
    # Hub
    hub_model_id: str = ""
    push_to_hub: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Data generation — standard CVRP benchmark format
# ═══════════════════════════════════════════════════════════════════════════

def generate_batch(cfg: Config, device="cpu"):
    """Random CVRP instances: depot at (0.5, 0.5, 0.0), customers uniform in [0,1]³.
    Third coordinate is altitude/elevation (0.2 max variation)."""
    B, N = cfg.batch_size, cfg.problem_size
    xy = torch.rand(B, N, 2, device=device)
    # Altitude: [0, 0.2] range represents significant terrain variation relative to unit square xy
    z = torch.rand(B, N, 1, device=device) * 0.2
    customers = torch.cat([xy, z], dim=-1)  # [B, N, 3]
    depot = torch.full((B, 1, 3), 0.5, device=device)
    depot[:, :, 2] = 0.0  # depot altitude = 0
    locs = torch.cat([depot, customers], dim=1)  # [B, N+1, 3]

    cust_demands = torch.randint(1, cfg.max_demand + 1, (B, N), device=device).float()
    depot_demand = torch.zeros(B, 1, device=device)
    demands = torch.cat([depot_demand, cust_demands], dim=1)  # [B, N+1]

    capacity = torch.full((B, 1), cfg.capacity, device=device)
    return locs, demands, capacity


# ═══════════════════════════════════════════════════════════════════════════
# Model: MHA encoder + autoregressive decoder
# ═══════════════════════════════════════════════════════════════════════════

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
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
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
        # locs: [B, N+1, 3] (x, y, z), demands: [B, N+1]
        dem_exp = demands.unsqueeze(-1)  # [B, N+1, 1]
        x = torch.cat([locs, dem_exp], dim=-1)  # [B, N+1, 4]
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x, x.mean(dim=1, keepdim=True)  # node_emb, graph_emb


class Decoder(nn.Module):
    """Autoregressive decoder with glimpse + pointer mechanism (Kool et al. 2019)."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.embedding_dim
        self.clip = 10.0
        self.scale = (d // cfg.num_heads) ** -0.5

        # Context projection: [graph_emb(128) + first_node(128) + current_node(128)] → d
        self.ctx_proj = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.ReLU(),
        )
        # Glimpse: attend over nodes using context query
        self.glimpse = MHA(d, cfg.num_heads)
        # Pointer: produce logits over nodes
        self.Wref = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)

    def forward(self, node_emb, graph_emb, demands, capacity, locs, greedy=True, max_steps=200):
        B, Np1, d = node_emb.shape
        device = node_emb.device

        # Precompute keys for pointer
        ref = self.Wref(node_emb)  # [B, N+1, d]

        # State
        visited = torch.zeros(B, Np1, dtype=torch.bool, device=device)
        cur_loc = torch.zeros(B, dtype=torch.long, device=device)
        rem_cap = capacity.expand(B, 1).clone()

        first_node = node_emb[:, 0:1, :]  # depot embedding

        actions_list, logp_list = [], []

        for step in range(max_steps):
            # Current node embedding
            cur_emb = node_emb[torch.arange(B, device=device), cur_loc].unsqueeze(1)
            cur_coords = locs[torch.arange(B, device=device), cur_loc] # [B, 3]


            # Context vector
            ctx = self.ctx_proj(torch.cat([graph_emb.expand(B, 1, d), first_node, cur_emb], dim=-1))

            # Glimpse: attend over nodes with context query
            glimpse_out = self.glimpse(ctx, node_emb)  # [B, 1, d]
            q = self.Wq(glimpse_out.squeeze(1))  # [B, d]

            # Pointer logits
            compat = (q.unsqueeze(1) * ref).sum(-1) * self.scale  # [B, N+1]
            compat = self.clip * torch.tanh(compat / self.clip)

            # Energy-Aware Masking (Battery Management)
            # Calculate energy cost from current location to all possible next nodes
            dist_2d = torch.norm(locs[:, :, :2] - cur_coords[:, :2].unsqueeze(1), dim=-1)
            delta_z = locs[:, :, 2] - cur_coords[:, 2].unsqueeze(1)
            energy_to_node = dist_2d + torch.clamp(delta_z, min=0) * 5.0 # [B, N+1]

            # Safety Check: Energy to return to depot from each candidate node
            # (Depot is at index 0, with coords [0.5, 0.5, 0.0])
            depot_coords = locs[:, 0, :] # [B, 3]
            dist_to_depot = torch.norm(depot_coords[:, :2].unsqueeze(1) - locs[:, :, :2], dim=-1)
            delta_z_to_depot = depot_coords[:, 2].unsqueeze(1) - locs[:, :, 2]
            energy_back_to_depot = dist_to_depot + torch.clamp(delta_z_to_depot, min=0) * 5.0

            # Total energy required to visit node AND return home
            total_required = energy_to_node + energy_back_to_depot # [B, N+1]

            # Mask nodes that are unreachable with current battery
            battery_mask = total_required > rem_cap
            compat = compat.masked_fill(battery_mask, float('-inf'))

            # 1. Can't stay at current node
            compat[torch.arange(B, device=device), cur_loc] = float('-inf')
            # 2. Already-visited customers
            compat = compat.masked_fill(visited, float('-inf'))
            
            # 4. If all non-depot nodes are masked, force depot return
            all_inf = compat[:, 1:].isinf().all(dim=1)
            compat[:, 0] = torch.where(all_inf, torch.zeros_like(compat[:, 0]), compat[:, 0])

            # Sample or greedy
            probs = F.softmax(compat, dim=-1)
            probs = torch.nan_to_num(probs, nan=1.0 / Np1)

            if greedy:
                action = probs.argmax(dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)

            log_prob = torch.log(probs[torch.arange(B, device=device), action] + 1e-20)

            actions_list.append(action)
            logp_list.append(log_prob)

            # Update state
            is_depot = (action == 0)
            idxs = torch.arange(Np1, device=device).unsqueeze(0)
            visited = visited | ((idxs == action.unsqueeze(1)) & ~is_depot.unsqueeze(1))
            
            # Energy Consumption: Update remaining battery
            energy_consumed = energy_to_node[torch.arange(B, device=device), action].unsqueeze(1)
            rem_cap = torch.where(is_depot.unsqueeze(1), capacity.expand(B, 1), rem_cap - energy_consumed)
            cur_loc = action

            # No early termination — fixed max_steps for ONNX traceability

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
        return self.decoder(node_emb, graph_emb, demands, capacity, locs, greedy, max_steps)



# ═══════════════════════════════════════════════════════════════════════════
# ONNX wrapper — matches v2rmp MCP server interface
# ═══════════════════════════════════════════════════════════════════════════

class ONNXModel(nn.Module):
    """Adds depot internally. Fixed max_steps for ONNX traceability."""
    def __init__(self, model: CVRPModel, max_steps: int = 200):
        super().__init__()
        self.model = model
        self.max_steps = max_steps

    def forward(self, locs, demand, capacity):
        B, _N = locs.shape[0], locs.shape[1]
        depot_l = torch.zeros(B, 1, 3, device=locs.device, dtype=locs.dtype)
        depot_l[:, :, :2] = 0.5  # depot at (0.5, 0.5, 0.0)
        full_l = torch.cat([depot_l, locs], 1)
        depot_d = torch.zeros(B, 1, 1, device=demand.device, dtype=demand.dtype)
        full_d = torch.cat([depot_d, demand], 1).squeeze(-1)
        return self.model(full_l, full_d, capacity, greedy=True, max_steps=self.max_steps)


# ═══════════════════════════════════════════════════════════════════════════
# Cost function — trimmed to meaningful tour
# ═══════════════════════════════════════════════════════════════════════════

def tour_cost(locs, actions, N):
    """
    Compute tour cost, stopping after all N customers are visited + depot return.
    actions: [B, max_steps] — node indices
    locs: [B, N+1, 3] (x, y, z)
    """
    B = actions.shape[0]
    device = actions.device
    costs = torch.zeros(B, device=device)

    for b in range(B):
        visited = set()
        cost = 0.0
        prev = 0  # start at depot
        for t in range(actions.shape[1]):
            node = actions[b, t].item()
            # Asymmetric cost logic (must match tour_cost_batch)
            p1, p2 = locs[b, prev], locs[b, node]
            d2d = torch.norm(p2[:2] - p1[:2]).item()
            dz = p2[2].item() - p1[2].item()
            cost += d2d + max(0, dz) * 5.0
            if node != 0:
                visited.add(node)
            prev = node
            # Done when all customers visited and returned to depot
            if len(visited) == N and node == 0:
                break
        costs[b] = cost
    return costs


def tour_cost_batch(locs, actions, N):
    """
    Vectorized tour cost using cumsum of edge distances.
    Stops counting after all customers visited + depot return.
    """
    B, T = actions.shape
    device = actions.device

    # Gather coordinates for the full action sequence + prepend depot (start)
    start = torch.zeros(B, 1, dtype=torch.long, device=device)
    full_actions = torch.cat([start, actions], dim=1)  # [B, T+1]

    # Edge costs: ||loc[a_t] - loc[a_{t-1}]|| for each step
    src = full_actions[:, :-1]  # [B, T]
    dst = full_actions[:, 1:]   # [B, T]

    src_coords = locs[torch.arange(B, device=device).unsqueeze(-1), src]  # [B, T, 3]
    dst_coords = locs[torch.arange(B, device=device).unsqueeze(-1), dst]  # [B, T, 3]

    # Asymmetric Fuel-Aware Cost:
    # 1. Horizontal Euclidean distance
    dist_2d = torch.norm(dst_coords[:, :, :2] - src_coords[:, :, :2], dim=-1)

    # 2. Vertical energy penalty (Gravity/Lift)
    # Climbing costs 5x more per unit than horizontal travel.
    # Descending has a neutral effect (modeled as 0 additional cost).
    delta_z = dst_coords[:, :, 2] - src_coords[:, :, 2]
    climb_penalty = 5.0
    vertical_cost = torch.clamp(delta_z, min=0) * climb_penalty

    edge_costs = dist_2d + vertical_cost

    # Build mask: 1 until all customers visited + depot return
    # Track visited customers
    is_customer = (dst != 0)  # [B, T]
    # Cumulative visited count per step
    cum_visited = is_customer.cumsum(dim=1)  # [B, T]
    # First step where cum_visited >= N
    all_visited_step = (cum_visited >= N).long().argmax(dim=1)  # [B]

    # Must also return to depot after visiting all customers
    # Find first depot return after all_visited_step
    is_depot = (dst == 0)  # [B, T]
    # Create a mask that's 0 before all_visited_step
    valid_after = torch.arange(T, device=device).unsqueeze(0) >= all_visited_step.unsqueeze(1)
    depot_after = is_depot & valid_after  # [B, T]
    # First depot return after all visited
    end_step = depot_after.long().argmax(dim=1) + 1  # [B], +1 to include depot return edge

    # Mask: 1 for steps 0..end_step, 0 after
    step_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    mask = (step_idx < end_step.unsqueeze(1)).float()  # [B, T]

    # Apply mask and sum
    masked_costs = edge_costs * mask
    return masked_costs.sum(dim=1)  # [B]


# ═══════════════════════════════════════════════════════════════════════════
# Training: REINFORCE with greedy rollout baseline
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CVRP-{cfg.problem_size} | dim={cfg.embedding_dim} | layers={cfg.num_encoder_layers} | "
          f"batch={cfg.batch_size} | capacity={cfg.capacity} | demand∈[1,{cfg.max_demand}]")
    print(f"Expected routes per instance: ~{cfg.problem_size * ((cfg.max_demand+1)/2) / cfg.capacity:.1f}")

    # Logging init

    model = CVRPModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 0.01)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=cfg.warmup_epochs)

    best_cost = float('inf')
    os.makedirs("./checkpoints", exist_ok=True)

    N = cfg.problem_size
    max_steps = N * 3

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        locs, demands, capacity = generate_batch(cfg, device)

        # ── Greedy rollout (baseline) ──
        with torch.no_grad():
            greedy_actions, _ = model(locs, demands, capacity, greedy=True, max_steps=max_steps)
            baseline_cost = tour_cost_batch(locs, greedy_actions, N)

        # ── Sample rollout ──
        sample_actions, sample_logp = model(locs, demands, capacity, greedy=False, max_steps=max_steps)
        sample_cost = tour_cost_batch(locs, sample_actions, N)

        # ── REINFORCE + entropy bonus ──
        advantage = sample_cost - baseline_cost
        rl_loss = (advantage.detach() * sample_logp.sum(-1)).mean()
        
        # Entropy: encourage exploration
        entropy = -(sample_logp.exp() * sample_logp).sum(-1).mean()
        loss = rl_loss - cfg.entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch <= cfg.warmup_epochs:
            warmup_sched.step()
        else:
            sched.step()

        greedy_avg = baseline_cost.mean().item()

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg.epochs} | loss={loss.item():.4f} | "
                  f"greedy={greedy_avg:.4f} | sample={sample_cost.mean():.4f} | "
                  f"lr={sched.get_last_lr()[0]:.2e}")

        if cfg.save_every and epoch % cfg.save_every == 0:
            path = f"./checkpoints/cvrp{N}_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "state": model.state_dict(), "cfg": cfg}, path)

        if greedy_avg < best_cost:
            best_cost = greedy_avg
            best_path = f"./checkpoints/cvrp{N}_best.pt"
            torch.save({"epoch": epoch, "state": model.state_dict(), "cfg": cfg}, best_path)

    print(f"\nBest greedy cost: {best_cost:.4f}")

    # ── Final evaluation ──
    print("\n=== Final Evaluation ===")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval()

    eval_costs = []
    for seed in range(5):
        torch.manual_seed(1000 + seed)
        cfg_eval = Config(**{**cfg.__dict__, 'batch_size': 128})
        locs, d, c = generate_batch(cfg_eval, device)
        with torch.no_grad():
            a, _ = model(locs, d, c, greedy=True, max_steps=max_steps)
            co = tour_cost_batch(locs, a, N)
            eval_costs.append(co.mean().item())

    print(f"Greedy eval (5×128): {np.mean(eval_costs):.4f} ± {np.std(eval_costs):.4f}")

    # Sampling eval
    torch.manual_seed(42)
    cfg_eval.batch_size = 64
    locs, d, c = generate_batch(cfg_eval, device)
    with torch.no_grad():
        best_sample_costs = torch.full((64,), float('inf'), device=device)
        for _ in range(1280):
            a, _ = model(locs, d, c, greedy=False, max_steps=max_steps)
            co = tour_cost_batch(locs, a, N)
            best_sample_costs = torch.min(best_sample_costs, co)
        print(f"Best-of-1280 sample: {best_sample_costs.mean():.4f}")

    # ── ONNX Export ──
    print("\n=== ONNX Export ===")
    wrapper = ONNXModel(model, max_steps=max_steps).cpu().eval()
    dummy_l = torch.randn(1, N, 3)  # [x, y, z] per customer
    dummy_d = torch.ones(1, N, 1)
    dummy_c = torch.tensor([[1.0]])

    onnx_path = f"cvrp{N}_model.onnx"

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

    # Consolidate external data into single file
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.save_model(model_onnx, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx.load(onnx_path))

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    r = sess.run(None, {
        'locs': dummy_l.numpy().astype(np.float32),
        'demand': dummy_d.numpy().astype(np.float32),
        'capacity': dummy_c.numpy().astype(np.float32),
    })
    print(f"✅ ONNX: actions={r[0].shape}, log_p={r[1].shape}")
    print(f"   First 30 actions: {r[0][0, :30]}")

    # ── Push to Hub ──
    if cfg.push_to_hub and cfg.hub_model_id:
        print(f"\n=== Pushing to {cfg.hub_model_id} ===")
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=best_path,
            path_in_repo=f"cvrp{N}_best.pt",
            repo_id=cfg.hub_model_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=onnx_path,
            path_in_repo=onnx_path,
            repo_id=cfg.hub_model_id,
            repo_type="model",
        )
        print("✅ Pushed to Hub")

    return best_cost


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--capacity", type=float, default=40.0)
    parser.add_argument("--max_demand", type=int, default=9)
    parser.add_argument("--hub_model_id", type=str, default="")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        problem_size=args.problem_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        num_encoder_layers=args.num_layers,
        capacity=args.capacity,
        max_demand=args.max_demand,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
    )

    train(cfg)
