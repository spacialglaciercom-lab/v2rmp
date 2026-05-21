# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "safetensors>=0.4",
#     "numpy>=1.24",
#     "torch>=2.0",
# ]
# ///

"""
Offline training pipeline for v2rmp Neural Models.

Trains on data/training_data.jsonl (produced by `cargo run --bin generate-training-data`).

Models:
  1. Solver Selector — classifier predicting best_solver from 28-dim instance features.
  2. Quality Predictor — regression (gap + tour_norm).
  3. AutoML Predictor — regression (5 hyperparams).
  4. Move Scorer — regression on synthetic 2-opt move features.
  5. GraphSAGE — placeholder identity weights.

All exported as Candle-compatible safetensors.
"""

import os
import json
import time
import numpy as np
from safetensors.numpy import save_file

try:
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
except ImportError:
    raise RuntimeError("PyTorch is required. Install with: pip install torch")

SOLVER_IDS = ["default", "clarke_wright", "sweep", "or_opt", "two_opt", "neural_guided"]
SOLVER_TO_IDX = {s: i for i, s in enumerate(SOLVER_IDS)}
NUM_SOLVERS = len(SOLVER_IDS)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(path="data/training_data.jsonl"):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries

def prepare_labels(entries):
    n = len(entries)
    X = np.zeros((n, 28), dtype=np.float32)
    Y_label = np.zeros((n,), dtype=np.int64)
    Y_quality = np.zeros((n, 2), dtype=np.float32)
    Y_automl = np.zeros((n, 5), dtype=np.float32)
    move_instances = []

    for i, e in enumerate(entries):
        X[i] = np.array(e["features"], dtype=np.float32)

        best = e["best_solver"]
        Y_label[i] = SOLVER_TO_IDX.get(best, 0)

        dists = {sd["solver"]: float(sd["distance"]) for sd in e["solver_dists"]}
        best_dist = min(dists.values())
        worst_dist = max(dists.values())

        gap_pct = 0.0
        if worst_dist > best_dist:
            gap_pct = ((worst_dist - best_dist) / (worst_dist + 1e-6)) * 50.0
        tour_norm = min(best_dist / 5000.0, 1.0)
        Y_quality[i, 0] = gap_pct / 50.0
        Y_quality[i, 1] = tour_norm

        ns = X[i, 0]
        dn = X[i, 5]
        cr = X[i, 19]
        Y_automl[i] = [ns, 0.3 + dn * 0.4, 0.1 + cr * 0.3, 0.9, 0.15]

        move_instances.extend(gen_moves(ns))

    return X, Y_label, Y_quality, Y_automl, move_instances

def gen_moves(n_stops_norm):
    moves = []
    n = max(int(n_stops_norm * 500), 10)
    for _ in range(min(n, 50)):
        d_ab = np.random.rand() * 10.0
        d_cd = np.random.rand() * 10.0
        d_ac = np.random.rand() * 10.0
        d_bd = np.random.rand() * 10.0
        delta = (d_ac + d_bd) - (d_ab + d_cd)
        avg_edge = max((d_ab + d_cd + d_ac + d_bd) / 4.0, 1e-6)
        feat = np.zeros(16, dtype=np.float32)
        feat[0] = d_ab
        feat[1] = d_cd
        feat[2] = d_ac
        feat[3] = d_bd
        feat[4] = delta
        feat[5:9] = np.array([d_ab, d_cd, d_ac, d_bd]) / avg_edge
        feat[9] = np.random.rand() * 20.0
        feat[10] = np.random.rand()
        feat[11] = np.random.rand() * n_stops_norm * 10
        feat[12] = feat[11] / max(n_stops_norm * 10, 1.0)
        feat[13:15] = np.random.rand(2)
        feat[15] = 1.0 if delta < -1e-6 else 0.0
        score = 1.0 / (1.0 + np.exp(delta))
        moves.append((feat, score))
    return moves

# ---------------------------------------------------------------------------
# PyTorch training helpers
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def train_classifier(name, X, Y, hidden_dims, epochs=int(os.environ.get('EPOCHS', 400)), lr=1e-3, batch_size=128, patience=40):
    print(f"\nTraining {name} ...")
    num_classes = int(Y.max()) + 1
    model = MLP(X.shape[1], hidden_dims, num_classes)

    # Smooth class weights: sqrt-inverse frequency
    counts = np.bincount(Y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = np.sqrt(1.0 / counts)
    weights = weights / weights.sum() * num_classes
    print(f"  Class counts: {counts}")
    print(f"  Class weights: {weights.round(3)}")

    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, patience // 3))

    n = X.shape[0]
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    tr_idx, val_idx = perm[:split], perm[split:]
    X_tr, Y_tr = torch.from_numpy(X[tr_idx]).float(), torch.from_numpy(Y[tr_idx]).long()
    X_val, Y_val = torch.from_numpy(X[val_idx]).float(), torch.from_numpy(Y[val_idx]).long()

    best_acc = -1.0
    best_state = None
    no_improve = 0
    nb = max(X_tr.shape[0] // batch_size, 1)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        p = np.random.permutation(X_tr.shape[0])
        for b in range(nb):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            xb, yb = X_tr[p][sl], Y_tr[p][sl]
            if xb.shape[0] == 0:
                continue
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vpred = model(X_val).argmax(dim=1)
            vacc = (vpred == Y_val).float().mean().item()
            vloss = crit(model(X_val), Y_val).item()

        sched.step(vloss)

        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 40 == 0 or no_improve == 0:
            print(f"  Epoch {epoch:3d} — tr_loss={epoch_loss / nb:.4f} val_acc={vacc:.4f}")

        if no_improve > patience:
            print(f"  Early stop @ epoch {epoch}")
            break

    print(f"  Best val accuracy: {best_acc:.4f}")
    return best_state

def train_regressor(name, X, Y, hidden_dims, epochs=int(os.environ.get('EPOCHS', 400)), lr=1e-3, batch_size=256, patience=40):
    print(f"\nTraining {name} ...")
    model = MLP(X.shape[1], hidden_dims, Y.shape[1])
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, patience // 3))

    n = X.shape[0]
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    tr_idx, val_idx = perm[:split], perm[split:]
    X_tr, Y_tr = torch.from_numpy(X[tr_idx]).float(), torch.from_numpy(Y[tr_idx]).float()
    X_val, Y_val = torch.from_numpy(X[val_idx]).float(), torch.from_numpy(Y[val_idx]).float()

    best_vloss = float('inf')
    best_state = None
    no_improve = 0
    nb = max(X_tr.shape[0] // batch_size, 1)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        p = np.random.permutation(X_tr.shape[0])
        for b in range(nb):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            xb, yb = X_tr[p][sl], Y_tr[p][sl]
            if xb.shape[0] == 0:
                continue
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vloss = crit(model(X_val), Y_val).item()

        sched.step(vloss)

        if vloss < best_vloss:
            best_vloss = vloss
            best_state = {k: v.cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or no_improve == 0:
            print(f"  Epoch {epoch:3d} — tr_loss={epoch_loss / nb:.6f} val_mse={vloss:.6f}")

        if no_improve > patience:
            print(f"  Early stop @ epoch {epoch}")
            break

    print(f"  Best val MSE: {best_vloss:.6f}")
    return best_state

def state_to_candle(state_dict, prefix="lin"):
    """Map PyTorch Sequential keys (net.0.weight, net.0.bias, ...) to Candle keys."""
    result = {}
    weight_keys = sorted([k for k in state_dict.keys() if "weight" in k])
    bias_keys = sorted([k for k in state_dict.keys() if "bias" in k])
    for i, k in enumerate(weight_keys, 1):
        result[f"{prefix}{i}.weight"] = state_dict[k]
    for i, k in enumerate(bias_keys, 1):
        result[f"{prefix}{i}.bias"] = state_dict[k]
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()
    os.makedirs("models", exist_ok=True)
    np.random.seed(42)

    print("Loading training data...")
    entries = load_training_data("data/training_data.jsonl")
    print(f"  {len(entries)} instances")

    X, Y_label, Y_quality, Y_automl, moves = prepare_labels(entries)
    print(f"  X={X.shape}  Y_label={Y_label.shape}  Y_quality={Y_quality.shape}")

    # 1. Solver Selector (classifier)
    sel_state = train_classifier("Solver Selector", X, Y_label, [128, 64],
                                 epochs=int(os.environ.get('EPOCHS', 600)), lr=1e-3, batch_size=128, patience=60)
    save_file(state_to_candle(sel_state), "models/solver_selector.safetensors")

    # Quick sanity check on full data
    m = MLP(28, [128, 64], NUM_SOLVERS)
    m.load_state_dict({k: torch.from_numpy(v) for k, v in sel_state.items()})
    m.eval()
    with torch.no_grad():
        pred = m(torch.from_numpy(X).float()).argmax(dim=1).numpy()
    print(f"  Train accuracy: {np.mean(pred == Y_label):.4f}")
    for j, sid in enumerate(SOLVER_IDS):
        print(f"    {sid:15s} pred={np.sum(pred == j):4d}  true={np.sum(Y_label == j):4d}")

    # 2. Quality Predictor
    q_state = train_regressor("Quality Predictor", X, Y_quality, [64, 32],
                              epochs=int(os.environ.get('EPOCHS', 400)), lr=1e-3, batch_size=256, patience=40)
    save_file(state_to_candle(q_state), "models/quality_predictor.safetensors")

    # 3. AutoML Predictor
    a_state = train_regressor("AutoML Predictor", X, Y_automl, [64],
                              epochs=int(os.environ.get('EPOCHS', 300)), lr=1e-3, batch_size=256, patience=30)
    save_file(state_to_candle(a_state), "models/automl.safetensors")

    # 4. Move Scorer
    move_X = np.array([m[0] for m in moves], dtype=np.float32)
    move_Y = np.array([[m[1]] for m in moves], dtype=np.float32)
    mv_state = train_regressor("Move Scorer", move_X, move_Y, [32, 16],
                                 epochs=int(os.environ.get('EPOCHS', 500)), lr=1e-3, batch_size=512, patience=50)
    save_file(state_to_candle(mv_state), "models/move_scorer.safetensors")

    # 5. GraphSAGE placeholder
    sage = {
        'conv1.lin_l.weight': np.eye(64, 10).astype(np.float32) * 0.1,
        'conv1.lin_l.bias': np.zeros(64, dtype=np.float32),
        'conv1.lin_r.weight': np.eye(64, 10).astype(np.float32) * 0.1,
        'conv2.lin_l.weight': np.eye(64, 64).astype(np.float32) * 0.1,
        'conv2.lin_l.bias': np.zeros(64, dtype=np.float32),
        'conv2.lin_r.weight': np.eye(64, 64).astype(np.float32) * 0.1,
    }
    save_file(sage, "models/graph_embed.safetensors")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Models written to models/")
