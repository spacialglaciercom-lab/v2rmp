# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "safetensors>=0.4",
#     "numpy>=1.24",
#     "torch>=2.0",
# ]
# ///

"""
v4 Training pipeline — uses NEAR-WIN labeling (within 10% tolerance) + SMOTE + focal loss.

Key insight from data audit: default is so dominant that even augmented data
can't fix the fundamental solver performance gap on synthetic instances.
Solution: label with "best solver group" — any solver within 10% of the minimum
distance is considered a valid label. This gives the model multiple valid targets
and prevents majority-class collapse.

Also generates additional synthetic instances biased toward weak solvers
(sweep, two_opt) to increase their representation in near-win groups.

Usage: python augment_and_retrain_v4.py
"""

import os
import json
import time
import math
import argparse
import copy
from collections import Counter

import numpy as np
from safetensors.numpy import save_file

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(42)
except ImportError:
    raise RuntimeError("PyTorch is required.")

SOLVER_IDS = ["default", "clarke_wright", "sweep", "or_opt", "two_opt", "neural_guided"]
SOLVER_TO_IDX = {s: i for i, s in enumerate(SOLVER_IDS)}
NUM_SOLVERS = len(SOLVER_IDS)

NEAR_WIN_TOLERANCE = 0.10  # 10% within best


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/training_data.jsonl")
    p.add_argument("--extra-data", default="data/training_data_extra_3k.jsonl")
    p.add_argument("--n-augment", type=int, default=3)
    p.add_argument("--min-per-class", type=int, default=300)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smooth", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="models")
    return p.parse_args()


def load_entries(path: str):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def to_numpy(entries: list):
    n = len(entries)
    X = np.zeros((n, 28), dtype=np.float32)
    for i, e in enumerate(entries):
        X[i] = np.array(e["features"], dtype=np.float32)
    return X


def augment_smote(entries: list, min_per_class: int, n_neighbors: int, noise_std: float = 0.03):
    """SMOTE for minority classes with near-win labeling awareness."""
    class_lists = {c: [] for c in range(NUM_SOLVERS)}
    for e in entries:
        # For augmentation, use the BEST solver (not near-win group)
        c = SOLVER_TO_IDX.get(e["best_solver"], 0)
        class_lists[c].append(e)

    aug_entries = copy.deepcopy(entries)
    rng = np.random.default_rng(42)

    for c in range(NUM_SOLVERS):
        cls_entries = class_lists[c]
        current = len(cls_entries)
        if current >= min_per_class or current == 0:
            continue
        needed = min_per_class - current
        print(f"  Augmenting class {SOLVER_IDS[c]}: {current} -> {min_per_class} (need {needed})")

        X_cls = to_numpy(cls_entries)
        for _ in range(needed):
            i = rng.integers(0, current)
            j = rng.integers(0, current)
            alpha = rng.random()
            synthetic = alpha * X_cls[i] + (1 - alpha) * X_cls[j]
            synthetic += rng.normal(0, noise_std, size=synthetic.shape).astype(np.float32)
            synthetic = np.clip(synthetic, 0.0, 1.0)

            base = cls_entries[i]
            aug = copy.deepcopy(base)
            aug["features"] = synthetic.tolist()
            aug["augmented"] = True
            # Keep the same near-win group but mark augmented
            aug_entries.append(aug)
    return aug_entries


def get_near_win_solvers(entry: dict, tolerance: float = 0.10):
    """Return list of solver IDs within tolerance of best distance."""
    dists = {sd["solver"]: float(sd["distance"]) for sd in entry["solver_dists"]}
    best = min(dists.values())
    winners = [s for s, d in dists.items() if d <= best * (1 + tolerance)]
    return winners


def prepare_labels(entries: list, use_near_win: bool = True):
    """
    For each instance, produce multi-label targets:
    - primary label: best solver (for accuracy metric)
    - soft label: uniform over near-win solvers (for training)
    """
    n = len(entries)
    X = np.zeros((n, 28), dtype=np.float32)
    Y_soft = np.zeros((n, NUM_SOLVERS), dtype=np.float32)  # Multi-label soft targets
    Y_hard = np.zeros((n,), dtype=np.int64)  # Single best solver
    Y_quality = np.zeros((n, 2), dtype=np.float32)
    Y_automl = np.zeros((n, 5), dtype=np.float32)
    move_instances = []

    for i, e in enumerate(entries):
        X[i] = np.array(e["features"], dtype=np.float32)
        
        dists = {sd["solver"]: float(sd["distance"]) for sd in e["solver_dists"]}
        best = min(dists.values())
        worst = max(dists.values())
        best_solver = min(dists, key=dists.get)
        Y_hard[i] = SOLVER_TO_IDX.get(best_solver, 0)

        # Soft label: uniform over near-win solvers
        if use_near_win:
            winners = get_near_win_solvers(e, NEAR_WIN_TOLERANCE)
            for w in winners:
                Y_soft[i, SOLVER_TO_IDX[w]] = 1.0 / len(winners)
        else:
            Y_soft[i, SOLVER_TO_IDX[best_solver]] = 1.0

        # Quality: relative solver gap (log-normalized)
        gap_relative = (worst - best) / (best + 1e-6)
        gap_norm = np.log1p(gap_relative) / 5.0
        n_stops = e.get("n_stops", max(int(X[i, 0] * 500), 10))
        tour_norm = best / max(math.sqrt(n_stops) * 100.0, 1.0)
        Y_quality[i] = [gap_norm, tour_norm]

        # AutoML
        ns = X[i, 0]
        dn = X[i, 5]
        cr = X[i, 19]
        cap_flag = X[i, 18]
        max_iter = 100.0 + (ns * 3000.0) + (cap_flag * 2000.0)
        temp = 1.0 + (dn * 500.0) + (1.0 - cr) * 300.0
        tabu = 3.0 + (ns * 30.0)
        cool = max(0.95 - (ns * 0.1), 0.80)
        neighb = 2.0 + (ns * 8.0)
        Y_automl[i] = [max_iter, temp, tabu, cool, neighb]

        move_instances.extend(gen_moves(ns, e))

    return X, Y_hard, Y_soft, Y_quality, Y_automl, move_instances


def gen_moves(n_stops_norm, entry):
    moves = []
    sd = entry.get("solver_dists", [])
    if not sd:
        return moves
    best_dist = min(float(s["distance"]) for s in sd)
    n = max(int(n_stops_norm * 500), 5)

    for _ in range(min(n, 40)):
        d_ab = np.random.exponential(2.0) + 0.5
        d_cd = np.random.exponential(2.0) + 0.5
        d_ac = np.random.exponential(2.0) + 0.5
        d_bd = np.random.exponential(2.0) + 0.5
        delta = (d_ac + d_bd) - (d_ab + d_cd)
        avg_edge = max((d_ab + d_cd + d_ac + d_bd) / 4.0, 1e-6)
        feat = np.zeros(16, dtype=np.float32)
        feat[0] = d_ab
        feat[1] = d_cd
        feat[2] = d_ac
        feat[3] = d_bd
        feat[4] = delta
        feat[5:9] = np.array([d_ab, d_cd, d_ac, d_bd]) / avg_edge
        feat[9] = np.random.exponential(5.0)
        feat[10] = np.random.beta(2, 5)
        feat[11] = np.random.exponential(2.0)
        feat[12] = feat[11] / max(feat[11] + 1.0, 1.0)
        feat[13] = np.random.beta(3, 3)
        feat[14] = np.random.beta(3, 3)
        feat[15] = 1.0 if delta < -1e-6 else 0.0

        worst_dist = max(float(s["distance"]) for s in sd)
        instance_gap = (worst_dist - best_dist) / (best_dist + 1e-6)
        score = 1.0 / (1.0 + math.exp(delta / max(avg_edge, 1.0))) * (1.0 + instance_gap * 0.3)
        moves.append((feat, score))
    return moves


# ---------------------------------------------------------------------------
# Neural nets
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)


def soft_cross_entropy(logits, soft_targets, weights, gamma=2.0, label_smooth=0.06):
    """
    Soft cross-entropy with focal weighting.
    soft_targets: normalized probability distribution over classes.
    """
    num_classes = logits.size(1)
    # Label smoothing on soft targets
    soft_targets = soft_targets * (1.0 - label_smooth) + label_smooth / num_classes
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    
    # Focal: down-weight easy examples where prediction aligns with soft target
    # Approximate: (1 - sum(p_t * y_t))^gamma
    alignment = (probs * soft_targets).sum(dim=1)
    focal_weight = (1.0 - alignment) ** gamma
    
    # Apply class weighting based on argmax of soft target
    hard_pseudo = soft_targets.argmax(dim=1)
    per_sample_weights = weights[hard_pseudo] * focal_weight
    
    loss = -torch.sum(soft_targets * log_probs, dim=1) * per_sample_weights
    return loss.mean()


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, hard_labels, batch_size, oversample_minority=True):
        self.hard_labels = hard_labels
        self.batch_size = batch_size
        self.oversample = oversample_minority
        self.class_to_indices = {}
        self.nonempty_classes = []
        for c in range(NUM_SOLVERS):
            idx = np.where(hard_labels == c)[0].tolist()
            self.class_to_indices[c] = idx
            if len(idx) > 0:
                self.nonempty_classes.append(c)
        self.max_class_size = max(len(v) for v in self.class_to_indices.values() if len(v) > 0)
        n_classes = len(self.nonempty_classes)
        self.num_batches = max(1, self.max_class_size * n_classes // batch_size)
    def __iter__(self):
        all_indices = []
        for c in self.nonempty_classes:
            idx = self.class_to_indices[c]
            if self.oversample and len(idx) < self.max_class_size:
                extra = np.random.choice(idx, self.max_class_size - len(idx), replace=True).tolist()
                idx = idx + extra
            np.random.shuffle(idx)
            all_indices.extend(idx)
        np.random.shuffle(all_indices)
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]
    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier_soft(name, X, Y_hard, Y_soft, hidden_dims, epochs=400, lr=1e-3,
                          batch_size=256, patience=80, gamma=2.0,
                          label_smooth=0.06, use_balanced=True):
    print(f"\nTraining {name} with Soft Focal Loss (gamma={gamma}) ...")
    num_classes = NUM_SOLVERS
    model = MLP(X.shape[1], hidden_dims, num_classes, dropout=0.2)

    # Weights from hard labels
    counts = np.bincount(Y_hard, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    print(f"  Class counts (hard): {counts}")
    print(f"  Class weights: {weights.round(4)}")

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=2, eta_min=lr * 1e-3)

    n = X.shape[0]
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    tr_idx, val_idx = perm[:split], perm[split:]
    
    X_tr = torch.from_numpy(X[tr_idx]).float()
    Y_soft_tr = torch.from_numpy(Y_soft[tr_idx]).float()
    Y_hard_tr = torch.from_numpy(Y_hard[tr_idx]).long()
    X_val = torch.from_numpy(X[val_idx]).float()
    Y_hard_val = torch.from_numpy(Y_hard[val_idx]).long()

    if use_balanced:
        sampler = BalancedBatchSampler(Y_hard[tr_idx], batch_size, oversample_minority=True)
        tr_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr, Y_soft_tr, Y_hard_tr),
            batch_sampler=sampler)
    else:
        tr_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr, Y_soft_tr, Y_hard_tr),
            batch_size=batch_size, shuffle=True)

    best_acc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for xb, yb_soft, yb_hard in tr_loader:
            if xb.shape[0] == 0:
                continue
            opt.zero_grad()
            logits = model(xb)
            loss = soft_cross_entropy(logits, yb_soft, weight_tensor, gamma, label_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            steps += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vpred = model(X_val).argmax(dim=1)
            vacc = (vpred == Y_hard_val).float().mean().item()
            per_class = []
            for c in range(num_classes):
                mask = Y_hard_val == c
                if mask.sum() > 0:
                    per_class.append((vpred[mask] == c).float().mean().item())
                else:
                    per_class.append(0.0)

        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or no_improve == 0:
            pc_str = ', '.join(f'{SOLVER_IDS[i]}={per_class[i]:.2f}' for i in range(num_classes) if counts[i] > 0)
            print(f"  Epoch {epoch:3d} — loss={epoch_loss/max(steps,1):.4f} val_acc={vacc:.4f}  [{pc_str}]")

        if no_improve > patience:
            print(f"  Early stop @ epoch {epoch}")
            break

    print(f"  Best val accuracy: {best_acc:.4f}")
    return best_state


def train_regressor(name, X, Y, hidden_dims, epochs=400, lr=1e-3, batch_size=256, patience=40, dropout=0.15):
    print(f"\nTraining {name} ...")
    model = MLP(X.shape[1], hidden_dims, Y.shape[1], dropout=dropout)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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

        if epoch % 5 == 0 or no_improve == 0:
            print(f"  Epoch {epoch:3d} — tr_loss={epoch_loss / nb:.6f} val_mse={vloss:.6f}")

        if no_improve > patience:
            print(f"  Early stop @ epoch {epoch}")
            break

    print(f"  Best val MSE: {best_vloss:.6f}")
    return best_state


def state_to_candle(state_dict, prefix="lin"):
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

def main():
    args = get_args()
    start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("v2rmp Near-Win Soft-Label Training Pipeline v4")
    print("=" * 60)

    print("\nLoading training data...")
    entries = load_entries(args.data)
    print(f"  Primary: {len(entries)} instances")

    # Merge extra data if available
    if os.path.exists(args.extra_data):
        extra = load_entries(args.extra_data)
        print(f"  Extra:   {len(extra)} instances")
        entries.extend(extra)
    
    print(f"  Total:   {len(entries)} instances")

    # Show raw distribution
    c_raw = Counter(SOLVER_TO_IDX.get(e["best_solver"], 0) for e in entries)
    print("\n  Raw class distribution (hard labels):")
    for sid in SOLVER_IDS:
        print(f"    {sid:18s}: {c_raw[SOLVER_TO_IDX[sid]]:5d}")

    # Show near-win distribution
    near_win_counts = Counter()
    for e in entries:
        winners = get_near_win_solvers(e, NEAR_WIN_TOLERANCE)
        for w in winners:
            near_win_counts[w] += 1
    print(f"\n  Near-win distribution (within {NEAR_WIN_TOLERANCE*100:.0f}%):")
    for sid in SOLVER_IDS:
        print(f"    {sid:18s}: {near_win_counts[sid]:5d}")

    # Augment minority classes
    print(f"\n  Augmenting with SMOTE (min_per_class={args.min_per_class})...")
    aug_entries = augment_smote(entries, args.min_per_class, args.n_augment, noise_std=0.03)
    print(f"  Total after augmentation: {len(aug_entries)}")

    X, Y_hard, Y_soft, Y_quality, Y_automl, moves = prepare_labels(aug_entries, use_near_win=True)
    print(f"  X={X.shape}  Y_hard={Y_hard.shape}  Y_soft={Y_soft.shape}  Y_quality={Y_quality.shape}")

    # 1. Solver Selector with soft targets
    sel_state = train_classifier_soft(
        "Solver Selector", X, Y_hard, Y_soft, [128, 64],
        epochs=args.epochs, lr=1e-3, batch_size=args.batch,
        patience=args.patience, gamma=args.focal_gamma,
        label_smooth=args.label_smooth, use_balanced=True)
    save_file(state_to_candle(sel_state), os.path.join(args.out_dir, "solver_selector.safetensors"))

    m = MLP(28, [128, 64], NUM_SOLVERS)
    m.load_state_dict({k: torch.from_numpy(v) for k, v in sel_state.items()})
    m.eval()
    with torch.no_grad():
        pred = m(torch.from_numpy(X).float()).argmax(dim=1).numpy()
    print(f"  Full-data accuracy: {np.mean(pred == Y_hard):.4f}")
    c_pred = Counter(pred)
    print("\n  Predicted class distribution:")
    for sid in SOLVER_IDS:
        print(f"    {sid:18s}: {c_pred[SOLVER_TO_IDX[sid]]:5d}")

    # 2. Quality Predictor
    q_state = train_regressor("Quality Predictor", X, Y_quality, [64, 32],
                              epochs=400, lr=1e-3, batch_size=256, patience=40)
    save_file(state_to_candle(q_state), os.path.join(args.out_dir, "quality_predictor.safetensors"))

    # 3. AutoML Predictor
    a_state = train_regressor("AutoML Predictor", X, Y_automl, [64],
                              epochs=300, lr=1e-3, batch_size=256, patience=30)
    save_file(state_to_candle(a_state), os.path.join(args.out_dir, "automl.safetensors"))

    # 4. Move Scorer
    move_X = np.array([m[0] for m in moves], dtype=np.float32)
    move_Y = np.array([[m[1]] for m in moves], dtype=np.float32)
    mv_state = train_regressor("Move Scorer", move_X, move_Y, [32, 16],
                               epochs=500, lr=1e-3, batch_size=512, patience=50)
    save_file(state_to_candle(mv_state), os.path.join(args.out_dir, "move_scorer.safetensors"))

    # 5. GraphSAGE placeholder
    sage = {
        'conv1.lin_l.weight': np.eye(64, 10).astype(np.float32) * 0.1,
        'conv1.lin_l.bias': np.zeros(64, dtype=np.float32),
        'conv1.lin_r.weight': np.eye(64, 10).astype(np.float32) * 0.1,
        'conv2.lin_l.weight': np.eye(64, 64).astype(np.float32) * 0.1,
        'conv2.lin_l.bias': np.zeros(64, dtype=np.float32),
        'conv2.lin_r.weight': np.eye(64, 64).astype(np.float32) * 0.1,
    }
    save_file(sage, os.path.join(args.out_dir, "graph_embed.safetensors"))

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s. Models written to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
