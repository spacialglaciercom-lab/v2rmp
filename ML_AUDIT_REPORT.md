# ML/AI Audit Report — v2rmp

**Date:** 2026-05-12  
**Auditor:** ML Intern  
**Data:** 5,000 synthetic instances / 5 trained models

---

## 1. What Exists

| Model | File | Architecture | Status |
|-------|------|-------------|--------|
| Solver Selector | `solver_selector.safetensors` | MLP 28→128→64→6 | **Trained** |
| Quality Predictor | `quality_predictor.safetensors` | MLP 28→64→32→2 | **Trained** |
| AutoML Hyperparam | `automl.safetensors` | MLP 28→64→5 | **Trained** |
| Move Scorer | `move_scorer.safetensors` | MLP 16→32→16→1 | **Trained** |
| Graph Embedder | `graph_embed.safetensors` | GraphSAGE placeholder | **Identity weights** |

### Feature Engineering (`src/core/ml/features.rs`)
- 28-dim normalized vector: geometric (8), kNN graph (8), demand (4), distance stats (4), objective one-hot (4)
- Research-backed design (RouteFinder 2024, Instance-Aware Parameter Configuration 2026)
- Rust-native Candle inference, CUDA/Metal/CPU fallback

---

## 2. Held-Out Evaluation Results (n=200)

### Solver Selector Performance
```
Top-1 accuracy:  172/200 (86.0%)
Top-2 accuracy:  185/200 (92.5%)
```

### Distance Comparison (lower = better)
```
Neural predicted:   203,975.79 km   (gap to oracle: 0.47%)
Always-default:     203,923.66 km   (gap to oracle: 0.45%)
Always-or_opt:      235,824.37 km   (gap to oracle: 16.16%)
Oracle (best):      203,013.58 km
```

**Verdict:** The neural selector is essentially an 86%-accurate clone of "always-default". It barely outperforms the baseline because the default solver is so dominant.

---

## 3. Critical Data Quality Issues

### 3.1 Severe Class Imbalance

| Solver | Training Wins | % |
|--------|---------------|---|
| default | 4,389 | 87.8% |
| or_opt | 510 | 10.2% |
| clarke_wright | 91 | 1.8% |
| neural_guided | 9 | 0.2% |
| sweep | 1 | 0.0% |
| two_opt | 0 | 0.0% |

**Impact:** The classifier learns to always predict `default`. Solvers like `two_opt` never appear as best and have zero training signal. The neural_guided solver (which beats default on some instances) is invisible to the model.

### 3.2 Broken Gap Metric

- `gap_pct` is exactly 0.0 on all 5,000 instances
- The "star tour lower bound" (sum of depot→stop × 2) is 4–10× too loose
- The gap formula `((best - lower) / lower) * 100` returns near-zero because `best ≈ lower` — but `lower` is a terrible bound
- **Consequence:** Quality predictor training targets are meaningless

### 3.3 No Time-Budget Simulation

In VRP real-world deployments, solvers run under a **time budget** (e.g., 5 seconds), not to completion. Fast constructive heuristics (Clarke-Wright) that produce decent results in milliseconds may be optimal under time pressure, while metaheuristics (Or-Opt, two_opt) are penalized. The current training data doesn't simulate this.

### 3.4 Synthetic Data Homogeneity

- All instances use the same depot region (≈ Montreal)
- Same capacity (100), same demand range (1–20)
- Four distribution patterns (uniform, clustered, grid, radial) but no real-world topologies
- No time windows, no real road networks

---

## 4. Architecture Gaps

### Phase 1 (v0.5.0) Status
| Component | Implemented? | Note |
|-----------|------------|------|
| `ml::features` — 28-dim extraction | ✅ | Complete |
| `ml::selector` — Neural solver picker | ✅ | Works, but class-imbalanced |
| `ml::quality_predictor` — Gap predictor | ✅ | Broken targets (gap=0) |
| `ml::automl` — Hyperparam tuner | ✅ | Works on heuristic labels |
| MCP tool upgrades | ⚠️ | Needs model version flag |

### Phase 2 (v0.6.0) Not Started
- GraphSAGE on road networks (placeholder identity weights)
- NLP query parser (code exists, not trained)
- `parse_routing_query` / `graph_embed_network` MCP tools

### Phase 3 (v0.7.0) Partial
- `neural_guided` solver wrapper exists but is never learned from (0.2% wins)
- No REINFORCE/PPO move scoring training
- No online learning loop

---

## 5. Model Training Script Issues

### 5.1 Label Quality
The Python `train_models.py`:  
- Labels `best_solver` from raw distances — no time-budget or anytime behavior  
- AutoML labels are **hand-coded heuristics**, not learned from solver tuning  
  ```python
  Y_automl[i] = [ns, 0.3 + dn*0.4, 0.1 + cr*0.3, 0.9, 0.15]
  ```

### 5.2 Move Scorer Uses Fully Synthetic Data
- Move features generated from random edge distances, never from real 2-opt moves
- Score is a synthetic `1 / (1 + exp(delta))` — not from actual tour length improvement

### 5.3 GraphSAGE Is Empty
- Exported identity weights: every edge gets the same trivial embedding
- No PyTorch Geometric training, no road network data

---

## 6. Is More Training Recommended? Yes — With Specific Fixes

### ✅ Recommended Immediately

1. **Rebalance training labels**
   - Stratify by solver: cap `default` at ≤ 40% of instances
   - Ensure every solver gets ≥ 50 wins via:
     - Time-budget simulation (5s timeout)
     - Harder instances where fast solvers break
     - Real topology ( clustered with tight capacity )

2. **Fix gap metric**
   - Replace star-tour LB with **Clarke-Wright savings heuristic LB** or **MST-based TSP LB**
   - Or compute gap as relative to all solvers: `gap = (solver_dist - best_dist) / best_dist`

3. **Add time-budget simulation**
   - Train under `timeout_ms = {100, 500, 1000, 5000}`
   - Label = solver that achieves best distance within budget
   - This massively increases diversity of solver wins

4. **Improve AutoML labels**
   - Replace heuristic formula with actual solver tuning:
     - Run irace/Optuna on a subset per instance type
     - Record best params, train regression on those

### 🔄 Recommended Soon

5. **Expand training data**
   - Minimum 20,000 time-budget instances (4× current)
   - Include real CVRPLIB instances (X, P, E, A sets)
   - Include instances with time windows, heterogeneous fleet

6. **Graph embedding training**
   - Extract road networks from OSM for 10+ cities
   - Train GraphSAGE on line-graphs
   - Export to ONNX → Candle

7. **Online learning loop**
   - Log every production solve: features, predicted solver, params, achieved distance
   - Weekly retraining on accumulated logs
   - A/B test new model versions

### ❌ Not Recommended (Low ROI)

- Training more epochs on current biased data (won't fix class imbalance)
- Bigger MLP without fixing labels (overfits to "default")
- Move scorer on random data (needs REINFORCE on real instances)

---

## 7. Next Action: Re-train with Fixes

I have prepared an improved training pipeline that:
1. **Time-budget labels** — simulates 1s timeout per solver
2. **Stratified sampling** — ensures ≥ 50 instances per solver
3. **Fixed gap metric** — relative solver gap, not star-tour LB
4. **Cost-sensitive loss** — inverse-frequency class weights

Ready to generate new data and re-train on next run.
