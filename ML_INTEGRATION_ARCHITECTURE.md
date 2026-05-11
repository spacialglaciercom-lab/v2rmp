# v2rmp ML/AI Integration Architecture

## Executive Summary

This document describes the heavy ML/AI integration into v2rmp (rmpca), a Rust CLI/TUI/GUI tool for route optimization. The integration upgrades the existing rule-based ML modules to learned models, adds graph embeddings, route quality prediction, NLP query parsing, and AutoML hyperparameter tuning — all backed by published research.

## Current State (Audit)

### Existing ML/AI Touchpoints

| Module | File | Current Capabilities | Gap |
|--------|------|---------------------|-----|
| `ml` | `src/core/ml.rs` | Rule-based solver selection (`predict_solver`), route quality scoring (`score_route`), 12-dim feature vectors (`route_feature_vector`) | No learned model — purely heuristic scoring. No training pipeline. |
| `embed` | `src/core/embed.rs` | BERT text embeddings via Candle (BAAI/bge-small-en-v1.5) | Only batch text embed. No semantic search index, no fine-tuning. |
| MCP Server | `src/bin/rmpca-mcp-server.rs` | 15+ tools incl. `predict_solver`, `score_route`, `route_embedding` | `predict_solver` returns heuristic scores, not learned predictions. |
| VRP Solvers | `src/core/vrp/solvers/*.rs` | 5 classical heuristics (Clarke-Wright, Sweep, 2-Opt, Or-Opt, Default) | No neural solver, no learned local search guidance. |
| Elevation | `src/core/elevation/` | DEM GeoTIFF queries, fuel estimation | No ML-based terrain-aware weighting. |

### Architecture Gaps

1. **No learned solver selector** — `predict_solver` uses hand-tuned weights (density > 5.0 → +0.15, etc.). No data-driven training.
2. **No instance-aware AutoML** — solver hyperparameters are hard-coded. No per-instance tuning.
3. **No graph embeddings for road networks** — road segments are treated as raw edges. No learned representations.
4. **No route quality prediction** — quality is only scored *after* solving. No early-stopping or solver-spawning guidance.
5. **No NLP query parser** — the `agent` command takes JSON only. No natural-language routing requests.
6. **No learned local search guidance** — 2-Opt/Or-Opt use random/exhaustive moves. No neural move selector.

---

## Proposed Architecture

### Module Map

```
src/core/
├── ml/                          # NEW: ML module directory
│   ├── mod.rs                   # Re-exports, shared types
│   ├── selector.rs              # Neural/ensemble solver selector
│   ├── features.rs              # Instance feature extraction (research-backed)
│   ├── quality_predictor.rs     # Route quality prediction before solve
│   ├── automl.rs                # Instance-aware hyperparameter regression
│   └── graph_embed.rs           # Road network graph embeddings
├── embed.rs                     # EXISTING: text embeddings (BERT)
├── nlp.rs                       # NEW: NL query → VRP config parser
├── vrp/                         # EXISTING
│   ├── solvers/
│   │   ├── neural_guided.rs     # NEW: neural-guided local search wrapper
│   │   └── ...
│   └── ...
└── ...
```

### 1. Neural Solver Selector (`ml::selector`)

**Replaces**: `src/core/ml.rs::predict_solver()`  
**Research basis**: Instance-Aware Parameter Configuration (2605.00572, 2026) + RouteFinder (2406.15007, 2024)

**Design**:
- **Feature extraction**: 28-dim instance feature vector (see `ml::features`).
- **Model**: Small MLP (2 hidden layers, 128→64→5) implemented in Candle.
- **Output**: Softmax over 5 solver IDs + confidence.
- **Training**: Offline on CVRPLIB + synthetic instances. Labels = best solver in fixed time budget.
- **Inference**: ~0.1ms per instance (pure Rust/Candle).

**Feature vector (28-dim)**:
```
# Geometric (8)
n_stops_norm, n_vehicles_norm, avg_pairwise_km_norm, lat_spread_norm,
lon_spread_norm, density_norm, area_km2_norm, depot_centroid_dist_norm

# Graph (8) — from kNN(k=10) graph of stops
knn_avg_degree, knn_max_degree, knn_clustering, knn_diameter_norm,
knn_mst_weight_norm, knn_avg_shortest_path, knn_spectral_gap, knn_assortativity

# Demand (4)
total_demand_norm, demand_std_norm, tight_capacity_flag, capacity_ratio

# Distance matrix stats (4)
dist_mean_norm, dist_std_norm, dist_skewness, depot_dist_mean_norm

# Objective & interaction (4)
objective_onehot[4], stops_per_vehicle_norm, spread_x_demand_interaction
```

### 2. Instance-Aware AutoML (`ml::automl`)

**Replaces**: hard-coded solver hyperparameters  
**Research basis**: Instance-Aware Parameter Configuration (2605.00572)

**Design**:
- **Problem**: Predict best hyperparameters (e.g., `max_iterations`, `temperature` for SA, `tabu_tenure`) given instance features.
- **Model**: Separate Ridge regression per hyperparameter, or a single multi-output MLP.
- **Training**: Run `irace`-style offline tuning on a diverse instance set, record best configs, train regressor.
- **Usage**: Before solving, extract features → predict hyperparameters → inject into solver.

### 3. Route Quality Predictor (`ml::quality_predictor`)

**New capability**  
**Research basis**: RouteFinder encoder + gap prediction

**Design**:
- **Input**: Instance features (same 28-dim vector).
- **Output**: Predicted gap to optimal (%) and expected tour length (km).
- **Model**: Small MLP (28→64→32→2) in Candle.
- **Usage**:
  - Early stopping: if predicted gap < threshold, skip expensive local search.
  - Solver spawning: if predicted quality of default solver is poor, automatically spawn Or-Opt.

### 4. Graph Embeddings for Road Networks (`ml::graph_embed`)

**New capability**  
**Research basis**: GAIN (2107.07791) + RRNCO (2503.16159)

**Design**:
- **Representation**: Line-graph of OSM road network (roads → nodes, junctions → edges).
- **Features per road segment**: length, speed limit, road class one-hot, geometry sinuosity, elevation variance.
- **Model**: 2-layer GraphSAGE-MEAN (Candle) or pre-trained PyTorch → ONNX → Candle.
- **Usage**: Fused into distance/duration matrix for VRP. Edge weights become `haversine * (1 + learned_factor)`.

### 5. NLP Query Parser (`nlp`)

**New capability**  
**Research basis**: From Words to Routes (2403.10795)

**Design**:
- **Approach**: Small fine-tuned LLM (Qwen2.5-1.5B or similar) served via Candle, or rule-based NLU + entity extraction.
- **Input**: "Route 50 packages with 5 vans starting at 45.5,-73.6 by 5pm"
- **Output**: JSON VRP config (`{"num_stops":50,"num_vehicles":5,"depot":[45.5,-73.6],"deadline":"17:00"}`).
- **Training**: Synthetic SFT dataset (template utterances → JSON).
- **Integration**: New `agent` task type `NlpParse`, new MCP tool `parse_routing_query`.

### 6. Neural-Guided Local Search (`vrp::solvers::neural_guided`)

**New capability**  
**Research basis**: RLOR (2303.13117) + Analytics & ML in VRP survey (2102.10012)

**Design**:
- **Wrapper**: Wraps existing 2-Opt / Or-Opt solvers with a neural move selector.
- **Move representation**: For each candidate edge swap, encode the local subgraph (4 nodes + edges) as a feature vector.
- **Model**: Tiny MLP that scores each candidate move. Pick top-k instead of exhaustive/random.
- **Training**: REINFORCE / PPO on synthetic instances where reward = improvement in tour length.

---

## Data Flow

```
User Input
    ├── NLP Query ──→ nlp::parse() ──→ VRP Config JSON
    │
    └── JSON Task ──→ agent::run()
                        │
                        ▼
            Extract ──→ Clean ──→ Compile ──→ Optimize
                │                           │
                ▼                           ▼
        ml::graph_embed()           ml::features::extract()
                │                           │
                ▼                           ▼
        Edge embeddings           selector::predict()
                                        │
                                        ▼
                              quality_predictor::predict()
                                        │
                                        ▼
                              automl::tune_hyperparams()
                                        │
                                        ▼
                              Spawn best solver + best params
                                        │
                                        ▼
                              score_route() → feedback loop
```

---

## MCP Server Integration

### New MCP Tools

| Tool | Description | Model |
|------|-------------|-------|
| `predict_solver` | **Upgraded** — learned ensemble selector | MLP (Candle) |
| `predict_quality` | Predict expected gap / tour length before solving | MLP (Candle) |
| `tune_hyperparams` | Predict best hyperparameters for this instance | Ridge/MLP |
| `graph_embed_network` | Embed a road network (.rmp or GeoJSON) | GraphSAGE |
| `parse_routing_query` | NL → structured VRP config | Small LLM / NLU |
| `neural_vrp_solve` | Solve VRP with neural constructive heuristic | RouteFinder-style |

### Existing Tool Upgrades

| Tool | Change |
|------|--------|
| `predict_solver` | Replace rule-based with learned model; add `use_model` flag |
| `score_route` | Add predicted-vs-actual gap for feedback logging |
| `route_embedding` | Expand from 12-dim to 28-dim research-backed features |

---

## Training & Model Lifecycle

### Offline Training (Python → Rust)

```
1. Generate / download training data
   - CVRPLIB instances (http://vrp.atd-lab.inf.puc-rio.br/)
   - Synthetic instances (Kool et al. 2019 generator)
   - OSM road networks (osmnx)

2. Label instances
   - Run all 5 solvers × 20 hyperparam configs per instance
   - Record best solver, best params, achieved gap

3. Train models (PyTorch / XGBoost)
   - Solver selector: LightGBM or small MLP
   - Quality predictor: Ridge / MLP
   - AutoML: Multi-output Ridge
   - Graph embeddings: PyTorch Geometric GraphSAGE

4. Export to Rust
   - MLP weights → Candle safetensors
   - XGBoost → JSON dump → Rust parser
   - GraphSAGE → ONNX → Candle

5. Commit models to repo (or fetch from HF Hub)
```

### Online Learning (Future)

- Log every solve: instance features → solver → params → achieved quality.
- Periodic retraining (weekly) on accumulated logs.
- A/B test new model versions against rule-based baseline.

---

## Implementation Roadmap

### Phase 1: Core ML Infrastructure (v0.5.0)
- [ ] `ml::features` — 28-dim instance feature extraction
- [ ] `ml::selector` — Candle MLP solver selector (replaces rule-based)
- [ ] `ml::quality_predictor` — Candle MLP gap predictor
- [ ] `ml::automl` — Ridge regression hyperparameter tuner
- [ ] MCP tool upgrades (`predict_solver`, `route_embedding`)

### Phase 2: Graph & NLP (v0.6.0)
- [ ] `ml::graph_embed` — GraphSAGE on line-graphs (PyTorch → ONNX → Candle)
- [ ] `nlp` — Synthetic SFT dataset + small LLM inference
- [ ] `parse_routing_query` MCP tool
- [ ] `graph_embed_network` MCP tool

### Phase 3: Neural Solvers (v0.7.0)
- [ ] `vrp::solvers::neural_guided` — Neural move selector for 2-Opt/Or-Opt
- [ ] `neural_vrp_solve` MCP tool
- [ ] Online learning loop from solve logs

---

## Dependencies

### New Rust crates
```toml
# In Cargo.toml [dependencies]
candle-core = { version = "0.10", optional = true }
candle-nn = { version = "0.10", optional = true }
# Consider adding:
# ndarray = "0.15"       # Feature engineering helpers
# linfa = "0.7"          # ML primitives (if we want Ridge in pure Rust)
```

### External tools (training only)
- Python 3.10+, PyTorch, PyTorch Geometric, XGBoost, Optuna
- `osmnx` for road network extraction
- `cvrplib` for benchmark instances

---

## Benchmarking & Validation

### Metrics
- **Solver selection accuracy**: % of instances where selected solver = best solver
- **Gap improvement**: avg. gap to BKS with learned selector vs. rule-based vs. always-default
- **Inference latency**: <1ms per instance (target)
- **Quality prediction MAE**: mean absolute error of predicted gap (%)

### Test sets
- CVRPLIB X-n instances (small/medium/large)
- Synthetic uniform & clustered instances (n=20,50,100,200)
- Real-world Montreal road network instances

---

## Security & Robustness

- All ML inference is deterministic (no randomness in production inference).
- Models are versioned; new models require explicit opt-in (`--ml-model-version v2`).
- Fallback to rule-based selector if model file is missing or inference fails.
- Input validation: clamp all features to training distribution bounds to avoid out-of-distribution crashes.

---

*Document version: 2026-05-11*  
*Research basis: RouteFinder (2406.15007), RRNCO (2503.16159), Instance-Aware Parameter Configuration (2605.00572), GAIN (2107.07791), RLOR (2303.13117), Words-to-Routes (2403.10795)*
