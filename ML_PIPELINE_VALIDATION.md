# ML Pipeline End-to-End Validation Guide

This document describes how to verify that the neural models (solver selector, quality predictor, hyperparameter tuner) are actually loaded and used — rather than silently falling back to heuristics.

## Problem

Before this fix, `predict_solver`, `predict_quality`, and `tune_hyperparams` all had silent fallback paths:
- If the `.safetensors` model file was missing, they returned **rule-based** or **heuristic** results.
- The caller (human or agent) had no way to know whether a neural model was actually used.

## Solution: `ml_ready` field

Each of the three ML MCP tools now returns an explicit `ml_ready` boolean:

| Tool | Before output | After output |
|------|---------------|--------------|
| `predict_solver` | `recommended`, `confidence`, `all_scores` | **+ `ml_ready`** |
| `predict_quality` | `predicted_gap_pct`, `predicted_tour_length_km` | **+ `ml_ready`** |
| `tune_hyperparams` | `max_iterations`, `temperature`, `tabu_tenure` | **+ `ml_ready`** |

Examples:

```json
// predict_solver → ml_ready=true (model loaded)
{
  "recommended": "or_opt",
  "confidence": 0.34,
  "ml_ready": true,
  "model_loaded": true
}

// predict_quality → ml_ready=true
{
  "predicted_gap_pct": 1.23,
  "predicted_tour_length_km": 45.6,
  "confidence": 0.8,
  "ml_ready": true
}

// tune_hyperparams → ml_ready=true
{
  "max_iterations": 10000,
  "temperature": 50.5,
  "tabu_tenure": 13,
  "cooling_rate": 0.92,
  "neighbourhood_radius": 5,
  "ml_ready": true
}
```

When models are absent, `ml_ready` is `false` and the same tools still work via rule-based heuristics — but now the agent **knows** it did not use learned intelligence.

---

## Rust implementation changes

### 1. `NeuralPrediction` struct (`src/core/ml/selector.rs`)

Added `model_used: bool` field: the neural code sets it to `true` when `NeuralSelector::predict()` succeeds, and `predict_solver()` sets it to `false` on the rule-based fallback path.

### 2. MCP server output (`src/bin/rmpca-mcp-server.rs`)

For each ML tool handler:
- `handle_predict_solver` → `"ml_ready": pred.model_used`
- `handle_predict_quality` → `"ml_ready": pred.model_used`
- `handle_tune_hyperparams` → `"ml_ready": params.model_used`

### 3. `SolverHyperparams` (`src/core/vrp/types.rs`)

Already had `model_used: bool`. The fallback constructor `default_fallback()` sets it to `false`.

---

## How to train models

### Quick smoke-test (reduced epochs)

```bash
cd /root/v2rmp

python3 augment_and_retrain_v4.py \
    --data data/training_data_extra_3k.jsonl \
    --extra-data data/extra_500.jsonl \
    --epochs 50 --patience 20 \
    --out-dir models/test_run
```

### Production training (full recipe)

```bash
python3 augment_and_retrain_v4.py \
    --data data/training_data_extra_3k.jsonl \
    --extra-data data/extra_500.jsonl \
    --epochs 400 --patience 80 \
    --focal-gamma 2.5 --label-smooth 0.08 \
    --batch 256 \
    --out-dir models
```

This produces 5 artifacts:

| File | Purpose | Expected size |
|------|---------|---------------|
| `models/solver_selector.safetensors` | Chooses best VRP solver | ~50 KB |
| `models/quality_predictor.safetensors` | Predicts gap to optimal | ~16 KB |
| `models/automl.safetensors` | Tunes SA/tab hyperparameters | ~9 KB |
| `models/move_scorer.safetensors` | Scores 2-opt moves | ~5 KB |
| `models/graph_embed.safetensors` | GraphSAGE road-network embeddings | ~39 KB |

### CI / automation

A reproducible shell script is provided:

```bash
./scripts/train_models_ci.sh \
    --epochs 50 \
    --data data/training_data_extra_3k.jsonl \
    --out-dir models/ci_run_20260512
```

It:
1. Checks Python/Rust dependencies
2. Runs the full training pipeline
3. Validates all 5 model artifacts exist and have non-zero size
4. Builds `rmpca-mcp-server` with `ml` feature enabled
5. Runs `tests/verify_ml_ready.py` to confirm `ml_ready: true`
6. Promotes models to `models/` only if all checks pass

---

## How to verify models are loaded

### Option 1: Script

```bash
cargo build --bin rmpca-mcp-server --profile quick --features ml,extract
python3 tests/verify_ml_ready.py
```

Expected output:
```
=== predict_solver ===
  ml_ready  : True
  model_loaded: True
  recommended: or_opt

=== predict_quality ===
  ml_ready  : True
  confidence: 0.8

=== tune_hyperparams ===
  ml_ready  : True
  max_iter  : 10000

✅ All ml_ready fields present and correct.
```

### Option 2: Manual JSON-RPC

```bash
echo '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}'   | cargo run --bin rmpca-mcp-server --features ml,extract --quiet
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"predict_solver","arguments":{"stops":[{"lat":45.76,"lon":-0.04,"demand":0},{"lat":45.761,"lon":-0.039,"demand":1}]}}}' | cargo run --bin rmpca-mcp-server --features ml,extract --quiet
```

Look for `"ml_ready":true` in the output.

### Option 3: Full MCP test harness

```bash
python3 tests/mcp_test_harness_v2.py
```

All ML tests should show `PASS` (not `SKIP` or `FAIL`).

---

## File map

| File | Role |
|------|------|
| `src/core/ml/selector.rs` | Neural solver selector + `model_used` flag on `NeuralPrediction` |
| `src/core/ml/quality_predictor.rs` | Quality predictor (already had `model_used` since earlier work) |
| `src/core/ml/automl.rs` | Hyperparam predictor (already had `model_used` since earlier work) |
| `src/core/vrp/types.rs` | `SolverHyperparams.model_used` field |
| `src/bin/rmpca-mcp-server.rs` | MCP handlers that expose `ml_ready` in JSON output |
| `tests/verify_ml_ready.py` | Standalone smoke-test script |
| `tests/mcp_test_harness_v2.py` | Full test harness covering 18+ tools |
| `scripts/train_models_ci.sh` | Reproducible training + validation + promotion shell script |
| `ML_TRAINING_IMPLEMENTATION_REPORT.md` | Background on focal loss, near-win labeling, SMOTE |
| `augment_and_retrain_v4.py` | The actual PyTorch → safetensors training script |

---

## Model discovery paths

The Rust binary looks for models in this order:

1. `models/*.safetensors` relative to the **current executable** directory
2. `models/*.safetensors` relative to the **current working directory**
3. Hard-coded fallback path

If you build with `cargo build --release`, place models in `target/release/models/`.
If you copy the binary elsewhere, ship the `models/` directory alongside it.
