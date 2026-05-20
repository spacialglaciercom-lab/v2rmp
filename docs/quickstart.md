# v2rmp Quick-Start Guide

Get the MCP server running and exercise the first tool in under 60 seconds.

---

## 1. Build and run the MCP server

```bash
cargo run --bin rmpca-mcp-server --features ml,extract --profile quick
```

The server listens on **stdin** and speaks **JSON-RPC 2.0** (one line per message).  
Stderr prints startup info; stdout is the pure JSON-RPC wire.

> **Tip:** `cargo build --bin rmpca-mcp-server --release` for a faster binary, then run `target/release/rmpca-mcp-server` directly.

---

## 2. Single-shot MCP tool call (bash one-liner)

Send a JSON-RPC `tools/call` request and see the response immediately:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"haversine_distance","arguments":{"from":{"lat":48.8566,"lon":2.3522},"to":{"lat":51.5074,"lon":-0.1278}}}}' | \
  cargo run --quiet --bin rmpca-mcp-server --features ml,extract
```

Expected output:
```json
{"id":1,"jsonrpc":"2.0","result":{"content":[{"text":"{\n  \"distance_km\": 344.01,\n  ...}""","type":"text"}]}}
```

---

## 3. Run the full regression test harness

```bash
python3 tests/mcp_test_harness_v2.py
```

This:
1. Generates synthetic GeoJSON, DEM, and VRP stop data on the fly.
2. Launches the MCP server.
3. Exercises **all 20 tools** and prints PASS / FAIL / SKIP for each.
4. Writes a JSON report to `mcp_test_data/mcp_test_report.json`.

**Expected result:**
```
Passed: 18  |  Skipped: 2  |  Failed: 0
```

The 2 skipped tools require internet (`pipeline` → Overture S3, `get_valhalla_matrix` → Valhalla API).

---

## 4. Verify the ML pipeline is actually working

```bash
cargo run --bin rmpca-mcp-server --features ml,extract --profile quick &
python3 tests/verify_ml_ready.py
```

You should see `ml_ready: True` for all three neural tools:

```
=== predict_solver ===
  ml_ready  : True
  model_loaded: True
  recommended: or_opt

=== predict_quality ===
  ml_ready  : True

=== tune_hyperparams ===
  ml_ready  : True
```

If `ml_ready` is `False`, the model files are missing. See **Training Models** below.

---

## 5. Train the ML models (if you need to)

```bash
python3 augment_and_retrain_v4.py \
    --data data/training_data_extra_3k.jsonl \
    --extra-data data/extra_500.jsonl \
    --epochs 400 --patience 80 \
    --focal-gamma 2.5 \
    --out-dir models
```

Produces:
- `models/solver_selector.safetensors`
- `models/quality_predictor.safetensors`
- `models/automl.safetensors`
- `models/move_scorer.safetensors`
- `models/graph_embed.safetensors`

A CI script that validates training + promotes artifacts is also available:

```bash
./scripts/train_models_ci.sh --epochs 50
```

---

## 6. Submit feedback to improve future predictions

After running `vrp_solve`, feed the actual result quality back:

```json
{
  "stops": [{"lat": 45.76, "lon": -0.04, "demand": 0}, ...],
  "solver_id": "clarke_wright",
  "actual_gap_pct": 3.5,
  "total_distance_km": 42.0
}
```

This appends a feature-labelled row to the training log so subsequent retraining learns from real-world solves.

See `ML_PIPELINE_VALIDATION.md` for the full feedback-loop documentation.

---

## File Map for Contributors

| File | What it does |
|------|--------------|
| `src/bin/rmpca-mcp-server.rs` | MCP server over stdio — all 20+ tools live here |
| `src/core/ml/selector.rs` | Neural solver selector (Candle MLP) |
| `src/core/ml/quality_predictor.rs` | Predicts gap/tour length before solving |
| `src/core/ml/automl.rs` | Instance-aware hyperparameter tuner |
| `src/core/ml/feedback.rs` | Logging solve results for online learning |
| `src/core/ml/features.rs` | 28-dim feature extraction from VRP instances |
| `augment_and_retrain_v4.py` | Full training pipeline (PyTorch → safetensors) |
| `tests/mcp_test_harness_v2.py` | Regression harness for every MCP tool |
| `tests/verify_ml_ready.py` | Smoke test for ML model readiness |
| `tests/synthetic_data.py` | Reusable data generators (urban, rural, mountain, DEM) |
| `models/*.safetensors` | Pre-trained Candle model weights |

---

## Troubleshooting

### `ML feature is not enabled`
Build with `--features ml`:
```bash
cargo run --bin rmpca-mcp-server --features ml,extract
```

### Models not found
The binary looks for `models/*.safetensors` relative to:
1. The current working directory
2. The executable directory
3. Falls back to hard-coded path

Copy models next to the binary or run from repo root.

### `pipeline` or `get_valhalla_matrix` hang
These tools require internet. They return `SKIP` gracefully in the test harness. In production they need outbound HTTPS.

### DEM errors (elevation tools)
The test harness generates synthetic DEMs via GDAL VRT. If GDAL is unavailable, install `python3-gdal` or skip elevation tests.
