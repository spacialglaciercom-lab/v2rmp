#!/usr/bin/env bash
# =============================================================================
# v2rmp ML Pipeline CI / Reproducible Training Script
# =============================================================================
# Usage:
#   ./scripts/train_models_ci.sh [--epochs N] [--data PATH] [--out-dir PATH]
#
# This script:
#   1. Validates training data exists and has expected format
#   2. Runs the full augment_and_retrain_v4.py pipeline
#   3. Verifies model artifacts are produced
#   4. Runs a smoke test via the MCP server to confirm ml_ready=True
#   5. Copies validated models to models/ only if all checks pass
#
# Exit codes:
#   0 = all models trained and validated successfully
#   1 = missing dependencies
#   2 = training failed
#   3 = artifact validation failed
#   4 = smoke test failed
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DATA="${REPO_ROOT}/data/training_data_extra_3k.jsonl"
EXTRA_DATA="${REPO_ROOT}/data/extra_500.jsonl"
DEFAULT_OUT="${REPO_ROOT}/models/ci_run_$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "v2rmp ML Pipeline — Reproducible Training"
echo "================================================================"
echo "Repo root: ${REPO_ROOT}"
echo "Date:      $(date)"
echo ""

# --- parse args ---------------------------------------------------------------
EPOCHS=50
PATIENCE=20
FOCAL_GAMMA=2.0
LABEL_SMOOTH=0.06
BATCH=256
DATA="${DEFAULT_DATA}"
EXTRA="${EXTRA_DATA}"
OUT_DIR="${DEFAULT_OUT}"
MIN_PER_CLASS=150

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2"; shift 2;;
        --patience)   PATIENCE="$2"; shift 2;;
        --focal-gamma) FOCAL_GAMMA="$2"; shift 2;;
        --label-smooth) LABEL_SMOOTH="$2"; shift 2;;
        --batch)      BATCH="$2"; shift 2;;
        --data)       DATA="$2"; shift 2;;
        --extra-data) EXTRA="$2"; shift 2;;
        --out-dir)    OUT_DIR="$2"; shift 2;;
        --min-per-class) MIN_PER_CLASS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# --- dependency checks --------------------------------------------------------
echo "[CHECK] Python dependencies ..."
for pkg in numpy torch safetensors scipy; do
    python3 -c "import ${pkg}" 2>/dev/null || { echo "ERROR: Python package '${pkg}' missing"; exit 1; }
done
echo "  OK"

echo "[CHECK] Rust toolchain ..."
cargo --version >/dev/null 2>&1 || { echo "ERROR: cargo not found"; exit 1; }
echo "  OK"

echo "[CHECK] Training data ..."
if [[ ! -f "$DATA" ]]; then
    echo "ERROR: Training data not found: $DATA"
    exit 1
fi
DATA_LINES=$(wc -l < "$DATA" | tr -d ' ')
echo "  OK  ($DATA_LINES rows in $DATA)"

mkdir -p "$OUT_DIR"

# --- training -----------------------------------------------------------------
echo ""
echo "[TRAIN] Running augment_and_retrain_v4.py ..."
python3 "${REPO_ROOT}/augment_and_retrain_v4.py" \
    --data "$DATA" \
    --extra-data "$EXTRA" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --focal-gamma "$FOCAL_GAMMA" \
    --label-smooth "$LABEL_SMOOTH" \
    --batch "$BATCH" \
    --min-per-class "$MIN_PER_CLASS" \
    --out-dir "$OUT_DIR"

# --- artifact validation -------------------------------------------------------
echo ""
echo "[VALIDATE] Checking model artifacts ..."
REQUIRED_FILES=(
    solver_selector.safetensors
    quality_predictor.safetensors
    automl.safetensors
    move_scorer.safetensors
    graph_embed.safetensors
)
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$OUT_DIR/$f" ]]; then
        echo "ERROR: Missing artifact: $OUT_DIR/$f"
        exit 3
    fi
    SIZE=$(stat -c%s "$OUT_DIR/$f" 2>/dev/null || stat -f%z "$OUT_DIR/$f" 2>/dev/null)
    echo "  OK  $f ($SIZE bytes)"
done

# --- smoke test via MCP server -----------------------------------------------
echo ""
echo "[SMOKE] Building MCP server with ML feature ..."
cargo build --bin rmpca-mcp-server --profile quick --features ml,extract >/tmp/ci_build.log 2>&1 || {
    echo "ERROR: MCP server build failed. See /tmp/ci_build.log"
    exit 4
}

echo "[SMOKE] Running MCP ml_ready check ..."
python3 "${REPO_ROOT}/tests/verify_ml_ready.py" || {
    echo "ERROR: Smoke test (ml_ready) failed."
    exit 4
}

# --- promote to production models/ -------------------------------------------
echo ""
echo "[PROMOTE] Copying validated models to models/ ..."
cp "$OUT_DIR"/*.safetensors "${REPO_ROOT}/models/"
echo "  OK"

# --- write report ------------------------------------------------------------
REPORT="${OUT_DIR}/training_report.md"
cat > "$REPORT" <<EOF
# ML Training Report

| Field | Value |
|-------|-------|
| Timestamp | $(date -Is) |
| Data | $DATA ($DATA_LINES rows) |
| Extra data | $EXTRA |
| Epochs | $EPOCHS |
| Patience | $PATIENCE |
| Focal gamma | $FOCAL_GAMMA |
| Label smooth | $LABEL_SMOOTH |
| Batch size | $BATCH |
| Min per class | $MIN_PER_CLASS |
| Output dir | $OUT_DIR |

## Artifacts

| File | Size (bytes) |
|------|-------------|
$(for f in "${REQUIRED_FILES[@]}"; do echo "| $f | $(stat -c%s "$OUT_DIR/$f" 2>/dev/null || stat -f%z "$OUT_DIR/$f" 2>/dev/null) |"; done)

## Result

All artifacts produced and smoke-tested successfully. Models promoted to \`models/\`.
EOF

echo ""
echo "================================================================"
echo "SUCCESS — Models trained, validated, and promoted."
echo "Report:  $REPORT"
echo "Models:  models/*.safetensors"
echo "================================================================"
exit 0
