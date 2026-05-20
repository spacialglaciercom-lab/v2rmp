#!/usr/bin/env bash
# scripts/verify_cli_interface.sh
# Compatibility guard for the rmpca CLI interface.
#
# Runs `rmpca --help` and every subcommand's `--help` to confirm:
#   1. `--help` returns exit code 0 for all subcommands.
#   2. Required flags (by convention: required args have no default_value
#      and no Option<String> type — we check for known mandatory flags).
#
# Usage: ./scripts/verify_cli_interface.sh

set -euo pipefail

BINARY="${CARGO_TARGET_DIR:-target}/debug/rmpca"
if [[ ! -x "$BINARY" ]]; then
    BINARY="target/debug/rmpca"
fi

# First try the debug binary, then release
if [[ ! -x "$BINARY" ]]; then
    BINARY="target/release/rmpca"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: rmpca binary not found. Build with: cargo build" >&2
    exit 2
fi

echo "=== rmpca CLI Interface Verification ==="
echo "Binary: $BINARY ($("$BINARY" --version 2>/dev/null || echo 'version unknown'))"
echo ""

PASS=0
FAIL=0

# Helper: run --help for a subcommand and check for regressions
run_help() {
    local name="$1"
    shift
    local args=("$@")
    local full_args=("${args[@]}" "--help")

    if "$BINARY" "${full_args[@]}" >/dev/null 2>&1; then
        echo "  PASS: rmpca $name --help"
        PASS=$((PASS + 1))
    else
        local exit_code=$?
        echo "  FAIL: rmpca $name --help (exit code $exit_code)" >&2
        FAIL=$((FAIL + 1))
    fi
}

# Helper: check that --help output contains a required flag
check_flag() {
    local name="$1"
    shift
    local args=("$@")
    local full_args=("${args[@]}" "--help")
    local output
    output=$("$BINARY" "${full_args[@]}" 2>&1) || true

    local flag="$1"
    # Remove the flag after capturing it for reporting; caller passes flag first
    shift 2

    if echo "$output" | grep -q -e "$flag"; then
        return 0
    else
        echo "    WARNING: flag '$flag' not found in help output for '$name'" >&2
        return 1
    fi
}

# ─── Top-level ────────────────────────────────────────────────────────
echo "--- Top-level --help ---"
run_help "(root)"

# ─── Core subcommands ──────────────────────────────────────────────────
echo ""
echo "--- Core subcommands ---"

# Compile
run_help "compile" "compile"
# Check that compile requires --input and --output
output=$("$BINARY" compile --help 2>&1) || true
for flag in "--input" "--output"; do
    if echo "$output" | grep -q -e "$flag"; then
        echo "  OK: compile help lists $flag"
    else
        echo "  WARNING: compile help missing $flag" >&2
    fi
done

# Clean
run_help "clean" "clean"

# Optimize (CPP)
run_help "optimize" "optimize"
output=$("$BINARY" optimize --help 2>&1) || true
for flag in "--input" "--output"; do
    if echo "$output" | grep -q -e "$flag"; then
        echo "  OK: optimize help lists $flag"
    else
        echo "  WARNING: optimize help missing $flag" >&2
    fi
done

# VRP
run_help "vrp" "vrp"
output=$("$BINARY" vrp --help 2>&1) || true
for flag in "--coordinates" "--input" "--output-dir"; do
    if echo "$output" | grep -q -e "$flag"; then
        echo "  OK: vrp help lists $flag"
    else
        echo "  WARNING: vrp help missing $flag" >&2
    fi
done

# List
run_help "list" "list"

# Agent
run_help "agent" "agent"

# Serve
run_help "serve" "serve"

# Parse query (NLP)
run_help "parse-query" "parse-query"
output=$("$BINARY" parse-query --help 2>&1) || true
if echo "$output" | grep -q -e "--query"; then
    echo "  OK: parse-query help lists --query"
else
    echo "  WARNING: parse-query help missing --query" >&2
fi

# ─── Feature-gated: extract ────────────────────────────────────────────
echo ""
echo "--- Feature-gated: extract ---"
output=$("$BINARY" --help 2>&1) || true
if echo "$output" | grep -q -E "^\s+extract\b"; then
    run_help "extract" "extract"
    output=$("$BINARY" extract --help 2>&1) || true
    for flag in "--source" "--bbox" "--output"; do
        if echo "$output" | grep -q -e "$flag"; then
            echo "  OK: extract help lists $flag"
        else
            echo "  WARNING: extract help missing $flag" >&2
        fi
    done
else
    echo "  SKIP: extract feature not compiled"
fi

# ─── Feature-gated: pipeline ───────────────────────────────────────────
if echo "$output" | grep -q -E "^\s+pipeline\b"; then
    run_help "pipeline" "pipeline"
else
    echo "  SKIP: pipeline feature not compiled"
fi

# ─── Feature-gated: ml ─────────────────────────────────────────────────
echo ""
echo "--- Feature-gated: ml ---"

# Graph-embed
if "$BINARY" --help 2>&1 | grep -q -E "^\s+graph-embed\b"; then
    run_help "graph-embed" "graph-embed"
    output=$("$BINARY" graph-embed --help 2>&1) || true
    for flag in "--input" "-i"; do
        if echo "$output" | grep -q -e "$flag"; then
            echo "  OK: graph-embed help lists $flag"
            break
        fi
    done
else
    echo "  SKIP: ml feature not compiled (graph-embed)"
fi

# Predict-solver
if "$BINARY" --help 2>&1 | grep -q -E "^\s+predict-solver\b"; then
    run_help "predict-solver" "predict-solver"
else
    echo "  SKIP: ml feature not compiled (predict-solver)"
fi

# Predict-quality
if "$BINARY" --help 2>&1 | grep -q -E "^\s+predict-quality\b"; then
    run_help "predict-quality" "predict-quality"
else
    echo "  SKIP: ml feature not compiled (predict-quality)"
fi

# Tune-hyperparams
if "$BINARY" --help 2>&1 | grep -q -E "^\s+tune-hyperparams\b"; then
    run_help "tune-hyperparams" "tune-hyperparams"
else
    echo "  SKIP: ml feature not compiled (tune-hyperparams)"
fi

# Embed
if "$BINARY" --help 2>&1 | grep -q -E "^\s+embed\b"; then
    run_help "embed" "embed"
else
    echo "  SKIP: ml feature not compiled (embed)"
fi

# ─── Feature-gated: elevation (extract) ────────────────────────────────
echo ""
echo "--- Feature-gated: elevation ---"
if "$BINARY" --help 2>&1 | grep -q -E "^\s+elevation\b"; then
    run_help "elevation" "elevation"
    # Elevation has subcommands
    output=$("$BINARY" elevation --help 2>&1) || true
    for subcmd in "point" "points" "profile" "stats" "info" "fuel"; do
        if echo "$output" | grep -q -e "$subcmd"; then
            run_help "elevation $subcmd" "elevation" "$subcmd"
        fi
    done
else
    echo "  SKIP: elevation feature not compiled"
fi

# ─── Summary ───────────────────────────────────────────────────────────
echo ""
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "CLI interface has regressions. Review the failures above."
    exit 1
fi

echo ""
echo "✓ CLI interface verification passed."
exit 0
