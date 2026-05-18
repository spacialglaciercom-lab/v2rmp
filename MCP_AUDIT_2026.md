# MCP Server & Tools Audit — 2026-05-17

**Auditor:** ML Intern (automated)  
**Project:** v2rmp v0.5.3 — Route Optimization TUI & Agent Engine  
**Scope:** All 3 MCP server binaries, config, tool schemas, protocol compliance, security, and HF deployability

---

## Executive Summary

The project has **3 MCP server binaries** implementing JSON-RPC 2.0 over stdio:

| Binary | Source | Tools | Status |
|--------|--------|-------|--------|
| `rmpca-mcp-server-legacy` | `src/bin/rmpca-mcp-server.rs` | 10 | ⚠️ Legacy — **active in mcp_config.json** |
| `rmpca-mcp` | `src/bin/mcp-server.rs` | 20 | ✅ Current — **NOT wired in config** |
| `zilliz-mcp-server` | `src/bin/zilliz-mcp-server.rs` | 1 | ✅ Separate codebase search server |

**Critical finding:** `mcp_config.json` points to the **legacy** binary (10 tools, less safe, fewer features). The newer server (20 tools, better error handling, AutoML, structured results) exists but isn't configured.

---

## 1. Configuration Issues

### 1.1 🔴 Config Points to Legacy Binary
`mcp_config.json` references `rmpca-mcp-server-legacy` but the project's active development is on `rmpca-mcp`. The new server has 2x the tools, better structured results, and timeout protection.

**Fix:** Update `mcp_config.json` to use `rmpca-mcp`:
```json
{
  "mcpServers": {
    "v2rmp": {
      "command": "/home/rmp/v2rmp/target/debug/rmpca-mcp",
      "args": []
    }
  }
}
```

### 1.2 🔴 Hardcoded Credentials in Config
`mcp_config.json` contains `DATABASE_URL` and `SUPABASE_DB_URL` with real credentials (even if masked as `...` in the committed version, the `.env` file has real values). The config should reference env vars or `.env` rather than duplicating secrets.

### 1.3 ⚠️ Two Servers Need Separate Config Entries
The `zilliz-mcp-server` isn't in `mcp_config.json` at all. If users want codebase search, they need a separate entry.

---

## 2. Security Issues

### 2.1 🔴 SQL Injection — `query_supabase` (Legacy Only)
The legacy server's `query_supabase` tool executes **arbitrary SQL** with zero restrictions. Any MCP client or LLM agent can run `DROP TABLE`, `DELETE`, or `UPDATE`.

**Status:** ❌ NOT fixed (legacy only — new server doesn't have this tool)

**Recommendation:** 
- Remove `query_supabase` from legacy or add `SET TRANSACTION READ ONLY` wrapper
- If DB query is needed in new server, implement as parameterized queries only

### 2.2 🟠 Path Traversal — File Path Parameters
Both servers accept file paths (e.g., `input_geojson`, `output_rmp`, `map_path`, `dem_path`) with no validation. An LLM agent could pass `../../etc/passwd` or `/tmp/malicious`.

**Affected tools (legacy):** `v2rmp_extract`, `v2rmp_compile`, `v2rmp_optimize`, `upload_to_r2`, `download_from_r2`  
**Affected tools (new):** `compile`, `optimize`, `clean`, `inspect_rmp`, `elevation_query`, `elevation_profile`, `pipeline`

**Fix:** Canonicalize paths and reject those outside allowed directories:
```rust
fn validate_path(path: &str) -> Result<PathBuf> {
    let canonical = std::fs::canonicalize(path)?;
    let allowed = std::fs::canonicalize("./")?;
    if !canonical.starts_with(&allowed) {
        anyhow::bail!("Path traversal denied");
    }
    Ok(canonical)
}
```

### 2.3 🟠 No Input Validation — BBox Coordinates
Neither server validates bounding box coordinates:
- No check that `min_lon < max_lon` or `min_lat < max_lat`
- No range validation ([-180,180] / [-90,90])
- Zero-valued bboxes silently accepted (legacy defaults to `0.0`)

### 2.4 🟠 Missing `additionalProperties: false`
**No tool in either server** sets `additionalProperties: false` on `inputSchema`. Per OWASP MCP Security Cheat Sheet and MCP spec best practices, this is required to prevent LLMs from injecting unexpected parameters.

**Impact:** An LLM could pass arbitrary extra fields that get silently ignored or could trigger unintended behavior.

### 2.5 🟡 R2 Credentials via Environment Variables
R2 access keys are passed via `env` in `mcp_config.json`. While this is the standard MCP pattern, the `.env` file containing real credentials is not in `.gitignore` (it IS in the `exclude` list in Cargo.toml but the file still exists on disk).

---

## 3. MCP Protocol Compliance

### 3.1 Protocol Version
Both servers advertise `protocolVersion: "2024-11-05"`. The latest MCP spec is **`2025-06-18`**. Key missing features:

| Feature | 2025-06-18 Requirement | Current Status |
|---------|----------------------|----------------|
| `ToolAnnotations` | `readOnlyHint`, `destructiveHint`, etc. | ❌ Not implemented |
| `title` field on tools | Human-readable display name | ❌ Not implemented |
| `outputSchema` | Structured output validation | ❌ Not implemented |
| `structuredContent` | Alternative to text-only results | ❌ Not implemented |
| `notifications/tools/list_changed` | Dynamic tool list updates | ❌ Not implemented |
| `logging/setLevel` | Client-controlled log verbosity | ❌ Not implemented |
| `notifications/progress` | Progress for long operations | ❌ Not implemented |
| Pagination for `tools/list` | Cursor-based pagination | ❌ Not needed (finite tools) |

### 3.2 Initialization State Tracking
Neither server tracks whether `initialize` has been called. A client could call `tools/call` before initialization.

**Fix:** Add `AtomicBool` flag:
```rust
static INITIALIZED: std::sync::atomic::AtomicBool = AtomicBool::new(false);
// Set on initialize, check on tools/call
```

### 3.3 JSON-RPC Validation
- **Legacy:** Does NOT validate `jsonrpc` field — silently processes any JSON
- **New:** Validates `jsonrpc: "2.0"` ✅
- **Zilliz:** Validates `jsonrpc: "2.0"` ✅

### 3.4 Proper JSON-RPC Error Codes
| Code | Meaning | Legacy | New | Zilliz |
|------|---------|--------|-----|--------|
| -32700 | Parse error | ❌ (silently skips) | ✅ | ✅ |
| -32600 | Invalid request | ❌ | ✅ | ✅ |
| -32601 | Method not found | ✅ | ✅ | ✅ |
| -32602 | Invalid params | ❌ (uses -32000) | ✅ | ✅ |
| -32000 | Server error | ✅ (all errors) | Mixed | N/A |

---

## 4. Tool-by-Tool Comparison

### Legacy Server (10 tools) — `rmpca-mcp-server-legacy`

| Tool | Quality | Issues |
|------|---------|--------|
| `list_r2_bucket` | 🟡 | New tokio runtime per call (anti-pattern); no pagination |
| `upload_to_r2` | 🟡 | Path traversal risk; new runtime per call |
| `download_from_r2` | 🟡 | Path traversal risk; new runtime per call |
| `query_supabase` | 🔴 | **SQL injection**; incomplete type handling (only 5/15+ PG types) |
| `v2rmp_extract` | 🟠 | No bbox validation; hardcoded `RoadClass::all_vehicle()`; feature-gated |
| `v2rmp_compile` | 🟠 | Only exposes `remove_isolates` from 17 CleanOptions fields |
| `v2rmp_optimize` | 🟠 | Only returns distance text; missing structured result fields; new runtime |
| `v2rmp_postgis_cpp` | ✅ | Well-documented schema; good parameter descriptions |
| `v2rmp_neural_optimize` | 🟡 | No array length validation; `.unwrap()` panics removed in new version |
| `v2rmp_generate_osmand_link` | ✅ | Simple, correct |

### New Server (20 tools) — `rmpca-mcp`

| Tool | Quality | Issues |
|------|---------|--------|
| `extract_overture` | ✅ | Good schema with descriptions; feature-gated |
| `extract_osm` | ✅ | Good schema; pbf_path optional |
| `compile` | ✅ | Clean schema; defaults documented |
| `optimize` | ✅ | Full turn penalties; Google Maps/OsmAnd links; structured result |
| `clean` | ✅ | **Excellent** — all 17 CleanOptions parameters exposed with descriptions |
| `vrp_solve` | ✅ | Full VRP solver with stops, objectives, Google Maps links |
| `elevation_query` | ✅ | Well-typed; feature-gated |
| `elevation_profile` | ✅ | Well-typed; sample interval configurable |
| `list_solvers` | ✅ | Simple utility |
| `elevation_stats` | ✅ | Feature-gated |
| `dem_info` | ✅ | Feature-gated |
| `fuel_estimate` | ✅ | Feature-gated; good physics model |
| `inspect_rmp` | ✅ | Much-needed utility |
| `predict_solver` | ✅ | ML-powered solver selection |
| `score_route` | ✅ | Multi-dimensional quality scoring |
| `route_embedding` | ✅ | 12-dim feature vector generation |
| `pipeline` | ✅ | End-to-end with 30s timeout protection |
| `haversine_distance` | ✅ | Simple utility |
| `get_valhalla_matrix` | ✅ | Real-road distances; 15s timeout |
| `predict_quality` | ✅ | Pre-solve quality prediction |
| `tune_hyperparams` | ✅ | AutoML hyperparameter tuning |
| `parse_routing_query` | ✅ | NLP → VRP config; LLM fallback |
| `submit_feedback` | ✅ | Online learning feedback loop |

### Zilliz Server (1 tool) — `zilliz-mcp-server`

| Tool | Quality | Issues |
|------|---------|--------|
| `search_codebase` | ✅ | Clean implementation; proper error handling; dimension validation |

---

## 5. Code Quality Issues

### 5.1 Legacy Server — New Tokio Runtime Per Call
Every async tool in the legacy server creates `tokio::runtime::Runtime::new()?`. This:
- Allocates a new multi-threaded runtime per call (expensive)
- Can hit OS thread limits under concurrent use
- Is a well-known Tokio anti-pattern

**Status:** ❌ Legacy only. New server uses `#[tokio::main]` ✅

### 5.2 Legacy Server — `.unwrap()` Panics (Partially Fixed)
The previous audit flagged `.unwrap()` panics in `drone_solve`. The legacy server still has risky patterns like:
```rust
.as_f64().unwrap()  // Will panic on non-numeric values
```
The new server uses `.context()` + `?` everywhere ✅

### 5.3 Monolithic Handler — Legacy
The legacy server has a 400-line `handle_tool_call` match block. The new server properly refactors into individual handler functions ✅

### 5.4 `drone_list_models` Returns Raw String
Mentioned in previous audit — still returns `"[\"FlyCart30\", \"Wing\"]"` as embedded JSON string in text field. Should return structured data.

### 5.5 Feature-Conditional Compilation
Tools gated behind `extract` feature (GDAL dependency) cleanly degrade. The new server handles this with `#[cfg(feature = "extract")]` on both tool definitions and handlers ✅

---

## 6. Testing Coverage

| Test File | Coverage |
|-----------|----------|
| `tests/mcp_test_harness.py` | Legacy server protocol tests |
| `tests/mcp_test_harness_v2.py` | Updated test harness |
| `tests/test_mcp_core.py` | Core MCP functionality |
| `tests/mcp_debug.py` | Debug utilities |
| `tests/unit_clean_compile.rs` | Clean + compile integration |
| `tests/unit_optimize.rs` | Optimize unit tests |
| `tests/dem_info_test.rs` | DEM info tests |
| `tests/neural_benchmark.rs` | Neural solver benchmarks |
| `tests/ml_inference_test.rs` | ML model inference tests |
| `tests/ml_behavior_test.rs` | ML behavior validation |
| `tests/overture_s3_integration.rs` | S3 extraction integration |

**Gap:** No automated tests for the new `rmpca-mcp` server binary. Tests only target legacy.

---

## 7. HF Spaces Deployment Readiness

The current stdio-only transport **cannot** be deployed to HF Spaces. HF requires either:
1. **Gradio with `mcp_server=True`** — auto-exposes MCP over HTTP
2. **Docker Space with SSE/Streamable HTTP** — custom HTTP transport layer

**Recommendation:** To deploy on HF Spaces:
- Wrap the Rust binary in a Python Gradio Space that calls it via subprocess
- Or implement SSE transport in Rust (using `rmcp` crate or custom)

---

## 8. Priority Matrix — Updated

| Priority | Item | Impact | Server |
|----------|------|--------|--------|
| 🔴 **P0** | Switch `mcp_config.json` to use `rmpca-mcp` (new binary) | Correctness | Config |
| 🔴 **P0** | Add `additionalProperties: false` to all tool schemas | Security (OWASP) | Both |
| 🔴 **P0** | Add path validation (canonicalize + allowlist) | Security | Both |
| 🟠 **P1** | Add bbox coordinate validation | Correctness | Both |
| 🟠 **P1** | Update protocol version to `2025-06-18` | Compliance | Both |
| 🟠 **P1** | Add `ToolAnnotations` (`readOnlyHint`, `destructiveHint`) | Compliance | New |
| 🟠 **P1** | Add `outputSchema` to tools with structured results | Compliance | New |
| 🟠 **P1** | Add `title` field to tool definitions | Compliance | New |
| 🟠 **P1** | Add initialization state tracking (`AtomicBool`) | Robustness | Both |
| 🟠 **P1** | Remove `query_supabase` from legacy or make read-only | Security | Legacy |
| 🟠 **P1** | Complete Postgres type handling (FLOAT, BOOL, JSON, UUID, etc.) | Correctness | Legacy |
| 🟠 **P1** | Add tests for new `rmpca-mcp` server | Quality | Tests |
| 🟡 **P2** | Add `notifications/progress` for long operations | UX | New |
| 🟡 **P2** | Add `logging/setLevel` support | Debugging | Both |
| 🟡 **P2** | Move credentials from config to `.env`-only | Security | Config |
| 🟡 **P2** | Implement HF Spaces deployment (SSE transport or Gradio wrapper) | Deployment | New |
| 🟡 **P2** | Add Zilliz server to `mcp_config.json` | Usability | Config |
| 🟢 **P3** | Add `structuredContent` to tool results | Compliance | New |
| 🟢 **P3** | Consider `rmcp` Rust SDK for future development | Maintainability | New |
| 🟢 **P3** | Add `r2_delete` and `r2_object_info` tools | Completeness | New |
| 🟢 **P3** | Deprecate/retire legacy server | Maintenance | Legacy |

---

## 9. Quick Wins (Can be done immediately)

1. **Switch config to new binary** — edit `mcp_config.json` to use `rmpca-mcp`
2. **Add `additionalProperties: false`** — single-line addition to each tool schema
3. **Add bbox validation** — 4 lines of bounds checking
4. **Add `title` fields** — cosmetic but helps LLM agent tool selection

## 10. Architecture Recommendation

Consolidate to a **single server binary** (`rmpca-mcp`) and retire the legacy server. The new server is strictly superior:
- 20 tools vs 10
- Individual handler functions vs monolithic match
- `#[tokio::main]` vs runtime-per-call
- Proper error propagation vs `.unwrap()` panics
- Structured results vs text-only
- Timeout protection on network operations
- AutoML integration

If R2/Supabase operations are needed, port them to the new server with proper security controls.
