# MCP Server Audit — `rmpca-mcp`

**Date:** 2026-05-14  
**File:** `src/bin/mcp-server.rs`  
**Current tools:** 14  

---

## 1. Critical Issues

### 1.1 SQL Injection / Unsafe `query_supabase` Tool
The `query_supabase` tool executes **arbitrary SQL** with no restrictions. Any MCP client (or LLM agent) can run `DROP TABLE`, `DELETE`, or `UPDATE` with no guardrails.

**Fix options:**
- Add a `read_only` boolean that wraps the query in a read-only transaction (`SET TRANSACTION READ ONLY`)
- Add an `allow_writes` server-side config flag (env var)
- Parse the SQL and reject DDL/DML unless explicitly allowed
- At minimum, document the risk in the tool description

### 1.2 `query_supabase` — Incomplete Type Handling
Only 5 Postgres types are handled: `TEXT`, `VARCHAR`, `NAME`, `INT4`, `INT8`. Missing:

| Type | Example |
|------|---------|
| `FLOAT4` / `FLOAT8` | PostGIS geometry columns, measurements |
| `NUMERIC` | Precise decimals |
| `BOOL` | Flags |
| `JSON` / `JSONB` | Properties, config |
| `TIMESTAMP` / `TIMESTAMPTZ` | Created_at, updated_at |
| `DATE` | Date columns |
| `UUID` | Primary keys |
| `INT2` (SMALLINT) | Counts, enums |

These all render as `<type not displayed>` — silently losing data.

### 1.3 New Tokio Runtime per Tool Call
Every async tool call (`list_r2_bucket`, `upload_to_r2`, `download_from_r2`, `query_supabase`) creates `tokio::runtime::Runtime::new()?`. This:
- Allocates a new multi-threaded runtime each call (expensive)
- Can hit OS thread limits under concurrent use
- Is a known anti-pattern

**Fix:** Create the runtime once in `main()` and pass it through, or use `tokio::main` + a global handle.

### 1.4 Unchecked `.unwrap()` Calls Will Panic
In `drone_solve`, depot and customer parsing uses `.unwrap()` on array indexing:
```rust
let depot = [depot_raw[0].as_f64().unwrap(), depot_raw[1].as_f64().unwrap()];
```
A malformed input (non-numeric values, wrong array length) will **crash the server process**.

**Fix:** Use `.context()` + `?` like other tools do.

### 1.5 No Input Validation
- **BBox:** No check that `min_lon < max_lon`, `min_lat < max_lat`, or that coordinates are in [-180,180]/[-90,90]
- **File paths:** No path traversal protection (`../../etc/passwd` is valid)
- **Array lengths:** No check that `demands` length matches `locations` length in neural tools
- **Model paths:** No validation that the `.onnx` file exists before loading

---

## 2. Protocol Compliance Issues

### 2.1 No Initialization State Tracking
The server doesn't track whether `initialize` has been called. An MCP client could call `tools/call` before initialization.

**Fix:** Add a `std::sync::atomic::AtomicBool` flag set on `initialize`, reject other methods if not initialized.

### 2.2 Missing `cancelled` Notification Support
Long-running operations (extract, compile, optimize, neural inference) have no cancellation support. MCP clients expect to be able to cancel in-flight operations.

### 2.3 No Progress Notifications
The MCP spec supports `notifications/progress` for long-running operations. The extract → compile → optimize pipeline can take minutes. There's no progress reporting.

### 2.4 Missing `logging/setLevel` Support
The MCP spec allows clients to set logging verbosity. The server ignores this entirely.

### 2.5 Tool Schema — Missing `additionalProperties: false`
All tool `inputSchema` should include `"additionalProperties": false` to prevent AI agents from passing unexpected arguments.

---

## 3. Missing Core Parameters in Existing Tools

### 3.1 `v2rmp_extract` — Missing Parameters
The core `ExtractRequest` supports `road_classes` but the MCP tool hardcodes `RoadClass::all_vehicle()` with no override option.

**Add:** `road_classes` array parameter.

### 3.2 `v2rmp_compile` — Missing Clean Options
`CleanOptions` has 17 fields. The MCP tool only exposes `remove_isolates`. Missing:
- `min_length_m` — minimum edge length
- `node_snap_m` — node merging distance
- `merge_parallel_edges` — dedup parallel roads
- `simplify_tolerance_m` — Douglas-Peucker simplification
- `max_components` — keep N largest connected components

### 3.3 `v2rmp_optimize` — Missing Parameters
- No `left_turn_penalty` or `right_turn_penalty` (only `u_turn_penalty` is exposed)
- No `oneway_mode` (hardcoded to `Respect`)
- Result only returns total distance — missing `total_segments`, `deadhead_distance_km`, `efficiency_pct`, `turns`, `elapsed_ms`

### 3.4 `drone_solve` — Missing `no_fly_zones`
`DroneVrpInstance` has a `no_fly_zones` field but the MCP tool always sets it to `vec![]`.

### 3.5 `drone_energy` — Missing Wind Direction
The energy model uses `wind_ms` (scalar) but doesn't accept wind direction, which significantly affects energy calculations for specific legs.

---

## 4. Missing Tools (Recommended Additions)

### 4.1 `v2rmp_inspect_map` ⭐ High Value
Inspect a `.rmp` file's metadata without running optimization:
- Node count, edge count
- Bounding box
- Total network length
- File size, CRC validity

This is the most commonly needed operation — "what map am I working with?"

### 4.2 `v2rmp_validate_map`
Validate `.rmp` file integrity (magic bytes + CRC32 check). Useful for agents to verify downloads.

### 4.3 `v2rmp_pipeline` ⭐ High Value
End-to-end extract → compile → optimize in a single call. The MCP plan doc (`MCP_TOOLS_PLAN.md.bak`) describes this workflow — it should be a single tool for agents.

```json
{
  "name": "v2rmp_pipeline",
  "description": "End-to-end: extract road network → compile to .rmp → optimize route",
  "inputSchema": {
    "source": { "type": "string", "enum": ["overture", "osm"] },
    "bbox": { ... },
    "clean_options": { ... },
    "optimize_options": { ... }
  }
}
```

### 4.4 `list_local_maps`
List `.rmp` files in a given directory. Currently agents have to shell out to `ls`/`find`.

### 4.5 `r2_delete` 
R2 has upload/download/list but no delete. For an ETL pipeline agent, delete is needed to manage storage.

### 4.6 `r2_object_info`
Get metadata (size, last modified) about a specific R2 object — useful for deciding whether to re-download.

### 4.7 `supabase_list_tables` (Safe Read-Only)
A safer alternative to `query_supabase` for exploration. Returns table names and row counts. This avoids the SQL injection risk for common exploration tasks.

### 4.8 `v2rmp_clean`
Standalone GeoJSON cleaning tool. The clean pipeline is powerful but currently only accessible as a side-effect of compile.

### 4.9 `drone_feasibility`
Check if a drone mission is feasible (battery, payload, range) without running the full solver. Useful for agents to validate before committing.

---

## 5. Schema Quality Improvements

### 5.1 Add Parameter Descriptions
Most parameters have no `description` field. AI agents rely on these to understand what to pass. Example:

```json
// Current
"min_lon": { "type": "number" }

// Improved  
"min_lon": {
  "type": "number",
  "description": "Western boundary of the bounding box in degrees (-180 to 180)"
}
```

### 5.2 Add Default Values to Schemas
Several parameters have implicit defaults (implemented in Rust code) but the schema doesn't declare them:
- `remove_isolates` → default: `true`
- `bucket` → default: `"v2rmp"`
- `u_turn_penalty` → default: `10.0`

### 5.3 Add Constraints to Schemas
```json
"min_lon": { "type": "number", "minimum": -180, "maximum": 180 }
"capacity": { "type": "number", "minimum": 0, "exclusiveMinimum": true }
```

### 5.4 Return Structured Results
`v2rmp_optimize` only returns `"Optimized: {:.2} km"` as a text string. The full `OptimizeResult` struct has rich data (efficiency, turn summary, deadhead distance). Return it as structured JSON.

Similarly, `v2rmp_compile` only returns node count. The `CompileResult` has input/output sizes, edge count, elapsed time.

---

## 6. Code Quality Issues

### 6.1 Duplicate `dotenvy::dotenv().ok()` Calls
Called separately in `R2Storage::from_env()` and in `query_supabase`. Should be called once in `main()`.

### 6.2 Flat Error Codes
All errors use code `-32000`. Use proper JSON-RPC error codes:
- `-32700` Parse error
- `-32600` Invalid Request
- `-32601` Method not found ✅ (already used)
- `-32602` Invalid params (for bad arguments)
- `-32000` to `-32099` Server-defined errors

### 6.3 `drone_list_models` Returns a Raw String
```json
{ "text": "[\"FlyCart30\", \"Wing\"]" }
```
This is a JSON string embedded in a text field. Should return proper structured data with descriptions:
```json
[
  {"name": "FlyCart30", "description": "Heavy-lift delivery drone, 30kg payload"},
  {"name": "Wing", "description": "Lightweight delivery drone, 1.2kg payload"}
]
```

### 6.4 No Timeout Protection
Operations like `run_optimize` or `solve_neural` on large maps could run for minutes. No timeout is enforced at the MCP level.

### 6.5 Monolithic Handler Function
`handle_tool_call` is a 300+ line `match` block. Should be refactored into individual handler functions for maintainability and testability.

---

## 7. Summary Priority Matrix

| Priority | Item | Impact |
|----------|------|--------|
| 🔴 P0 | Fix SQL injection in `query_supabase` | Security |
| 🔴 P0 | Fix `.unwrap()` panics in `drone_solve` | Stability |
| 🔴 P0 | Add input validation (bbox, paths, arrays) | Correctness |
| 🟠 P1 | Share single Tokio runtime | Performance |
| 🟠 P1 | Add `v2rmp_inspect_map` tool | Usability |
| 🟠 P1 | Add `v2rmp_pipeline` tool | Agent workflow |
| 🟠 P1 | Return structured results from optimize/compile | Usability |
| 🟠 P1 | Complete Postgres type handling | Correctness |
| 🟠 P1 | Add `additionalProperties: false` to schemas | Spec compliance |
| 🟡 P2 | Add missing parameters to existing tools | Completeness |
| 🟡 P2 | Add parameter descriptions to schemas | Agent usability |
| 🟡 P2 | Add `r2_delete` and `r2_object_info` | Storage mgmt |
| 🟡 P2 | Refactor into handler functions | Maintainability |
| 🟡 P2 | Add `list_local_maps` tool | Agent workflow |
| 🟢 P3 | Add progress notifications | UX |
| 🟢 P3 | Add initialization state tracking | Spec compliance |
| 🟢 P3 | Add `supabase_list_tables` (safe) | Safety |
| 🟢 P3 | Add `v2rmp_validate_map` tool | QA |
| 🟢 P3 | Proper JSON-RPC error codes | Standards |
