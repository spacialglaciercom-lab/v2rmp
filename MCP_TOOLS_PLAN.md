# v2rmp MCP Tools — Implementation Plan

## Current MCP Tools (4)

| Tool | Description |
|------|-------------|
| `extract_overture` | Extract road network from Overture Maps S3 Parquet |
| `extract_osm` | Extract road network from OSM PBF/Overpass |
| `compile` | Compile GeoJSON → .rmp binary (optional clean with defaults) |
| `optimize` | Run CPP/VRP route optimization on .rmp file |

## New MCP Tools to Implement (by priority)

### HIGH Priority

| # | Tool | Description | Source |
|---|------|-------------|--------|
| 1 | **`clean`** | Full GeoJSON cleaning pipeline with all `CleanOptions` parameters. Current `compile` only offers `clean: bool` with defaults — this exposes `min_length_m`, `node_snap_m`, `max_components`, `simplify_tolerance_m`, `dedupe_edges`, `remove_isolates`, `merge_parallel_edges`, `merge_parallel_edge_properties`, `property_merge_strategy`, `make_valid`, `drop_invalid`, `remove_selfloops`, `include_polygons`, `include_points`, `required_attrs`. | `src/core/clean.rs` → `clean_geojson()` |
| 2 | **`vrp_solve`** | Dedicated VRP solver taking explicit stop coordinates (lat/lon array) + depot, not an .rmp file. Supports solver selection (clarke_wright, sweep, two_opt, or_opt, default), number of vehicles, vehicle capacity, and distance matrix. Returns routes with per-vehicle stop assignments and statistics. | `src/core/vrp/registry.rs` → `solve_with()`, `src/core/vrp/types.rs` |
| 3 | **`elevation_query`** | Query elevation at one or more lat/lon points from a local DEM GeoTIFF. Returns elevation or null for each point (no-coverage). | `src/core/elevation/local.rs` → `LocalDem::get_elevation()` / `get_elevations()` |
| 4 | **`elevation_profile`** | Sample elevation along a route at fixed intervals. Returns total ascent, descent, min/max/avg elevation, distance, and per-sample points. | `src/core/elevation/local.rs` → `LocalDem::route_profile()` |

### MEDIUM Priority

| # | Tool | Description | Source |
|---|------|-------------|--------|
| 5 | **`elevation_stats`** | Compute elevation statistics (min, max, avg, coverage %) within a bounding box from a DEM file. | `src/core/elevation/local.rs` → `bbox_stats()` |
| 6 | **`dem_info`** | Return DEM metadata (width, height, bbox, nodata, pixel size). | `src/core/elevation/local.rs` → `info()` |
| 7 | **`fuel_estimate`** | Calculate fuel consumption from an elevation profile (base L/km + grade adjustments). | `src/core/elevation/fuel.rs` → `FuelCalculator::calculate()` |
| 8 | **`inspect_rmp`** | Parse .rmp binary and return node count, edge count, bounding box without running optimization. | `src/core/optimize.rs` → `read_rmp_file()` |
| 9 | **`pipeline`** | End-to-end: extract → clean → compile → optimize in one call. | `src/cli.rs` → `run_pipeline_cmd()` |
| 10 | **`get_valhalla_matrix`** | Fetch real-road distance/time matrix from Valhalla/OSRM. | `src/core/vrp/utils.rs` → `get_valhalla_matrix()` |

### LOW Priority

| # | Tool | Description | Source |
|---|------|-------------|--------|
| 11 | **`list_solvers`** | Return available VRP solver IDs and labels. | `src/core/vrp/registry.rs` → `get_solver_list()` |
| 12 | **`embed`** | Generate BERT embeddings for text strings. | `src/core/embed.rs` → `run_embed()` |
| 13 | **`haversine_distance`** | Great-circle distance between two lat/lon points. | `src/core/mod.rs` → `haversine_m()` |
| 14 | **`classify_turn`** | Classify bearing delta as left/right/u_turn/straight. | `src/core/optimize.rs` → `classify_turn()` |
| 15 | **`parse_csv_stops`** | Parse CSV coordinates into VRPSolverStop objects. | `src/core/vrp/utils.rs` → `parse_csv_stops()` |

---

## Key Design Gap

The MCP `compile` tool's `clean: bool` parameter only uses `CleanOptions::default()`. The CLI's `rmpca clean` command exposes fine-grained control over cleaning parameters. The new `clean` tool bridges this gap.

## Implementation Notes

- All tools follow the existing JSON-RPC 2.0 stdio protocol pattern in `rmpca-mcp-server.rs`
- Tools use `send()` / `send_err()` / `tool_success()` helpers
- Each tool needs: tool definition (name, description, input_schema) + handler function + match arm in main loop
- Elevation tools require GDAL (feature-gated behind `extract` feature) — need conditional compilation