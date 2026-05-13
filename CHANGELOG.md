# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.7] - 2026-05-13

### Changed
- Excluded session logs, Python scripts, test artifacts, and documentation drafts from crate package to reduce download size.

## [0.4.6] - 2026-05-11

### Added
- **AutoML hyperparameter tuner**: Predicts optimal solver settings (max iterations, temperature, tabu tenure) based on instance features using a learned MLP model.
- **Instance Feature Extraction**: 28-dimensional geometric and graph-theoretic feature vector ( Prim MST, spectral gap, clustering coefficient) for VRP instances.
- **Learned Solver Selector**: Neural ensemble that recommends the best algorithm for a given routing problem.
- **Neural-Guided Local Search**: Solvers now use MLP-based move scoring (16-dim features) to prioritize the best neighborhood improvements.
- **LLM-based NLP Query Parser**: Integration with Qwen2.5-0.5B-Instruct for natural language routing request parsing.
- **Route Quality Predictor**: Pre-solve estimation of gap-to-optimal and tour length.
- **Expanded MCP Server**: Now includes 21 tools covering the full AI/ML routing stack.

### Changed
- **Heavy Feature Gating**: All AI/ML dependencies (candle, tokenizers) are now optional and gated behind the `ml` feature flag.

## [0.4.3] - 2026-05-08

### Added
- **S3 integration tests**: Anonymous public-bucket access verified for Overture Maps (list, download, partition discovery).

### Changed
- **BBox deduplication**: Three identical `BBox` structs (osm, overture, elevation) extracted into canonical `src/core/geo_types.rs` with `Copy + PartialEq`. All modules re-export from single source.
- **TUI version auto-sync**: Version string in header and startup log now uses `env!("CARGO_PKG_VERSION")` instead of hardcoded `v0.3.9`.
- **FileBrowser ESC fix**: Cancelling file browser now restores previous view instead of leaving user stuck in dead `FileBrowser` state.
- **CLI entry-point guard**: Stale `View::FileBrowser` with no backing browser falls through to normal key processing.

## [0.4.2] - 2026-05-07

### Added
- **Elevation engine**: DEM GeoTIFF queries via GDAL — point, multi-point, route profile, bbox stats, DEM info, and fuel consumption calculator (`LocalDem`, `FuelCalculator`).
- **Embedding engine**: Text embedding via fastembed with `candle-core` / `candle-nn` / `candle-transformers`.
- **Headless serve mode** (`rmpca serve`): JSON-RPC/STDIO server accepting `AgentTask` payloads for long-running frontend integrations.
- **`elevation` CLI**: Subcommands: `point`, `points`, `profile`, `stats`, `info`, `fuel`.
- **`embed` CLI**: Generate embeddings for one or more texts.

### Changed
- Updated crate description to "rmpca — Route Optimization TUI & Agent Engine".

## [0.4.1] - 2026-05-05

### Added
- Multi-vehicle VRP CLI command (`rmpca vrp`) with algorithm selection, waypoints JSON, depot specification, capacity, and GPX route output.
- VRP solver algorithms: greedy (default), savings (Clarke-Wright), local-search (2-Opt), simulated-annealing (Or-Opt).

## [0.4.0] - 2026-05-05

### Added
- Fully async pipeline with `tokio`: extract → clean → compile → optimize.
- OSM PBF extraction support via `osmpbf` 0.3.
- Overture Maps S3 extraction via `object_store` with Parquet/WKB parsing.
- `pipeline` CLI command and `PipelineResult` JSON output.
- `rmpca-extract` standalone binary.

## [0.3.9] - 2026-05-05

### Added
- Full TUI wiring to core logic (Extraction, Compilation, Optimization, Clean).
- Road network cleaning module with geometry repair and graph optimization.
- Enhanced file browser with filtering by extension and parent directory navigation.
- CRC32 integrity checking for binary .rmp format.
- 36 unit tests across core algorithms and application state.

## [0.3.8] - 2026-05-03

### Added
- Initial VRP engine: Clarke-Wright, Sweep, 2-Opt solvers.
- Cached maps browser and saved routes discovery.

## [0.3.7] - 2026-05-03

### Added
- Agent & List CLI commands for machine-to-machine workflows.

## [0.3.5] - 2026-05-03

### Added
- VRP Engine Integration (Clarke-Wright, Sweep, 2-Opt).

## [0.1.1] - 2026-05-03

### Added
- Clipboard support in TUI input fields (Ctrl+C/V/X).
- Full TUI implementation with all view modules.

## [0.1.0] - 2026-05-03

### Added
- Initial release of v2rmp (rmpca).
- Interactive TUI for route optimization workflow.
- Extract road networks from Overture Maps S3 (Parquet) and OpenStreetMap PBF.
- Compile GeoJSON to binary `.rmp` format with CRC32 integrity.
- Route optimization using Chinese Postman Problem algorithm.
- Turn penalty support and oneway street handling.
- Node deduplication with 1-meter precision snapping.

[Unreleased]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.4.3...HEAD
[0.4.3]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.3.9...v0.4.0
[0.3.9]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.3.5...v0.3.7
[0.3.5]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.1.1...v0.3.5
[0.1.1]: https://github.com/spacialglaciercom-lab/v2rmp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/spacialglaciercom-lab/v2rmp/releases/tag/v0.1.0
