# rmpca - Route Optimization TUI

[![Crates.io](https://img.shields.io/crates/v/v2rmp.svg)](https://crates.io/crates/v2rmp)
[![Documentation](https://docs.rs/v2rmp/badge.svg)](https://docs.rs/v2rmp)
[![License](https://img.shields.io/crates/l/v2rmp.svg)](LICENSE)

A powerful Terminal User Interface (TUI) and Command-Line Interface (CLI) for route optimization using the Chinese Postman Problem (CPP) algorithm. Extract road networks from Overture Maps or OpenStreetMap, compile them into efficient binary formats, and optimize routes with turn penalties and depot constraints.

## Features

- 🖥️ **Interactive TUI**: Beautiful terminal interface built with `ratatui`
- 🤖 **Agent-Friendly CLI**: Full command-line interface with `--json` output for automation
- 🗺️ **Data Extraction**: Extract road networks from Overture Maps S3 (Parquet) or OpenStreetMap PBF files
- 🧹 **GeoJSON Cleaning**: Repair geometries, deduplicate edges, and optimize graph topology
- 🔧 **Binary Compilation**: Convert GeoJSON to optimized `.rmp` binary format with CRC32 integrity checking
- 🚗 **Route Optimization**: Solve the Chinese Postman Problem with:
  - Eulerian circuit finding (Hierholzer's algorithm)
  - Turn penalties (left, right, u-turn)
  - Depot location support
  - Oneway street handling
- 🚛 **VRP (Preview)**: Multi-vehicle routing support (Greedy, Savings, Local Search)
- 📁 **Cached Maps Browser**: Automatically scans for compiled `.rmp` files
- ⚡ **Performance**: Concurrent S3 file processing, efficient graph algorithms

## Installation

### From crates.io

```bash
cargo install v2rmp
```

### From source

```bash
git clone https://github.com/spacialglaciercom-lab/v2rmp.git
cd v2rmp
cargo build --release
```

## Modes of Operation

### 1. Interactive TUI
Launch the full interface by running `rmpca` without arguments.

```bash
rmpca
```

### 2. Command-Line Interface (CLI)
Use `rmpca <COMMAND>` for scripts, agents, and batch processing. All commands support a `--json` flag for structured output.

```bash
# Full pipeline in one command
rmpca pipeline --bbox "-74.02,40.68,-73.95,40.80)" --output-dir my_area/

# Extract only
rmpca extract --bbox "-74.02,40.68,-73.95,40.80" --json

# Clean data
rmpca clean -i input.geojson -o cleaned.geojson

# Compile to binary
rmpca compile -i cleaned.geojson -o map.rmp

# Optimize route (Outputs GPX)
rmpca optimize -i map.rmp -o route.gpx --depot "40.71,-74.01"

# Multi-vehicle VRP (Wireframe)
rmpca vrp -i map.rmp --vehicles 5 --depot "40.71,-74.01" --algo savings
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `extract` | Fetch data from Overture Maps or OSM |
| `clean` | Repair and simplify GeoJSON networks |
| `compile` | Convert GeoJSON to efficient `.rmp` binary |
| `optimize` | CPP route optimization (Outputs GPX) |
| `vrp` | Multi-agent Vehicle Routing Problem |
| `pipeline` | Run extract → clean → compile → optimize |

### AI Agent Integration
The `--json` flag on any command ensures that `stdout` contains only structured JSON, while logs and errors go to `stderr`.

```bash
rmpca optimize -i city.rmp --json | jq .efficiency_pct
```

## Quick Start (TUI)

## Usage

### 1. Extract Road Network

1. Navigate to **Extract Data** view
2. Press `b` to set bounding box (format: `min_lat,min_lon,max_lat,max_lon`)
3. Press `Tab` or `s` to toggle between OSM and Overture data sources
4. For OSM: Press `p` to set path to local `.osm.pbf` file
5. Press `Enter` to start extraction
6. Output: `extract_YYYYMMDD_HHMMSS.geojson`

### 2. Compile to Binary

1. Navigate to **Compile Map** view
2. Press `i` to set input GeoJSON file path
3. Press `o` to set output path (optional, auto-derived from input)
4. Press `Enter` to compile
5. Output: `.rmp` binary file

### 3. Browse Cached Maps

1. Navigate to **Browse Cached Maps** view
2. View all `.rmp` files in current directory
3. Use `↑↓` to select a map
4. Press `Enter` to use selected map for optimization

### 4. Optimize Route

1. Navigate to **Optimize Route** view
2. Press `c` to set compiled map file (`.rmp`) or select from Browse view
3. Configure turn penalties:
   - `l` - Left turn penalty (default: 1.0)
   - `R` - Right turn penalty (default: 0.0)
   - `u` - U-turn penalty (default: 5.0)
4. Press `d` to set depot coordinates (optional)
5. Press `Enter` to optimize
6. Output: `route_YYYYMMDD_HHMMSS.json`

## Binary Format (.rmp)

The `.rmp` format is a compact binary representation:

```
[4 bytes]  Magic "RMP1"
[4 bytes]  Node count (u32 LE)
[4 bytes]  Edge count (u32 LE)
[N * 16]   Nodes: lat(f64) + lon(f64)
[E * 17]   Edges: from(u32) + to(u32) + weight_m(f64) + oneway(u8)
[4 bytes]  CRC32 checksum
```

## Keyboard Shortcuts

### Global
- `q` - Quit application
- `Esc` - Return to home
- `h` / `F1` - Toggle help
- `↑↓` - Navigate menus
- `Enter` - Select / confirm

### View-specific
- **Extract**: `Tab`/`s` toggle source, `b` set bbox, `p` set PBF path (OSM only)
- **Compile**: `i` input file, `o` output file
- **Browse Maps**: `↑↓` navigate, `Enter` select
- **Optimize**: `c` cache file, `l`/`R`/`u` penalties, `d` depot

## Library Usage

```rust
use v2rmp::core::{extract, compile, optimize};

// Extract from Overture Maps
let extract_req = extract::ExtractRequest {
    source: extract::ExtractSource::Overture,
    bbox: extract::BBoxRequest {
        min_lon: -73.59,
        min_lat: 45.49,
        max_lon: -73.55,
        max_lat: 45.52,
    },
    road_classes: extract::RoadClass::all_vehicle(),
    output_path: "output.geojson".to_string(),
    pbf_path: None,
};
let result = extract::run_extract(&extract_req)?;

// Extract from OSM PBF
let extract_req = extract::ExtractRequest {
    source: extract::ExtractSource::Osm,
    bbox: extract::BBoxRequest {
        min_lon: -73.59,
        min_lat: 45.49,
        max_lon: -73.55,
        max_lat: 45.52,
    },
    road_classes: extract::RoadClass::all_vehicle(),
    output_path: "output.geojson".to_string(),
    pbf_path: Some("data.osm.pbf".to_string()),
};
let result = extract::run_extract(&extract_req)?;

// Compile to binary
let compile_req = compile::CompileRequest {
    input_geojson: "output.geojson".to_string(),
    output_rmp: "map.rmp".to_string(),
    compress: false,
    road_classes: vec![],
};
let result = compile::run_compile(&compile_req)?;

// Optimize route
let optimize_req = optimize::OptimizeRequest {
    cache_file: "map.rmp".to_string(),
    route_file: Some("route.json".to_string()),
    turn_penalties: optimize::TurnPenalties::default(),
    depot: None,
    oneway_mode: optimize::OnewayMode::Respect,
};
let result = optimize::run_optimize(&optimize_req)?;
```

## Algorithms

### Chinese Postman Problem (CPP)
1. Build graph from road network
2. Find odd-degree vertices
3. Minimum weight perfect matching (greedy nearest-neighbor)
4. Add duplicate edges to make all vertices even-degree
5. Find Eulerian circuit (Hierholzer's algorithm)
6. Calculate efficiency metrics

### Turn Classification
- Bearing-based turn detection
- Categories: straight (±45°), left/right (45-135°), u-turn (>135°)

## Data Sources

### Overture Maps
- Source: AWS S3 public bucket (`overturemaps-us-west-2`)
- Format: Parquet with WKB geometry
- Release: 2026-04-15.0
- No authentication required
- **Status**: ✅ Fully functional

### OpenStreetMap
- Format: PBF (Protocol Buffer Format)
- Source: Local `.osm.pbf` files (download from Geofabrik, BBBike, etc.)
- **Status**: ✅ Fully functional

## Performance

- **Concurrent S3 Processing**: Up to 10 files in parallel
- **Node Deduplication**: 1-meter precision snapping
- **Binary Format**: ~90% compression vs GeoJSON
- **Graph Algorithms**: O(E log V) for matching, O(E) for Eulerian circuit

## Requirements

- Rust 1.70+
- Internet connection (for Overture Maps extraction)
- Local OSM PBF file (for OpenStreetMap extraction)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgments

- [Overture Maps Foundation](https://overturemaps.org/) for open road network data
- [OpenStreetMap](https://www.openstreetmap.org/) contributors for global map data
- [ratatui](https://github.com/ratatui-org/ratatui) for the excellent TUI framework
- Chinese Postman Problem algorithm based on classical graph theory

## Roadmap

- [x] **v0.1.0**: Overture extraction, OSM PBF extraction, cached maps browser, basic optimization, TUI
- [ ] **v0.2.0**: Async/threaded operations, comprehensive tests (✅ 36 tests pass), performance optimizations
- [ ] **v0.3.0**: Multi-depot support, advanced features (✅ GeoJSON Cleaning implemented)
- [ ] **v0.4.0+**: Time windows, capacity constraints
- [ ] **v1.0.0**: Stable API, backward compatibility guarantees

## Support

- Documentation: [docs.rs/v2rmp](https://docs.rs/v2rmp)
- Issues: [GitHub Issues](https://github.com/spacialglaciercom-lab/v2rmp/issues)
- Discussions: [GitHub Discussions](https://github.com/spacialglaciercom-lab/v2rmp/discussions)
