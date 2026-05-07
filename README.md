# rmpca - Route Optimization TUI & Agent Engine

[![Crates.io](https://img.shields.io/crates/v/v2rmp.svg)](https://crates.io/crates/v2rmp)
[![Documentation](https://docs.rs/v2rmp/badge.svg)](https://docs.rs/v2rmp)
[![License](https://img.shields.io/crates/l/v2rmp.svg)](LICENSE)

A powerful Terminal User Interface (TUI) and Command-Line Interface (CLI) for route optimization using both the Chinese Postman Problem (CPP) and multi-vehicle Vehicle Routing Problem (VRP) algorithms. Extract road networks from Overture Maps or OpenStreetMap, compile them into efficient binary formats, and optimize routes with turn penalties and depot constraints.

## Features

- 🖥️ **Interactive TUI**: Beautiful terminal interface built with `ratatui`
- 🤖 **Agent-First CLI**: Purpose-built `agent` command for JSON-task automation
- 🗺️ **Data Extraction**: Extract road networks from Overture Maps S3 (Parquet) or OpenStreetMap PBF files
- 🧹 **GeoJSON Cleaning**: Repair geometries, deduplicate edges, and optimize graph topology
- 🔧 **Binary Compilation**: Convert GeoJSON to optimized `.rmp` binary format with CRC32 integrity checking
- 🚗 **Route Optimization (CPP)**: Solve the Chinese Postman Problem with turn penalties and oneway support
- 🚛 **VRP Engine**: Multi-vehicle routing with multiple solvers:
  - **Clarke-Wright Savings**: Classic heuristic for capacity and distance
  - **Sweep Algorithm**: Geometric partitioning
  - **Local Search**: 2-Opt and Or-Opt path improvements
- 📁 **Resource Discovery**: `list` command to discover maps and routes programmatically
- ⚡ **Asynchronous Runtime**: Fully non-blocking I/O using `tokio` for high-performance extraction and processing

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
# Discover resources
rmpca list maps --json
rmpca list routes --json

# Run a task via Agent (JSON payload)
rmpca agent --task task.json --json

# Multi-vehicle VRP
rmpca vrp -i map.rmp --vehicles 5 --algo savings --waypoints stops.json --depot "40.71,-74.01" --output-dir routes/

# Single-vehicle CPP Optimization (Outputs GPX)
rmpca optimize -i map.rmp -o route.gpx --depot "40.71,-74.01"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `extract` | Fetch data from Overture Maps or OSM |
| `clean` | Repair and simplify GeoJSON networks |
| `compile` | Convert GeoJSON to efficient `.rmp` binary |
| `optimize` | Single-vehicle CPP route optimization |
| `vrp` | Multi-vehicle Routing Problem solver |
| `agent` | Execute complex tasks from JSON payloads |
| `list` | Enumerate maps, routes, and solvers |
| `pipeline` | Run extract → clean → compile → optimize |

### AI Agent Task Format
The `agent` command consumes a JSON payload, allowing agents to trigger complex workflows without manual flag management.

```json
{
  "type": "optimize",
  "input": "manhattan.rmp",
  "output": "delivery_route.gpx",
  "num_vehicles": 1,
  "solver_id": "clarke_wright",
  "oneway": "respect",
  "left_penalty": 1.5,
  "depot": [40.7128, -74.0060]
}
```

## Usage (TUI)

### 1. Extract Road Network
- Navigate to **Extract Data** view
- Set bounding box (format: `min_lat,min_lon,max_lat,max_lon`)
- Toggle between OSM and Overture sources
- Output: `extract_YYYYMMDD_HHMMSS.geojson`

### 2. Compile to Binary
- Navigate to **Compile Map** view
- Set input GeoJSON file path
- Output: `.rmp` binary file

### 3. Browse & Optimize
- Use **Browse Cached Maps** to select a compiled map
- Configure turn penalties and depot in the **Optimize Route** view
- Run optimization to generate a GPX/GeoJSON route

## Binary Format (.rmp)

The `.rmp` format is a compact binary representation optimized for graph traversals:

```
[4 bytes]  Magic "RMP1"
[4 bytes]  Node count (u32 LE)
[4 bytes]  Edge count (u32 LE)
[N * 16]   Nodes: lat(f64) + lon(f64)
[E * 17]   Edges: from(u32) + to(u32) + weight_m(f64) + oneway(u8)
[4 bytes]  CRC32 checksum
```

## Library Usage

```rust
use v2rmp::core::{extract, compile, optimize, vrp};

// VRP Solver Example
let vrp_input = vrp::VRPSolverInput {
    stops: my_stops,
    vehicles: 3,
    ..Default::default()
};
let solver = vrp::get_solver("clarke-wright")?;
let output = solver.solve(&vrp_input).await?;
```

## Data Sources

### Overture Maps
- **Status**: ✅ Production Ready
- **Source**: AWS S3 public bucket (`overturemaps-us-west-2`)
- **Format**: Parquet/WKB via `reqwest` and `parquet` crates

### OpenStreetMap
- **Status**: ✅ Production Ready
- **Source**: Local `.osm.pbf` files
- **Parser**: `osmpbf` crate

## Performance
- **Snapping**: 1-meter precision node deduplication
- **Compression**: ~90% reduction vs GeoJSON
- **Speed**: Optimization on 10,000+ edges in <500ms

## Roadmap
- [x] **v0.3.0**: VRP Engine Integration (Clarke-Wright, Sweep, 2-Opt)
- [x] **v0.3.5**: Agent & List commands for machine-to-machine workflows
- [x] **v0.4.0**: Fully Async pipeline and OSM PBF support
- [x] **v0.4.1**: Multi-vehicle VRP CLI command operational with waypoints integration
- [ ] **v0.5.0**: Time Window support (VRPTW) and 3D terrain-aware routing

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
