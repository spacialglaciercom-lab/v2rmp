# rmpca - Route Optimization TUI & Agent Engine

[![Crates.io](https://img.shields.io/crates/v/v2rmp.svg)](https://crates.io/crates/v2rmp)
[![Documentation](https://docs.rs/v2rmp/badge.svg)](https://docs.rs/v2rmp)
[![License](https://img.shields.io/crates/l/v2rmp.svg)](LICENSE)

A powerful Terminal User Interface (TUI) and Command-Line Interface (CLI) for route optimization using both the Chinese Postman Problem (CPP) and multi-vehicle Vehicle Routing Problem (VRP) algorithms. Extract road networks from Overture Maps or OpenStreetMap, compile them into efficient binary formats, and optimize routes with turn penalties and depot constraints.

## Features

- 🖥️ **Interactive TUI**: Beautiful terminal interface built with `ratatui`
- 🖼️ **Graphical UI (egui)**: Full-featured desktop GUI with map visualization, file dialogs, and drag-and-drop controls
- 📐 **BBox Filtering**: Optimize only a subset of a large map by specifying a bounding box — no need to compile a separate .rmp for each area
- 🤖 **Agent-First CLI**: Purpose-built `agent` command for JSON-task automation
- 🗺️ **Data Extraction**: Extract road networks from Overture Maps S3 (Parquet) or OpenStreetMap PBF files
- 🧹 **GeoJSON Cleaning**: Repair geometries, deduplicate edges, and optimize graph topology
- 🔧 **Binary Compilation**: Convert GeoJSON to optimized `.rmp` binary format with CRC32 integrity checking
- 🚗 **Route Optimization (CPP)**: Solve the Chinese Postman Problem with turn penalties and oneway support
- 🚛 **VRP Engine**: Multi-vehicle routing with multiple solvers:
  - **Clarke-Wright Savings**: Classic heuristic for capacity and distance
  - **Sweep Algorithm**: Geometric partitioning
  - **Local Search**: 2-Opt path improvements
  - **Simulated Annealing**: Or-Opt metaheuristic
- ⛰️ **Elevation & Terrain**: DEM GeoTIFF queries (point, profile, stats, fuel calculation)
- 🧠 **Embedding Engine**: Generate text embeddings via fastembed for semantic search
- 🌐 **Headless Server**: JSON-RPC/STDIO `serve` mode for frontend integrations
- 🤝 **MCP Server**: Model Context Protocol server with 15 tools for AI agent integration (extract, compile, optimize, clean, VRP, elevation, fuel, and more)
- 📁 **Resource Discovery**: `list` command to discover maps and routes programmatically
- ⚡ **Asynchronous Runtime**: Fully non-blocking I/O with `tokio` for high-performance extraction and processing

## Installation

### From crates.io

```bash
# TUI + CLI (default — no GUI dependencies)
cargo install v2rmp

# Include the egui desktop GUI
cargo install v2rmp --features gui

# Everything (GUI + ML + extraction)
cargo install v2rmp --all-features
```

### From source

```bash
git clone https://github.com/spacialglaciercom-lab/v2rmp.git
cd v2rmp

# TUI only
cargo build --release

# With GUI
cargo build --release --features gui
```

### Feature Flags

| Feature    | Default | Description |
|------------|---------|-------------|
| `cli`      | ✅      | TUI + CLI (ratatui, crossterm) |
| `gui`      |         | Desktop GUI with map visualization (eframe, egui, rfd) |
| `extract`  | ✅      | Road network extraction from Overture/OSM |
| `ml`       | ✅      | Text embeddings via candle/hf-hub |

## Modes of Operation

### 1. Interactive TUI
Launch the full interface by running `rmpca` without arguments.

```bash
rmpca
```

### 2. Graphical UI (egui)
Launch the desktop GUI with map visualization, file pickers, and all workflow views. Requires the `gui` feature.

```bash
cargo run --release --features gui --bin web-ui
```

The GUI includes all the same views as the TUI — Extract, Clean, Compile, Optimize, VRP — plus an interactive map canvas with zoom/pan, native file dialogs, and a **Bounding Box Filter** on the Optimize page to limit optimization to a sub-region of a large .rmp file.

### 3. Command-Line Interface (CLI)
Use `rmpca <COMMAND>` for scripts, agents, and batch processing. All commands support a `--json` flag for structured output.

```bash
# Discover resources
rmpca list maps --json
rmpca list routes --json

# Run a task via Agent (JSON payload)
rmpca agent --task task.json --json

# Multi-vehicle VRP
rmpca vrp -i map.rmp --vehicles 5 --algo savings --coordinates stops.csv --depot "40.71,-74.01" --output-dir routes/

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
| `elevation` | DEM GeoTIFF queries (point, profile, stats, fuel) |
| `embed` | Generate text embeddings via fastembed |
| `serve` | Headless JSON-RPC/STDIO server |

### MCP Server

rmpca ships an MCP (Model Context Protocol) server that exposes the route optimization pipeline as tools for AI agents. Compatible with Claude Desktop, Claude Code, Cursor, Continue, and any MCP-compliant client.

#### Quick Start

1. **Add to your MCP client config.** The `cargo run` command handles building automatically — no separate build step needed.

   **Claude Desktop** — edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

   ```json
   {
     "mcpServers": {
       "rmpca": {
         "command": "cargo",
         "args": ["run", "--bin", "rmpca-mcp-server", "--release"],
         "cwd": "/path/to/v2rmp"
       }
     }
   }
   ```

   **Claude Code** — run in your project directory:

   ```bash
   claude mcp add rmpca -- cargo run --bin rmpca-mcp-server --release
   ```

   **Cursor / Continue / other clients** — add the same JSON to your client's MCP server config under `mcpServers`.

2. **Restart your MCP client** (or reload the server list).

3. **Ask your AI agent to use the tools.** The agent will discover all four tools automatically.

#### Available Tools

| Tool | Required Params | Description |
|------|----------------|-------------|
| `extract_overture` | `bbox` | Extract road network from Overture Maps S3 by bounding box |
| `extract_osm` | `bbox` | Extract road network from OSM (local PBF or Overpass API) |
| `compile` | `input`, `output` | Convert GeoJSON to binary `.rmp` format with optional cleaning |
| `optimize` | `input` | Run CPP (edge coverage) or VRP (multi-vehicle) route optimization |
| `clean` | `input`, `output` | Full 11-stage GeoJSON cleaning pipeline with all CleanOptions |
| `vrp_solve` | `stops` | Solve VRP from explicit lat/lon stop coordinates (no .rmp needed) |
| `elevation_query` | `dem_path`, `points` | Query elevation at lat/lon points from a local DEM GeoTIFF |
| `elevation_profile` | `dem_path`, `route` | Sample elevation along a route at fixed intervals |
| `elevation_stats` | `dem_path`, `bbox` | Elevation statistics (min, max, avg, coverage) in a bbox |
| `dem_info` | `dem_path` | Return DEM metadata (width, height, bbox, nodata, pixel size) |
| `fuel_estimate` | `samples` | Calculate fuel consumption from an elevation profile |
| `inspect_rmp` | `input` | Parse .rmp binary: node count, edge count, bbox |
| `pipeline` | `bbox` | End-to-end: extract → clean → compile → optimize |
| `get_valhalla_matrix` | `locations` | Fetch real-road distance/time matrix from Valhalla/OSRM |
| `list_solvers` | — | Return available VRP solver IDs and labels |

#### Example Conversation

```
User:  Extract Overture data for Monaco, compile it, and optimize a sweeper route

Agent: [calls extract_overture with bbox {min_lon:7.409, min_lat:43.723, max_lon:7.439, max_lat:43.751}]
       → 2340 nodes, 3102 edges, 45.6 km → extract-output.geojson
       [calls compile with input="extract-output.geojson", output="monaco.rmp"]
       → 1500 nodes, 2100 edges, 12 KB
       [calls optimize with input="monaco.rmp", mode="cpp"]
       → 52.3 km total, 92% efficiency, 12ms
```

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

#### Bounding Box Filter
When working with large maps (e.g., a full city), you can restrict optimization to a smaller area instead of running CPP on the entire network. In the **Optimize Route** view (TUI or GUI), set a bounding box in `min_lon,min_lat,max_lon,max_lat` format. Only nodes and edges within that box are included in the optimization. This dramatically reduces solve time for large datasets.

The core function is also available programmatically:

```rust
use v2rmp::core::optimize::filter_bbox;

let (sub_nodes, sub_edges) = filter_bbox(
    &nodes, &edges,
    Some((45.48, 45.52, -73.62, -73.55)), // min_lat, max_lat, min_lon, max_lon
);
// Pass None to use the full map
```

## Elevation & Fuel

Query DEM GeoTIFF files from the CLI or MCP server. Supported via GDAL — works with any GDAL-compatible raster format.

```bash
# DEM metadata
rmpca elevation --dem dem.tif info

# Single point query
rmpca elevation --dem dem.tif point --lon -73.58 --lat 45.50

# Batch point query from JSON file (or stdin)
echo '[[-73.58,45.50],[-73.57,45.51]]' | rmpca elevation --dem dem.tif points

# Route elevation profile
echo '[[-73.58,45.50],[-73.57,45.51],[-73.56,45.52]]' | rmpca elevation --dem dem.tif profile -s 100

# Bbox elevation statistics
rmpca elevation --dem dem.tif stats --bbox "-73.60,45.48,-73.55,45.52" --step 10

# Fuel consumption from a route (base consumption L/km)
echo '[[-73.58,45.50],[-73.57,45.51]]' | rmpca elevation --dem dem.tif fuel -s 100 --base-consumption 0.08
```

All subcommands support `--json` for machine-readable output. See the MCP server section above for the corresponding `elevation_query`, `elevation_profile`, `elevation_stats`, `dem_info`, and `fuel_estimate` tools.

## VRP Coordinate Input

The VRP solver accepts a **CSV file** of coordinates as its primary input. This replaces the previous JSON waypoints format.

### CSV Format

The file must have a header row. Required columns are `lat` and `lon`; `label`, `demand`, and `type` are optional.

| Column   | Required | Description |
|----------|----------|-------------|
| `lat`    | ✅        | Latitude (WGS-84) |
| `lon`    | ✅        | Longitude (WGS-84) |
| `label`  |          | Human-readable name (auto-generated if omitted) |
| `demand` |          | Per-stop demand (default: 0 for depots, 1 for stops) |
| `type`   |          | `depot` or `stop` (default: `stop`; first row becomes depot if no `type` column) |

### Example

```csv
lat,lon,label,demand,type
45.5017,-73.5673,Montreal Depot,0,depot
45.5032,-73.5701,Customer A,5,stop
45.5100,-73.5800,Customer B,3,stop
45.4980,-73.5750,Customer C,2,stop
45.5050,-73.5900,Customer D,4,stop
```

Minimal (just coordinates — first row is automatically treated as the depot):

```csv
lat,lon
45.5017,-73.5673
45.5032,-73.5701
45.5100,-73.5800
45.4980,-73.5750
```

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
- [x] **v0.4.1**: Multi-vehicle VRP CLI command operational with CSV coordinates input
- [x] **v0.4.2**: Elevation engine (DEM GeoTIFF), Embedding engine (fastembed), Headless serve mode
- [x] **v0.4.3**: BBox deduplication, TUI version auto-sync, FileBrowser ESC fix, CLI guard
- [x] **v0.4.4**: MCP server with 15 tools (extract, compile, optimize, clean, vrp_solve, elevation query/profile/stats, dem_info, fuel_estimate, inspect_rmp, pipeline, get_valhalla_matrix, list_solvers)
- [ ] **v0.5.0**: Time Window support (VRPTW) and 3D terrain-aware routing

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
