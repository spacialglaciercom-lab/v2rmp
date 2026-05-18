# rmpca - Route Optimization TUI & Agent Engine

[![Crates.io](https://img.shields.io/crates/v/v2rmp.svg)](https://crates.io/crates/v2rmp)
[![Documentation](https://docs.rs/v2rmp/badge.svg)](https://docs.rs/v2rmp)
[![License](https://img.shields.io/crates/l/v2rmp.svg)](LICENSE)

A powerful Terminal User Interface (TUI) and agent engine for route optimization. Solve Chinese Postman Problem (CPP), Vehicle Routing Problem (VRP), and complex CVRP with neural networks. Extract road networks from Overture Maps, OpenStreetMap, PMTiles, or PostGIS databases, compile them into efficient binary formats, and optimize routes with turn penalties, depot constraints, and multi-vehicle support.

## Features

### 🗺️ Data Extraction
- **Overture Maps**: Extract from AWS S3 public bucket (Parquet with WKB geometry)
- **OpenStreetMap**: Extract from local `.osm.pbf` files (Geofabrik, BBBike, etc.)
- **PMTiles**: Extract vector tiles from `.pmtiles` archives with layer filtering and zoom level selection
- **PostGIS**: Direct extraction from PostgreSQL/PostGIS `road_edges` tables

### 🔧 Binary Compilation
- Convert GeoJSON to optimized `.rmp` binary format
- CRC32 integrity checking
- ~90% compression vs GeoJSON
- Node deduplication at 1-meter precision

### 🚗 Route Optimization
- **CPP**: Chinese Postman Problem with Hierholzer's algorithm, perfect matching, and turn penalties
- **VRP**: Multi-vehicle routing with capacity constraints
  - Greedy algorithm
  - Clarke-Wright Savings
  - Sweep algorithm
  - Two-Opt local search
  - OR-Tools (if available)
  - Neural-Guided heuristic
  - Neural ONNX direct inference
- **Drone VRP**: Physics-based energy modeling with Dorling et al. (2017) power model
  - **Neural GNN Solver** 🆕: Reinforcement Learning agent (GCN-based) for energy-aware routing in complex terrain
  - Supported models: FlyCart30, Wing
  - Wind speed consideration
  - No-fly zone support
  - Integrated with `drone-rl-agent` on Hugging Face

### 🤖 Machine Learning
- **Solver Selector**: Auto-selects best VRP algorithm based on instance features
- **Quality Predictor**: Predicts expected route quality before solving
- **Move Scorer**: Neural network for evaluating route improvement moves
- **AutoML**: Automated hyperparameter tuning
- **Drone RL Agent** 🆕: Graph Convolutional Network (GCN) policy for autonomous decision making
  - **State Space**: 67-dim (64-dim FastRP embeddings + 3-dim meta-state: Wind, Battery, Payload)
  - **Inference**: Powered by `ort` (ONNX Runtime) v2.0
  - **Training**: RL-trained on Montreal's Mile End road network
- **Graph Embeddings**: Generate dense vector representations for road network nodes and edges
  - **node2vec**: Biased random walks + Skip-gram (Grover & Leskovec, 2016)
  - **LINE**: 1st + 2nd order proximity preservation (Tang et al., 2015)
  - **FastRP**: Sparse random projection — fastest method (Chen et al., 2019)
  - **Spatial**: Handcrafted structural features (degree, centrality, clustering, coordinates)
  - Available in CLI (`rmpca graph-embed`), TUI (Graph Embeddings view), and Agent JSON
  - No pre-trained model required — all methods work unsupervised directly from `.rmp` files
- **NLP Parser**: Natural language query parsing into structured routing requests

### ☁️ Cloud & Storage
- **R2 Cloud Storage**: Cloudflare R2 object storage integration
  - List, upload, download, delete objects
  - Environment variable configuration (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`)
- **Cached Maps Browser**: Automatically scans for `.rmp` files in current directory

### 🔗 Integration
- **MCP Server**: Model Context Protocol server for AI agent integration
  - 20+ tools: extract, compile, optimize, vrp_solve, elevation, partition, etc.
  - JSON-RPC/STDIO interface
- **OsmAnd Deep Links**: Generate `osmand://import?url=` links for mobile GPX import
- **Google Maps URLs**: Generate clickable direction URLs (auto-sampled to ≤20 waypoints)
- **Chained Pipeline**: One-shot binary for PMTiles → compile → CPP optimize → GPX output

### 🖥️ User Interfaces
- **Interactive TUI**: Beautiful terminal interface built with `ratatui`
- **CLI**: Full command-line interface with JSON output support
- **Web UI**: Experimental web-based interface (feature-gated)

### ⚡ Performance
- Concurrent S3 file processing (up to 10 files in parallel)
- GNN/Attention-based neural solvers
- Efficient graph algorithms: O(E log V) for matching, O(E) for Eulerian circuit
- Async runtime for I/O operations

---

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

### With all features

```bash
# Requires libgdal-dev for extract feature
sudo apt-get install libgdal-dev  # Ubuntu/Debian
brew install gdal                  # macOS
cargo build --release --all-features
```

---

## Quick Start

### Launch the TUI

```bash
rmpca
```

Use arrow keys to navigate, `Enter` to select, `q` to quit, `Esc` to return home.

### Command-line extraction

```bash
# Extract from Overture Maps (requires extract feature)
rmpca extract --bbox "45.49,-73.59,45.52,-73.55" --output montreal.geojson

# Extract from OSM PBF file (requires extract feature)
rmpca extract --source osm --pbf-path data.osm.pbf --bbox "45.49,-73.59,45.52,-73.55" --output montreal.geojson

# Extract from PMTiles
rmpca-extract --pmtiles-path montreal.pmtiles --bbox "45.49,-73.59,45.52,-73.55" --output montreal.geojson

# Extract from PostGIS (direct)
rmpca-extract --postgis --bbox "45.49,-73.59,45.52,-73.55" --database-url postgresql://user:pass@host/db
```

---

## Usage

### TUI Views

#### 1. Extract Road Network
1. Navigate to **Extract Data** view
2. Press `b` to set bounding box (format: `min_lat,min_lon,max_lat,max_lon`)
3. Press `Tab` or `s` to toggle between OSM, Overture, PMTiles, and PostGIS sources
4. For OSM: Press `p` to set path to local `.osm.pbf` file
5. For PMTiles: Press `P` to set `.pmtiles` file path
6. For PostGIS: Press `D` to set database URL
7. Press `Enter` to start extraction
8. Output: `extract_YYYYMMDD_HHMMSS.geojson`

#### 2. PMTiles Extract (Dedicated View)
1. Navigate to **PMTiles Extract** view
2. Press `P` to set `.pmtiles` file path
3. Press `b` to set bounding box
4. Press `o` to set output path
5. Press `L` to set layer name (optional)
6. Press `z` to set zoom level (optional)
7. Press `Enter` to extract

#### 3. PostGIS CPP (Dedicated View)
1. Navigate to **PostGIS CPP** view
2. Press `b` to set bounding box
3. Press `D` to set database URL
4. Press `t` to set table name (default: `road_edges`)
5. Configure turn penalties: `l`, `R`, `u`
6. Press `Enter` to solve directly from PostGIS

#### 4. Compile to Binary
1. Navigate to **Compile Map** view
2. Press `i` to set input GeoJSON file path
3. Press `o` to set output path (optional, auto-derived from input)
4. Press `Enter` to compile
5. Output: `.rmp` binary file

#### 5. Browse Cached Maps
1. Navigate to **Browse Cached Maps** view
2. View all `.rmp` files in current directory
3. Use `↑↓` to select a map
4. Press `Enter` to use selected map for optimization

#### 6. Optimize Route (CPP)
1. Navigate to **Optimize Route** view
2. Press `c` to set compiled map file (`.rmp`) or select from Browse view
3. Configure turn penalties:
   - `l` - Left turn penalty (default: 1.0)
   - `R` - Right turn penalty (default: 0.0)
   - `u` - U-turn penalty (default: 5.0)
4. Press `d` to set depot coordinates (optional)
5. Press `L` to set OsmAnd base URL for GPX import links
6. Press `Enter` to optimize
7. Output: `route_YYYYMMDD_HHMMSS.json` + optional GPX
8. **Output Links**: Google Maps URLs and OsmAnd deep links are generated automatically

#### 7. VRP Solver (Heuristic)
1. Navigate to **VRP Solver** view
2. Press `i` to set road network `.rmp` (optional, for map preview)
3. Press `w` to set waypoints JSON (e.g. `[[lat,lon], ...]`)
4. Press `v` to set number of vehicles
5. Press `a` to toggle algorithms (Greedy, Savings, Local Search, OR-Tools, Neural-Guided)
6. Press `k` to set vehicle capacity
7. Press `L` to set OsmAnd base URL for GPX import links
8. Press `Enter` to solve

#### 8. Neural ONNX Solver
1. Navigate to **Neural ONNX Solver** view
2. Press `m` to select an ONNX model (defaults to `cvrp50_model.onnx`)
3. Press `w` to select waypoints JSON file
4. Press `v` to set number of vehicles
5. Press `k` to set vehicle capacity
6. Press `o` to set output directory
7. Press `Enter` to run high-performance GNN-based optimization

#### 9. Graph Embeddings 🆕
Generate dense vector representations for road network nodes and edges — useful for downstream ML tasks, similarity search, and graph analysis.

1. Navigate to **Graph Embeddings** view
2. Press `I` to browse and select an input `.rmp` file
3. Press `M` to cycle through methods: `line` → `fastrp` → `spatial` → `node2vec`
4. Press `D` to set embedding dimensions (default: 64)
5. Press `E` to toggle edge embeddings on/off (derived via Hadamard product)
6. Press `O` to set output `.json` file path (optional — auto-saves to `*_embeddings.json`)
7. Press `Enter` to generate embeddings

**Method comparison (14K node road network):**

| Method | Speed | Description |
|--------|-------|-------------|
| `fastrp` | <1s | Sparse random projection, no optimization needed |
| `spatial` | ~2s | Handcrafted: degree, betweenness centrality, eigenvector centrality, clustering coefficient, coordinates |
| `line` | ~3s | SGD over edges preserving 1st + 2nd order graph proximity |
| `node2vec` | ~25s | Biased random walks + Skip-gram, captures community structure and structural roles |

#### 9. Drone VRP
1. Navigate to **Drone VRP** view
2. Press `m` to select drone model (FlyCart30 or Wing)
3. Press `d` to set depot coordinates
4. Press `w` to set customer waypoints JSON
5. Press `D` to set demands JSON
6. Press `s` to set wind speed (m/s)
7. Press `Enter` to solve with energy calculation

#### 10. Zone Partition
1. Navigate to **Partition** view
2. Press `e` to set edge list JSON or `p` for points JSON
3. Press `k` to set number of zones
4. Press `b` to set balance metric (distance, time, count)
5. Press `Enter` to partition using spectral clustering

---

## Binaries

| Binary | Description | Features Required |
|--------|-------------|-------------------|
| `rmpca` | Main TUI application | default |
| `rmpca-extract` | CLI data extraction | `extract` |
| `rmpca-drone` | Drone VRP solver CLI | default |
| `rmpca-chained` | One-shot pipeline: PMTiles → compile → CPP → GPX | default |
| `rmpca-mcp` | Lightweight MCP server (JSON-RPC/STDIO) | default |
| `rmpca-mcp-server-legacy` | Full MCP server with 20+ tools | default |
| `zilliz-mcp-server` | Vector database MCP integration | default |
| `web-ui` | Experimental web interface | `gui` |
| `generate-training-data` | Generate ML training data | `extract` |
| `evaluate-selector` | Evaluate VRP solver selector | `extract` |

---

## CLI Commands

```bash
# Main application (TUI)
rmpca

# Extract road network (requires extract feature)
rmpca extract --source overture --bbox "min_lat,min_lon,max_lat,max_lon" --output output.geojson
rmpca extract --source osm --pbf-path data.osm.pbf --bbox "..." --output output.geojson

# Compile GeoJSON to binary
rmpca compile --input input.geojson --output map.rmp

# Clean GeoJSON road network
rmpca clean --input input.geojson --output cleaned.geojson

# Optimize route (CPP)
rmpca optimize --cache map.rmp --route output.json --left-penalty 1.0 --uturn-penalty 5.0

# Solve VRP
rmpca vrp --waypoints waypoints.json --vehicles 5 --capacity 1000

# Full pipeline: extract -> clean -> compile -> optimize
rmpca pipeline --bbox "..." --output-prefix montreal

# List available resources
rmpca list

# Agent mode: execute JSON task plan
rmpca agent --plan tasks.json

# Serve mode: headless JSON-RPC/STDIO server
rmpca serve

# Generate text embeddings (requires ml feature)
rmpca embed --text "route optimization" --text "vehicle routing"

# Elevation queries from DEM GeoTIFF (requires extract feature)
rmpca elevation point --dem elevation.tif --lon -73.5 --lat 45.5
rmpca elevation profile --dem elevation.tif --input route.json
rmpca elevation fuel --dem elevation.tif --input route.json --base-consumption 0.1

# Predict best VRP solver (requires ml feature)
rmpca predict-solver --waypoints waypoints.json --vehicles 5

# Predict route quality (requires ml feature)
rmpca predict-quality --waypoints waypoints.json --vehicles 5

# Parse natural language query
rmpca parse-query --query "optimize route from Montreal to Toronto with 3 stops"

# Generate graph embeddings (requires ml feature)
rmpca graph-embed -i map.rmp -m line -d 64 -o embeddings.json
rmpca graph-embed -i map.rmp -m fastrp -d 128 --include-edges
rmpca graph-embed -i map.rmp -m node2-vec -d 32 --walk-length 20 --num-walks 5
rmpca graph-embed -i map.rmp -m spatial -d 64 -o spatial_features.json
```

---

## MCP Server

The Model Context Protocol (MCP) server provides AI agent integration with 20+ tools.

### Starting the server

```bash
# Lightweight server
rmpca-mcp

# Full server with all tools
rmpca-mcp-server-legacy
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `extract_overture` | Extract from Overture Maps S3 |
| `extract_osm` | Extract from OSM PBF |
| `extract_pmtiles` | Extract from PMTiles archive |
| `extract_postgis` | Extract from PostGIS database |
| `compile` | Compile GeoJSON to .rmp |
| `optimize` | Solve CPP |
| `vrp_solve` | Solve VRP |
| `clean` | Clean GeoJSON |
| `pipeline` | Full extract→clean→compile→optimize pipeline |
| `elevation_query` | Query elevation at a point |
| `elevation_profile` | Get elevation profile along a route |
| `elevation_stats` | Get elevation statistics for a bbox |
| `dem_info` | Show DEM file metadata |
| `fuel_estimate` | Calculate fuel consumption |
| `predict_solver` | Predict best VRP solver |
| `score_route` | Score a route |
| `route_embedding` | Generate route embedding |
| `tune_hyperparams` | Tune solver hyperparameters |
| `list_r2_bucket` | List R2 bucket objects |
| `upload_to_r2` | Upload file to R2 |
| `download_from_r2` | Download file from R2 |
| `postgis_cpp` | Solve CPP directly from PostGIS |
| `partition` | Partition graph into zones |
| `partition_from_points` | Partition from point list |
| `haversine_distance` | Calculate haversine distance |
| `get_valhalla_matrix` | Get Valhalla distance matrix |
| `inspect_rmp` | Inspect .rmp file |
| `list_solvers` | List available VRP solvers |
| `parse_query` | Parse natural language query |
| `graph_embed` | Generate node/edge embeddings for a road network |
| `submit_feedback` | Submit feedback for ML models |
| `generate_osmand_link` | Generate OsmAnd deep link |

### MCP Client Configuration

```json
{
  "mcpServers": {
    "rmpca": {
      "command": "cargo",
      "args": ["run", "--bin", "rmpca-mcp-server-legacy", "--release"]
    }
  }
}
```

---

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

---

## Keyboard Shortcuts

### Global
| Key | Action |
|-----|--------|
| `q` | Quit application |
| `Esc` | Return to home |
| `h` / `F1` | Toggle help |
| `↑↓` | Navigate menus |
| `Enter` | Select / confirm |
| `Tab` | Cycle through fields |

### View-specific
| View | Shortcut | Action |
|------|----------|--------|
| **Extract** | `b` | Set bounding box |
| **Extract** | `Tab`/`s` | Toggle source (Overture/OSM/PMTiles/PostGIS) |
| **Extract** | `p` | Set PBF path (OSM only) |
| **Extract** | `P` | Set PMTiles path |
| **Extract** | `D` | Set database URL (PostGIS) |
| **Compile** | `i` | Set input file |
| **Compile** | `o` | Set output file |
| **Browse** | `↑↓` | Navigate |
| **Browse** | `Enter` | Select |
| **Optimize** | `c` | Set cache file |
| **Optimize** | `r` | Set route file |
| **Optimize** | `l`/`R`/`u` | Set turn penalties |
| **Optimize** | `d` | Set depot |
| **Optimize** | `L` | Set OsmAnd base URL |
| **VRP** | `i` | Set road network |
| **VRP** | `w` | Set waypoints |
| **VRP** | `v` | Set vehicle count |
| **VRP** | `a` | Toggle algorithm |
| **VRP** | `k` | Set capacity |
| **VRP** | `L` | Set OsmAnd base URL |
| **Neural** | `m` | Set model path |
| **Neural** | `w` | Set waypoints |
| **Neural** | `v` | Set vehicle count |
| **Neural** | `k` | Set capacity |
| **Neural** | `o` | Set output directory |
| **Drone** | `m` | Set drone model |
| **Drone** | `d` | Set depot |
| **Drone** | `w` | Set customers |
| **Drone** | `D` | Set demands |
| **Drone** | `s` | Set wind speed |
| **Graph Embed** | `I` | Set input .rmp file (file browser) |
| **Graph Embed** | `O` | Set output .json file |
| **Graph Embed** | `M` | Cycle embedding method |
| **Graph Embed** | `D` | Set dimensions |
| **Graph Embed** | `E` | Toggle edge embeddings |
| **Partition** | `e` | Set edge list |
| **Partition** | `p` | Set points |
| **Partition** | `k` | Set zone count |
| **Partition** | `b` | Set balance metric |

---

## Library Usage

```rust
use v2rmp::core::{extract, compile, optimize, pmtiles_extract, postgis_cpp};
use v2rmp::core::drone::{DroneModel, DroneSolver, DroneVrpInstance};
use v2rmp::core::neural_routing::{NeuralRouteRequest, NeuralInferenceEngine};
use v2rmp::core::partition::{partition_edges, partition_from_points, PartitionResponse};
use v2rmp::core::r2::R2Storage;
use v2rmp::core::vrp::utils::{generate_osmand_import_url, build_haversine_matrix};

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

// Extract from PMTiles
let pmtiles_req = pmtiles_extract::PmtilesExtractRequest {
    pmtiles_path: "montreal.pmtiles".to_string(),
    min_lon: -73.59,
    min_lat: 45.49,
    max_lon: -73.55,
    max_lat: 45.52,
    output_path: "output.geojson".to_string(),
    zoom: Some(14),
    layer_name: Some("transportation".to_string()),
};
let result = pmtiles_extract::run_pmtiles_extract(&pmtiles_req).await?;

// Extract from PostGIS and solve CPP directly
let postgis_req = postgis_cpp::PostGisCppRequest {
    bbox: [-73.59, 45.49, -73.55, 45.52],
    road_classes: vec!["residential".to_string(), "tertiary".to_string()],
    oneway_mode: postgis_cpp::OneWayMode::Respect,
    turn_penalties: postgis_cpp::TurnPenalties {
        left: 1.0,
        right: 0.0,
        uturn: 5.0,
    },
    depot: Some((45.505, -73.565)),
    database_url: Some("postgresql://user:pass@host/db".to_string()),
    table_name: Some("road_edges".to_string()),
    output_path: Some("route.json".to_string()),
};
let result = postgis_cpp::run_postgis_cpp(&postgis_req)?;

// Compile to binary
let compile_req = compile::CompileRequest {
    input_geojson: "output.geojson".to_string(),
    output_rmp: "map.rmp".to_string(),
    prune_disconnected: false,
    compress: false,
    road_classes: vec![],
    clean_options: None,
};
let result = compile::run_compile(&compile_req)?;

// Optimize route
let optimize_req = optimize::OptimizeRequest {
    cache_file: "map.rmp".to_string(),
    route_file: Some("route.json".to_string()),
    turn_penalties: optimize::TurnPenalties {
        left: 1.0,
        right: 0.0,
        uturn: 5.0,
    },
    depot: None,
    oneway_mode: optimize::OnewayMode::Respect,
};
let result = optimize::run_optimize(&optimize_req)?;

// Generate OsmAnd deep link
let gpx_url = "https://example.com/route.gpx";
let osmand_link = generate_osmand_import_url(gpx_url);
// Returns: osmand://import?url=https%3A%2F%2Fexample.com%2Froute.gpx

// Drone VRP
let drone_instance = DroneVrpInstance {
    drone_model: DroneModel::FlyCart30,
    depot: [45.505, -73.565],
    customers: vec![[45.506, -73.566], [45.507, -73.567]],
    demands_kg: vec![1.0, 2.0],
    wind_speed_ms: 2.5,
    no_fly_zones: vec![],
};
let result = DroneSolver::solve(&drone_instance)?;

// Neural ONNX routing
let mut engine = NeuralInferenceEngine::new("cvrp50_model.onnx")?;
let req = NeuralRouteRequest {
    locations: vec![[45.505, -73.565, 0.0], [45.506, -73.566, 0.0]],
    demands: vec![0.0, 1.0],
    capacity: 10.0,
    model_path: "cvrp50_model.onnx".to_string(),
};
let result = engine.solve(&req)?;

// Graph partitioning
let edges = vec![];
let response: PartitionResponse = partition_edges(&edges, 5, 3, "time")?;

// Graph embeddings — generate node/edge vectors from .rmp
#[cfg(feature = "ml")]
{
    use v2rmp::core::ml::node_embed::{EmbedConfig, EmbedMethod, embed_graph};
    use v2rmp::core::optimize::read_rmp_file;

    let data = std::fs::read("map.rmp")?;
    let (nodes, edges) = read_rmp_file(&data)?;

    let config = EmbedConfig {
        method: EmbedMethod::Line,
        dimensions: 64,
        include_edges: true,
        ..Default::default()
    };
    let result = embed_graph(&nodes, &edges, &config)?;
    // result.nodes: Vec<NodeEmbedding> — each node has a 64-dim vector
    // result.edges: Option<Vec<EdgeEmbedding>> — hadamard product of node pairs
}

// R2 Cloud Storage
let r2 = R2Storage::from_env("my-bucket")?;
let objects = r2.list_objects(Some("prefix/")).await?;
r2.upload_object("path/to/file.txt", vec![1, 2, 3]).await?;
let data = r2.download_object("path/to/file.txt").await?;
```

---

## Algorithms

### Chinese Postman Problem (CPP)
1. Build graph from road network
2. Find odd-degree vertices
3. Minimum weight perfect matching (greedy nearest-neighbor or Blossom V)
4. Add duplicate edges to make all vertices even-degree
5. Find Eulerian circuit (Hierholzer's algorithm)
6. Calculate efficiency metrics and turn statistics

### VRP Solvers
| Solver | Description | Complexity |
|--------|-------------|------------|
| Greedy | Nearest neighbor insertion | O(n²) |
| Clarke-Wright | Savings algorithm | O(n² log n) |
| Sweep | Angular clustering | O(n log n) |
| Two-Opt | Local search improvement | O(n²) per iteration |
| OR-Tools | Google OR-Tools wrapper | Varies |
| Neural-Guided | ML-guided heuristic | O(n²) |
| Neural ONNX | Direct neural inference | O(1) forward pass |
| Neural GNN | RL Agent + 2-Opt Hybrid | O(n²) |

### Turn Classification
- **Straight**: ±45° bearing change
- **Left/Right**: 45-135° bearing change
- **U-turn**: >135° bearing change

### Drone Energy Model
Uses Dorling et al. (2017) physics-based model:
- Power = Pₚ + (Pₗ - Pₚ) × (payload / max_payload)
- Energy (Wh) = (Power × Distance / Speed) / 3600
- Wind impact: ground_speed = (cruise_speed - wind_speed).max(1.0)

---

## Data Sources

### Overture Maps
- **Source**: AWS S3 public bucket (`overturemaps-us-west-2`)
- **Format**: Parquet with WKB geometry
- **Release**: 2026-04-15.0
- **Authentication**: None required
- **Status**: ✅ Fully functional

### OpenStreetMap
- **Format**: PBF (Protocol Buffer Format)
- **Source**: Local `.osm.pbf` files
- **Download**: Geofabrik, BBBike, OSM extracts
- **Status**: ✅ Fully functional

### PMTiles
- **Format**: `.pmtiles` vector tile archives
- **Source**: Local files
- **Layers**: Supports `transportation` and custom layers
- **Zoom**: Automatic or manual selection
- **Status**: ✅ Fully functional

### PostGIS
- **Database**: PostgreSQL with PostGIS extension
- **Table**: `road_edges` (configurable)
- **Columns**: Requires `id`, `source`, `target`, `cost`, `oneway`, `geometry`
- **Status**: ✅ Fully functional

---

## Performance

- **Concurrent S3 Processing**: Up to 10 files in parallel
- **Node Deduplication**: 1-meter precision snapping
- **Binary Format**: ~90% compression vs GeoJSON
- **Graph Algorithms**: O(E log V) for matching, O(E) for Eulerian circuit
- **Neural Inference**: <100ms for CVRP-50 on modern CPU

---

## Requirements

- **Rust**: 1.70+
- **Internet connection**: Required for Overture Maps extraction
- **Local OSM PBF file**: Required for OpenStreetMap extraction

### Optional Dependencies

| Feature | Dependency | Install Command |
|---------|------------|-----------------|
| `extract` | `libgdal-dev` | `sudo apt-get install libgdal-dev` (Ubuntu/Debian) |
| `extract` | `libgdal-dev` | `brew install gdal` (macOS) |
| `extract` | `gdal` | Ensure `gdal` is in `PATH` (Windows via Conda/OSGeo4W) |
| `gui` | `libxcb`, `libx11` | `sudo apt-get install libxcb-render0-dev libx11-dev` |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key | - |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret key | - |
| `R2_ENDPOINT` | Cloudflare R2 endpoint URL | - |
| `DATABASE_URL` | PostgreSQL connection URL | - |

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## Acknowledgments

- [Overture Maps Foundation](https://overturemaps.org/) for open road network data
- [OpenStreetMap](https://www.openstreetmap.org/) contributors for global map data
- [ratatui](https://github.com/ratatui-org/ratatui) for the excellent TUI framework
- [Model Context Protocol](https://modelcontextprotocol.io/) for AI agent integration
- [Cloudflare R2](https://www.cloudflare.com/products/r2/) for cloud storage
- Chinese Postman Problem algorithm based on classical graph theory
- Drone energy model based on Dorling et al. (2017)
- Graph embeddings: node2vec (Grover & Leskovec, 2016), LINE (Tang et al., 2015), FastRP (Chen et al., 2019)

---

## Roadmap

- [x] **v0.1.0**: Overture extraction, OSM PBF extraction, cached maps browser, basic optimization, TUI
- [x] **v0.2.0**: Async/threaded operations, comprehensive tests, performance optimizations
- [x] **v0.3.0**: Multi-depot support, VRP solvers, neural routing
- [x] **v0.4.0**: Time windows, capacity constraints, PMTiles support
- [x] **v0.5.0**: MCP server, PostGIS integration, R2 cloud storage, Drone VRP, OsmAnd/Google Maps links
- [x] **v0.5.2**: Graph embeddings (node2vec, LINE, FastRP, Spatial) — CLI + TUI + Agent
- [x] **v0.5.3**: GNN RL Drone Agent, `ort` v2.0 integration, HF model card
- [ ] **v0.6.0**: Advanced ML features, improved TUI, better documentation
- [ ] **v1.0.0**: Stable API, backward compatibility guarantees

---

## Support

- **Documentation**: [docs.rs/v2rmp](https://docs.rs/v2rmp)
- **Issues**: [GitHub Issues](https://github.com/spacialglaciercom-lab/v2rmp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/spacialglaciercom-lab/v2rmp/discussions)
- **Crates.io**: [crates.io/crates/v2rmp](https://crates.io/crates/v2rmp)
