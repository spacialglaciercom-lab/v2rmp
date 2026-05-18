#!/usr/bin/env python3
"""
v2rmp SFT Training Dataset Builder
===================================
Generates a comprehensive ChatML-formatted JSONL dataset for fine-tuning an LLM
to act as a v2rmp route optimization agent.

Covers:
  1. System prompt defining the v2rmp ecosystem role
  2. MCP tool calling — proper JSON-RPC format
  3. CLI commands — all rmpca subcommands
  4. Agent task execution — task plan format

Usage:
  python v2rmp_dataset_builder.py --output dataset.jsonl --num-examples 2000
"""

import json
import argparse
import random
import os
from typing import List, Dict, Any
from dataclasses import dataclass, field

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert route optimization agent for the v2rmp ecosystem (rmpca CLI). \
You help users extract road networks, compile maps, optimize routes, solve VRP/CPP problems, \
query elevation data, and orchestrate multi-step geospatial pipelines.

Key capabilities:
- **Extract**: Road networks from Overture Maps, OSM PBF, PMTiles, or PostGIS
- **Compile**: GeoJSON → optimized .rmp binary format (~90% compression)
- **Clean**: Repair GeoJSON with deduplication, simplification, snap tolerance, component pruning
- **Optimize**: Chinese Postman Problem (CPP) with turn penalties, or Vehicle Routing Problem (VRP)
- **VRP Solvers**: Greedy, Clarke-Wright Savings, Sweep, Two-Opt, OR-Opt, Neural-Guided, Neural ONNX
- **Elevation**: DEM queries (point, profile, stats, fuel estimation)
- **ML**: Solver selection, quality prediction, hyperparameter tuning, graph embeddings (node2vec, LINE, FastRP, spatial)
- **Cloud**: Cloudflare R2 storage (list, upload, download)
- **Integration**: OsmAnd deep links, Google Maps URLs, GPX export

Binary format (.rmp): [RMP1 magic][u32 nodes][u32 edges][lat+f64 + lon+f64 per node][from_u32 + to_u32 + weight_f64 + oneway_u8 per edge][CRC32]

When the user asks for a route optimization task, determine the correct tool/command, \
provide the exact invocation, and explain the output. For multi-step tasks, provide \
a task plan JSON that can be fed to `rmpca agent --plan tasks.json`.

Always respond with precise command syntax and parameter values. \
Use --json flag for machine-readable output in CLI commands.
"""

SYSTEM_PROMPT_MCP = """\
You are an expert route optimization agent connected to the v2rmp MCP server. \
You help users by calling the appropriate MCP tools to extract road networks, \
compile maps, optimize routes, solve VRP/CPP problems, query elevation data, \
and orchestrate multi-step geospatial pipelines.

Available MCP tools:
- extract_overture: Extract from Overture Maps S3
- extract_osm: Extract from OSM PBF/Overpass
- compile: GeoJSON → .rmp binary
- optimize: CPP/VRP route optimization on .rmp
- clean: Full GeoJSON cleaning pipeline
- vrp_solve: Dedicated VRP solver with stops/depots
- elevation_query: Point elevation from DEM
- elevation_profile: Route elevation profile
- elevation_stats: Bbox elevation statistics
- dem_info: DEM file metadata
- fuel_estimate: Fuel consumption from elevation
- inspect_rmp: Inspect .rmp binary metadata
- list_solvers: List available VRP solvers
- pipeline: End-to-end extract→clean→compile→optimize
- haversine_distance: Great-circle distance
- get_valhalla_matrix: Real-road distance matrix
- predict_solver: ML solver recommendation
- predict_quality: Pre-solve quality prediction
- score_route: Multi-dimensional route scoring
- route_embedding: 12-dim route feature vector
- tune_hyperparams: AutoML hyperparameter tuning
- submit_feedback: ML feedback loop
- parse_routing_query: NLP → VRP config
- graph_embed: Node/edge graph embeddings
- generate_osmand_link: OsmAnd deep link

Respond with the exact JSON-RPC tool call when a user requests an action. \
Format: {"jsonrpc": "2.0", "id": <int>, "method": "tools/call", "params": {"name": "<tool>", "arguments": {...}}}
"""

# ─── Data Templates ───────────────────────────────────────────────────────────

@dataclass
class ConversationExample:
    """A single training example in ChatML format."""
    messages: List[Dict[str, str]]

    def to_jsonl(self) -> str:
        return json.dumps({"messages": self.messages}, ensure_ascii=False)


def make_system_message(content: str) -> Dict[str, str]:
    return {"role": "system", "content": content}

def make_user_message(content: str) -> Dict[str, str]:
    return {"role": "user", "content": content}

def make_assistant_message(content: str) -> Dict[str, str]:
    return {"role": "assistant", "content": content}


# ─── 1. CLI Command Examples ─────────────────────────────────────────────────

CLI_EXAMPLES: List[Dict] = [
    # Extract
    {
        "user": "Extract road network data for Montreal downtown from Overture Maps",
        "assistant": """\
To extract road network data for Montreal downtown from Overture Maps, use:

```bash
rmpca extract --source overture --bbox "-73.59,45.49,-73.55,45.52" --output montreal_downtown.geojson
```

This will:
- Connect to the Overture Maps AWS S3 public bucket
- Download road segments within the bounding box
- Write a GeoJSON file with all vehicle-accessible roads

Parameters:
- `--source overture`: Use Overture Maps (default)
- `--bbox "MIN_LON,MIN_LAT,MAX_LON,MAX_LAT"`: Bounding box in WGS-84
- `--output`: Output GeoJSON file path (default: extract-output.geojson)

For JSON output (agent consumption):
```bash
rmpca --json extract --source overture --bbox "-73.59,45.49,-73.55,45.52" --output montreal_downtown.geojson
```"""
    },
    {
        "user": "How do I extract from a local OSM PBF file?",
        "assistant": """\
To extract from a local OSM PBF file:

```bash
rmpca extract --source osm --pbf-path data.osm.pbf --bbox "-73.59,45.49,-73.55,45.52" --output montreal.geojson
```

Parameters:
- `--source osm`: Use OpenStreetMap source
- `--pbf-path`: Path to the local `.osm.pbf` file (required for OSM source)
- `--bbox`: Bounding box in WGS-84 coordinates

You can download PBF files from:
- Geofabrik: https://download.geofabrik.de/
- BBBike: https://extract.bbbike.org/
- OSM extracts: https://osm-pts.download"""

    },
    {
        "user": "Extract roads from a PMTiles archive",
        "assistant": """\
Use `rmpca-extract` for PMTiles extraction:

```bash
rmpca-extract --pmtiles-path montreal.pmtiles --bbox "-73.59,45.49,-73.55,45.52" --output montreal_pmtiles.geojson
```

Optional parameters:
- `--layer-name transportation`: Filter to specific layer
- `--zoom 14`: Set zoom level (auto-detected by default)

The PMTiles extractor reads vector tiles directly from the archive without a tile server."""
    },
    {
        "user": "Extract from PostGIS database",
        "assistant": """\
To extract from a PostGIS database:

```bash
rmpca-extract --postgis --bbox "-73.59,45.49,-73.55,45.52" --database-url postgresql://user:pass@host/db
```

The database table (default: `road_edges`) must have columns:
- `id`, `source`, `target`, `cost`, `oneway`, `geometry`

You can configure the table name:
```bash
rmpca-extract --postgis --bbox "..." --database-url postgresql://... --table-name my_roads
```"""
    },
    # Compile
    {
        "user": "Compile my GeoJSON file to binary format",
        "assistant": """\
To compile a GeoJSON file to the optimized .rmp binary format:

```bash
rmpca compile --input montreal.geojson --output montreal.rmp
```

This produces a compact binary file (~90% compression vs GeoJSON) with:
- CRC32 integrity checking
- Node deduplication at 1-meter precision

Options:
- `--clean`: Run cleaning pipeline before compilation
- `--prune-disconnected`: Keep only the largest connected component

```bash
rmpca compile --input montreal.geojson --output montreal.rmp --clean --prune-disconnected
```"""
    },
    # Clean
    {
        "user": "Clean a GeoJSON road network file",
        "assistant": """\
To clean a GeoJSON road network:

```bash
rmpca clean --input raw.geojson --output cleaned.geojson
```

Available cleaning options:
- `--min-length-m 0.5`: Remove edges shorter than 0.5 meters
- `--node-snap-m 2.0`: Snap nodes within 2 meters
- `--max-components 1`: Keep only the largest connected component
- `--simplify-tolerance-m 0.5`: Simplify geometry with 0.5m tolerance
- `--no-dedupe`: Skip edge deduplication
- `--no-remove-isolates`: Skip isolated node removal

Full example with all options:
```bash
rmpca clean --input raw.geojson --output cleaned.geojson \\
  --min-length-m 0.5 --node-snap-m 2.0 --max-components 1 \\
  --simplify-tolerance-m 0.5
```"""
    },
    # Optimize
    {
        "user": "Run CPP route optimization on my map",
        "assistant": """\
To optimize a route using the Chinese Postman Problem (CPP) solver:

```bash
rmpca optimize --input montreal.rmp --output route.json \\
  --left-penalty 1.0 --right-penalty 0.0 --uturn-penalty 5.0
```

Parameters:
- `--input`: Path to compiled .rmp file
- `--output`: Output route file path
- `--left-penalty`: Left turn penalty in meters (default: 1.0)
- `--right-penalty`: Right turn penalty in meters (default: 0.0)
- `--uturn-penalty`: U-turn penalty in meters (default: 5.0)
- `--depot`: Depot coordinates as LAT,LON
- `--oneway`: Oneway handling: respect|ignore|reverse (default: respect)

With depot and Google Maps link:
```bash
rmpca optimize --input montreal.rmp --output route.gpx \\
  --depot 45.505,-73.565 --google-maps \\
  --osmand-base-url https://pub-xxx.r2.dev
```"""
    },
    {
        "user": "I need to solve a VRP with 5 vehicles",
        "assistant": """\
To solve a Vehicle Routing Problem with multiple vehicles:

```bash
rmpca vrp --input montreal.rmp --output-dir routes/ \\
  --vehicles 5 --capacity 1000 \\
  --depot 45.505,-73.565 --coordinates stops.csv \\
  --algo savings --google-maps
```

Parameters:
- `--vehicles`: Number of vehicles (default: 1)
- `--capacity`: Vehicle capacity constraint
- `--depot`: Depot coordinates (LAT,LON), can specify multiple
- `--coordinates`: CSV file with stop coordinates (columns: lat,lon[, label, demand, type])
- `--algo`: Algorithm: greedy|savings|local-search|simulated-annealing (default: greedy)

Algorithms:
- `greedy`: Nearest neighbor insertion — O(n²)
- `savings`: Clarke-Wright Savings — O(n² log n)
- `local-search`: Two-Opt local search — O(n²) per iteration
- `simulated-annealing`: OR-Opt metaheuristic

Output: One GPX file per vehicle in the output directory."""
    },
    # Pipeline
    {
        "user": "Run the full pipeline from extraction to optimization",
        "assistant": """\
To run the full end-to-end pipeline (extract → clean → compile → optimize):

```bash
rmpca pipeline --bbox "-73.59,45.49,-73.55,45.52" --output-dir ./output \\
  --depot 45.505,-73.565 --prune-disconnected
```

This executes 4 stages:
1. **Extract**: Downloads road network from Overture Maps
2. **Clean**: Runs cleaning pipeline with default options
3. **Compile**: Converts cleaned GeoJSON to .rmp binary
4. **Optimize**: Runs CPP optimization on the compiled map

For OSM source:
```bash
rmpca pipeline --bbox "..." --source osm --pbf-path data.osm.pbf --output-dir ./output
```

Output files in `./output/`:
- `extract.geojson`: Raw extracted data
- `cleaned.geojson`: Cleaned data
- `network.rmp`: Compiled binary
- `route.gpx`: Optimized route"""
    },
    # Elevation
    {
        "user": "Query elevation at a specific point",
        "assistant": """\
To query elevation at a specific point from a DEM GeoTIFF:

```bash
rmpca elevation point --dem elevation.tif --lon -73.565 --lat 45.505
```

Other elevation subcommands:
- `points`: Query multiple points from a JSON file
- `profile`: Get elevation profile along a route
- `stats`: Elevation statistics for a bounding box
- `info`: DEM file metadata
- `fuel`: Calculate fuel consumption from a route profile

Example — elevation profile:
```bash
rmpca elevation profile --dem elevation.tif --input route.json --interval 10.0
```

Example — fuel estimation:
```bash
rmpca elevation fuel --dem elevation.tif --input route.json --base-consumption 0.35
```"""
    },
    # ML commands
    {
        "user": "Predict the best VRP solver for my stops",
        "assistant": """\
To predict the best VRP solver using the ML model:

```bash
rmpca predict-solver --input stops.json --json
```

The input JSON should be an array of stops (first = depot):
```json
[
  {"lat": 45.505, "lon": -73.565, "label": "Depot"},
  {"lat": 45.508, "lon": -73.554, "label": "Stop 1", "demand": 1.0}
]
```

Output includes recommended solver, confidence, and all solver scores.

Related ML commands:
- `rmpca predict-quality --input stops.json --json`: Predict route quality before solving
- `rmpca tune-hyperparams --input stops.json --json`: Get AutoML hyperparameters"""
    },
    {
        "user": "Generate graph embeddings for my road network",
        "assistant": """\
To generate graph embeddings from a .rmp file:

```bash
rmpca graph-embed -i montreal.rmp -m fastrp -d 64 -o embeddings.json
```

Available methods:
- `node2vec`: Biased random walks + Skip-gram (~25s for 14K nodes)
- `line`: 1st + 2nd order proximity preservation (~3s)
- `fastrp`: Sparse random projection, fastest (<1s)
- `spatial`: Handcrafted features: degree, centrality, clustering (~2s)

Options:
- `-d 64`: Embedding dimensions (default: 64)
- `--include-edges`: Derive edge embeddings via Hadamard product
- `--walk-length 30`: Walk length for node2vec
- `--num-walks 10`: Walks per node for node2vec
- `--p 1.0 --q 1.0`: Return/in-out parameters for node2vec
- `--lr 0.025`: Learning rate
- `--epochs 5`: Training epochs for LINE

Example with node2vec:
```bash
rmpca graph-embed -i montreal.rmp -m node2vec -d 128 \\
  --walk-length 20 --num-walks 5 --p 0.5 --q 2.0 --include-edges
```"""
    },
    # List
    {
        "user": "List available map files",
        "assistant": """\
To list available resources:

```bash
# List compiled .rmp maps
rmpca list maps

# List route files (.gpx, .geojson)
rmpca list routes
```

Scans the current directory for matching file extensions."""
    },
    # Serve
    {
        "user": "Start the headless server for automation",
        "assistant": """\
To start the headless JSON-RPC/STDIO server:

```bash
rmpca serve
```

The server reads JSON task objects from stdin (one per line) and writes JSON results to stdout.

Each line is a task object with a `type` field:
```json
{"type": "compile", "input": "map.geojson", "output": "map.rmp"}
```

Supported task types: extract, compile, clean, optimize, vrp, pipeline, embed, graph-embed, elevation.

For MCP server mode (AI agent integration):
```bash
rmpca-mcp
```

Legacy MCP server with all tools:
```bash
rmpca-mcp-server-legacy
```"""
    },
    # Parse query
    {
        "user": "Parse a natural language routing query",
        "assistant": """\
To parse a natural language routing query into structured JSON:

```bash
rmpca parse-query --query "optimize route from Montreal to Toronto with 3 stops" --json
```

Output is a structured VRP configuration object that can be fed to the VRP solver."""
    },
    # Embed
    {
        "user": "Generate text embeddings",
        "assistant": """\
To generate BERT embeddings for text strings:

```bash
rmpca embed --text "route optimization" --text "vehicle routing" --text "road network"
```

Requires the `ml` feature flag. Returns embedding vectors for each input text.

For graph-level embeddings from .rmp files, use `rmpca graph-embed` instead."""
    },
]


# ─── 2. MCP Tool Calling Examples ────────────────────────────────────────────

MCP_TOOL_EXAMPLES: List[Dict] = [
    # extract_overture
    {
        "user": "I need to extract roads from Overture Maps for the Plateau area in Montreal",
        "assistant": """\
I'll call the `extract_overture` MCP tool to extract the road network for Montreal's Plateau area:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "extract_overture", "arguments": {"bbox": {"min_lon": -73.59, "min_lat": 45.51, "max_lon": -73.56, "max_lat": 45.53}, "output_path": "plateau.geojson"}}}
```

This will:
1. Connect to the Overture Maps S3 public bucket (no auth required)
2. Download Parquet files containing road segments in the bounding box
3. Parse WKB geometry and write a GeoJSON file

The bounding box covers roughly: St-Laurent to Papineau, Sherbrooke to Mont-Royal.

For specific road classes only (e.g., residential and tertiary):
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "extract_overture", "arguments": {"bbox": {"min_lon": -73.59, "min_lat": 45.51, "max_lon": -73.56, "max_lat": 45.53}, "road_classes": ["residential", "tertiary", "secondary"], "output_path": "plateau.geojson"}}}
```"""
    },
    # extract_osm
    {
        "user": "Extract from my local OSM PBF for Ottawa",
        "assistant": """\
I'll extract roads from your local OSM PBF file:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "extract_osm", "arguments": {"bbox": {"min_lon": -75.80, "min_lat": 45.35, "max_lon": -75.60, "max_lat": 45.50}, "pbf_path": "ottawa.osm.pbf", "output_path": "ottawa.geojson"}}}
```

If no PBF file is provided, it falls back to the Overpass API:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "extract_osm", "arguments": {"bbox": {"min_lon": -75.80, "min_lat": 45.35, "max_lon": -75.60, "max_lat": 45.50}, "output_path": "ottawa.geojson"}}}
```"""
    },
    # compile
    {
        "user": "Compile the extracted GeoJSON into binary format",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "compile", "arguments": {"input": "plateau.geojson", "output": "plateau.rmp", "clean": true, "prune_disconnected": true}}}
```

This will:
1. Run the cleaning pipeline on the GeoJSON (deduplication, snap, simplify)
2. Prune disconnected subgraphs, keeping the largest component
3. Compile to binary .rmp format (~90% compression)

The result includes: node_count, edge_count, input/output size, compression ratio, and elapsed time."""
    },
    # optimize (CPP)
    {
        "user": "Optimize a route on the Plateau map with turn penalties",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "optimize", "arguments": {"input": "plateau.rmp", "output": "plateau_route.gpx", "mode": "cpp", "oneway_mode": "respect", "left_penalty": 1.0, "right_penalty": 0.0, "uturn_penalty": 5.0, "depot": {"lat": 45.5200, "lon": -73.5750}}}}
```

Parameters explained:
- `mode: "cpp"`: Chinese Postman Problem — covers every edge (road segment)
- `oneway_mode: "respect"`: Honor one-way street restrictions
- Turn penalties: Left turns cost 1.0m extra, U-turns cost 5.0m extra
- `depot`: Starting/ending point (snapped to nearest node)

The result includes: total_distance_km, total_segments, num_routes, turn statistics, and per-route coordinate arrays."""
    },
    # optimize (VRP mode)
    {
        "user": "Solve a VRP on my compiled map with 3 vehicles",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "optimize", "arguments": {"input": "montreal.rmp", "mode": "vrp", "vehicles": 3, "solver": "clarke_wright", "depot": {"lat": 45.505, "lon": -73.565}}}}
```

Alternatively, use the dedicated `vrp_solve` tool for explicit stop coordinates:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "vrp_solve", "arguments": {"stops": [{"lat": 45.505, "lon": -73.565, "label": "Depot", "demand": 0.0}, {"lat": 45.508, "lon": -73.554, "label": "Stop 1", "demand": 1.0}, {"lat": 45.495, "lon": -73.578, "label": "Stop 2", "demand": 1.0}, {"lat": 45.512, "lon": -73.555, "label": "Stop 3", "demand": 1.0}], "num_vehicles": 3, "vehicle_capacity": 100.0, "solver": "clarke_wright"}}}
```

Available VRP solvers: `default`, `clarke_wright`, `sweep`, `two_opt`, `or_opt`, `neural_guided`."""
    },
    # clean
    {
        "user": "Clean a GeoJSON with specific parameters",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "clean", "arguments": {"input": "raw.geojson", "output": "cleaned.geojson", "min_length_m": 0.5, "node_snap_m": 2.0, "max_components": 1, "simplify_tolerance_m": 0.5, "dedupe_edges": true, "remove_isolates": true, "merge_parallel_edges": true, "remove_selfloops": true}}}
```

This provides fine-grained control over all 17+ cleaning parameters. The result includes statistics on features removed, nodes snapped, edges merged, etc."""
    },
    # elevation_query
    {
        "user": "What's the elevation at coordinates 45.505, -73.565?",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "elevation_query", "arguments": {"dem_path": "elevation.tif", "points": [{"lon": -73.565, "lat": 45.505}]}}}
```

Returns elevation in meters, or null if the point has no DEM coverage. For multiple points:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "elevation_query", "arguments": {"dem_path": "elevation.tif", "points": [{"lon": -73.565, "lat": 45.505}, {"lon": -73.554, "lat": 45.508}, {"lon": -73.578, "lat": 45.495}]}}}
```"""
    },
    # elevation_profile
    {
        "user": "Get the elevation profile along my route",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "elevation_profile", "arguments": {"dem_path": "elevation.tif", "route": [{"lon": -73.565, "lat": 45.505}, {"lon": -73.554, "lat": 45.508}, {"lon": -73.578, "lat": 45.495}], "sample_interval_m": 10.0}}}
```

Returns: total distance (km), total ascent/descent (m), min/max/avg elevation, and per-sample point data."""
    },
    # inspect_rmp
    {
        "user": "What's in my .rmp file?",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "inspect_rmp", "arguments": {"path": "montreal.rmp"}}}
```

This parses the binary header and returns:
- Node count and edge count
- Bounding box (min/max lat/lon)
- File size and CRC32 integrity status

Without running any optimization."""
    },
    # pipeline
    {
        "user": "Run the full pipeline: extract, clean, compile, and optimize",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "pipeline", "arguments": {"bbox": {"min_lon": -73.59, "min_lat": 45.49, "max_lon": -73.55, "max_lat": 45.52}, "depot": {"lat": 45.505, "lon": -73.565}, "prune_disconnected": true}}}
```

This executes all 4 stages in sequence with a 30-second timeout protection:
1. Extract from Overture Maps
2. Clean with default options
3. Compile to .rmp binary
4. Optimize with CPP solver

For OSM source, add: `"source": "osm", "pbf_path": "data.osm.pbf"`"""
    },
    # predict_solver
    {
        "user": "Which VRP solver should I use for 50 stops?",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "predict_solver", "arguments": {"stops": [{"lat": 45.505, "lon": -73.565, "label": "Depot"}, {"lat": 45.508, "lon": -73.554, "label": "Stop 1"}, {"lat": 45.512, "lon": -73.555, "label": "Stop 2"}], "num_vehicles": 5}}}
```

The ML model analyzes instance features (num stops, spread, density, etc.) and returns:
- `recommended`: Best solver ID
- `confidence`: Prediction confidence (0-1)
- `runner_up`: Second-best solver
- `all_scores`: Scores for all available solvers"""
    },
    # haversine_distance
    {
        "user": "Calculate the distance between Montreal and Quebec City",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "haversine_distance", "arguments": {"from": {"lat": 45.5017, "lon": -73.5673}, "to": {"lat": 46.8139, "lon": -71.2080}}}}
```

Returns the great-circle distance in meters using the Haversine formula."""
    },
    # score_route
    {
        "user": "Score the quality of my route",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "score_route", "arguments": {"route": [{"lat": 45.505, "lon": -73.565}, {"lat": 45.508, "lon": -73.554}, {"lat": 45.512, "lon": -73.555}], "stops": [{"lat": 45.505, "lon": -73.565}, {"lat": 45.508, "lon": -73.554}]}}}}
```

Returns a multi-dimensional quality score including:
- Total distance and efficiency
- Coverage ratio (stops visited / total)
- Route compactness
- Turn complexity"""
    },
    # graph_embed
    {
        "user": "Generate graph embeddings for my road network using FastRP",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "graph_embed", "arguments": {"map_path": "montreal.rmp", "method": "fastrp", "dimensions": 64, "include_edges": true}}}
```

Available methods and approximate speed (14K node network):
- `fastrp`: <1s — Sparse random projection
- `spatial`: ~2s — Handcrafted structural features
- `line`: ~3s — 1st + 2nd order proximity
- `node2vec`: ~25s — Biased random walks + Skip-gram

For node2vec with custom parameters:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "graph_embed", "arguments": {"map_path": "montreal.rmp", "method": "node2vec", "dimensions": 128, "walk_length": 20, "num_walks": 5, "p": 0.5, "q": 2.0}}}}
```"""
    },
    # fuel_estimate
    {
        "user": "Estimate fuel consumption for a hilly route",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "fuel_estimate", "arguments": {"dem_path": "elevation.tif", "route": [{"lon": -73.565, "lat": 45.505}, {"lon": -73.554, "lat": 45.508}, {"lon": -73.578, "lat": 45.495}], "base_consumption_l_per_km": 0.35}}}
```

Uses the Dorling et al. (2017) physics-based model:
- Calculates grade adjustments from the elevation profile
- Returns: total fuel (L), avg consumption (L/km), elevation penalty/benefit"""
    },
    # generate_osmand_link
    {
        "user": "Create an OsmAnd deep link for importing my GPX",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "generate_osmand_link", "arguments": {"gpx_url": "https://pub-abc123.r2.dev/route.gpx"}}}
```

Returns: `osmand://import?url=https%3A%2F%2Fpub-abc123.r2.dev%2Froute.gpx`

This link can be opened on a mobile device with OsmAnd installed to directly import the GPX route."""
    },
    # submit_feedback
    {
        "user": "Submit feedback that the neural solver performed poorly",
        "assistant": """\
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "submit_feedback", "arguments": {"solver_used": "neural_guided", "num_stops": 50, "actual_distance_km": 142.5, "predicted_distance_km": 128.0, "satisfaction": "poor", "notes": "Route had excessive backtracking on the eastern segment"}}}
```

Feedback is logged for online learning and future AutoML improvements."""
    },
]


# ─── 3. Agent Task Plan Examples ──────────────────────────────────────────────

AGENT_TASK_EXAMPLES: List[Dict] = [
    {
        "user": "Create a task plan to extract and optimize Mile End Montreal",
        "assistant": """\
Here's a multi-step task plan for extracting and optimizing Mile End:

```json
[
  {
    "type": "extract",
    "bbox": "-73.590,45.520,-73.565,45.535",
    "source": "overture",
    "output": "mile_end.geojson"
  },
  {
    "type": "clean",
    "input": "mile_end.geojson",
    "output": "mile_end_clean.geojson",
    "min_length_m": 0.5,
    "node_snap_m": 1.0,
    "max_components": 1,
    "simplify_tolerance_m": 0.5
  },
  {
    "type": "compile",
    "input": "mile_end_clean.geojson",
    "output": "mile_end.rmp",
    "prune_disconnected": true
  },
  {
    "type": "optimize",
    "input": "mile_end.rmp",
    "output": "mile_end_route.gpx",
    "mode": "cpp",
    "depot": [45.5275, -73.5775],
    "oneway": "respect",
    "left_penalty": 1.0,
    "right_penalty": 0.0,
    "uturn_penalty": 5.0
  }
]
```

Execute with:
```bash
rmpca agent --plan mile_end_plan.json --json
```

Each step runs sequentially and the JSON output includes results from all stages."""
    },
    {
        "user": "Create a VRP task plan for delivery routes",
        "assistant": """\
Here's a task plan for extracting a network and solving a VRP for delivery routes:

```json
[
  {
    "type": "extract",
    "bbox": "-73.650,45.450,-73.550,45.550",
    "source": "overture",
    "output": "montreal_central.geojson"
  },
  {
    "type": "compile",
    "input": "montreal_central.geojson",
    "output": "montreal_central.rmp",
    "clean": true,
    "prune_disconnected": true
  },
  {
    "type": "vrp",
    "input": "montreal_central.rmp",
    "output_dir": "delivery_routes/",
    "vehicles": 5,
    "algo": "savings",
    "capacity": 500.0,
    "depot": ["45.505,-73.565"],
    "coordinates": "delivery_stops.csv",
    "google_maps": true,
    "osmand_base_url": "https://pub-xxx.r2.dev"
  }
]
```

The CSV file should have columns: `lat,lon` (required), `label,demand,type` (optional).

This will produce 5 GPX route files in `delivery_routes/` plus Google Maps links."""
    },
    {
        "user": "Build a task plan for elevation-aware route optimization",
        "assistant": """\
Here's a plan that incorporates elevation data into route planning:

```json
[
  {
    "type": "extract",
    "bbox": "-73.620,45.480,-73.560,45.530",
    "source": "overture",
    "output": "plateau_montroyal.geojson"
  },
  {
    "type": "compile",
    "input": "plateau_montroyal.geojson",
    "output": "plateau_montroyal.rmp",
    "clean": true
  },
  {
    "type": "optimize",
    "input": "plateau_montroyal.rmp",
    "output": "plateau_route.gpx",
    "mode": "cpp",
    "left_penalty": 2.0,
    "uturn_penalty": 10.0,
    "depot": [45.5150, -73.5800]
  }
]
```

After running, query elevation on the result:
```bash
rmpca elevation profile --dem elevation.tif --input plateau_route.gpx --interval 10.0
rmpca elevation fuel --dem elevation.tif --input plateau_route.gpx --base-consumption 0.35
```

The higher turn penalties help avoid steep turns on Mount Royal's slopes."""
    },
    {
        "user": "Write an agent task to generate graph embeddings",
        "assistant": """\
```json
{
  "type": "graph_embed",
  "input": "montreal.rmp",
  "method": "node2vec",
  "dim": 64,
  "walk_length": 30,
  "num_walks": 10,
  "include_edges": true,
  "output": "montreal_embeddings.json"
}
```

Execute with:
```bash
rmpca agent --plan embed_task.json --json
```

Or directly via CLI:
```bash
rmpca graph-embed -i montreal.rmp -m node2vec -d 64 --include-edges -o montreal_embeddings.json
```"""
    },
]


# ─── 4. Multi-turn Conversation Examples ──────────────────────────────────────

MULTI_TURN_EXAMPLES: List[List[Dict]] = [
    [
        {"role": "user", "content": "I have a GeoJSON file of Montreal roads. How do I optimize a route?"},
        {"role": "assistant", "content": """\
Great! Here's the workflow:

**Step 1: Compile to binary format**
```bash
rmpca compile --input montreal.geojson --output montreal.rmp --clean --prune-disconnected
```

**Step 2: Optimize the route**
```bash
rmpca optimize --input montreal.rmp --output route.gpx --left-penalty 1.0 --uturn-penalty 5.0
```

The `--clean` flag runs the cleaning pipeline before compilation, and `--prune-disconnected` ensures only the main connected road network is used.

Would you like to specify a depot location or adjust turn penalties?"""},
        {"role": "user", "content": "Yes, set depot at 45.505, -73.565 and increase the U-turn penalty to 10"},
        {"role": "assistant", "content": """\
Updated command:

```bash
rmpca optimize --input montreal.rmp --output route.gpx \
  --depot 45.505,-73.565 \
  --left-penalty 1.0 --right-penalty 0.0 --uturn-penalty 10.0
```

The higher U-turn penalty (10.0) will discourage the solver from making U-turns, which is useful for:
- Safety in urban areas
- Avoiding left turns across traffic
- Following natural traffic flow patterns

The depot coordinates will be snapped to the nearest road network node. The route will start and end at this location."""},
    ],
    [
        {"role": "user", "content": "I need to plan delivery routes for 8 vehicles serving 100 stops across Montreal"},
        {"role": "assistant", "content": """\
For 100 stops with 8 vehicles, I recommend the Clarke-Wright Savings algorithm which scales well:

**Prepare your stops CSV** (`delivery_stops.csv`):
```csv
lat,lon,label,demand
45.505,-73.565,Depot,0
45.508,-73.554,Customer 1,5.0
45.495,-73.578,Customer 2,3.0
...
```

**Compile the road network:**
```bash
rmpca compile --input montreal.geojson --output montreal.rmp --clean --prune-disconnected
```

**Solve the VRP:**
```bash
rmpca vrp --input montreal.rmp --output-dir routes/ \
  --vehicles 8 --capacity 500 \
  --depot 45.505,-73.565 \
  --coordinates delivery_stops.csv \
  --algo savings --google-maps
```

For even better results, let the ML solver selector pick the best algorithm:
```bash
rmpca predict-solver --input stops.json --json
```

Would you like me to also set up OsmAnd links for mobile navigation?"""},
        {"role": "user", "content": "Yes, use https://pub-mybucket.r2.dev as the base URL"},
        {"role": "assistant", "content": """\
Complete command with OsmAnd deep links:

```bash
rmpca vrp --input montreal.rmp --output-dir routes/ \
  --vehicles 8 --capacity 500 \
  --depot 45.505,-73.565 \
  --coordinates delivery_stops.csv \
  --algo savings --google-maps \
  --osmand-base-url https://pub-mybucket.r2.dev
```

This will generate for each vehicle:
- `routes/vehicle_1.gpx` through `routes/vehicle_8.gpx`
- Google Maps URLs (auto-sampled to ≤20 waypoints)
- OsmAnd deep links like: `osmand://import?url=https%3A%2F%2Fpub-mybucket.r2.dev%2Fvehicle_1.gpx`

Drivers can open the OsmAnd links on their phones to get turn-by-turn navigation.

To upload the routes to R2 for sharing:
```bash
rmpca-mcp-server-legacy  # or configure rmpca-mcp
```
Then use the `upload_to_r2` tool for each GPX file."""},
    ],
    [
        {"role": "user", "content": "What's the difference between CPP and VRP optimization modes?"},
        {"role": "assistant", "content": """\
**Chinese Postman Problem (CPP)** and **Vehicle Routing Problem (VRP)** solve different problems:

| Aspect | CPP | VRP |
|--------|-----|-----|
| Goal | Traverse **every road segment** | Visit **specific stops** |
| Use case | Snow plowing, street sweeping, mail delivery | Package delivery, food delivery, logistics |
| Input | .rmp file (road network) | .rmp + stop coordinates |
| Output | Single route covering all edges | Multiple routes assigned to vehicles |
| Vehicles | 1 (typically) | 1 to N |

**When to use CPP:**
- You need to cover every street (municipal services)
- Route inspection, pothole scanning, garbage collection

**When to use VRP:**
- You have specific delivery locations
- Multiple vehicles with capacity constraints
- Need to minimize total distance while serving all stops

Set the mode with `rmpca optimize --mode cpp` or `--mode vrp`."""},
    ],
]


# ─── 5. Error Handling / Troubleshooting Examples ─────────────────────────────

TROUBLESHOOTING_EXAMPLES: List[Dict] = [
    {
        "user": "My compilation failed with 'No valid features found'",
        "assistant": """\
This typically means the GeoJSON has no LineString features (road segments). Common causes:

1. **Wrong file type**: Ensure it's a road network GeoJSON, not points or polygons
2. **Empty extraction**: The bounding box might not contain any roads

**Debug steps:**
```bash
# Check what's in your GeoJSON
python3 -c "import json; d=json.load(open('your.geojson')); print(f'Features: {len(d[\"features\"])}'); print([f['geometry']['type'] for f in d['features'][:5]])"
```

3. **Try cleaning first**:
```bash
rmpca clean --input your.geojson --output cleaned.geojson --max-components 1
```

4. **Check with inspect_rmp** (if you have a compiled file):
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "inspect_rmp", "arguments": {"path": "your.rmp"}}}
```"""
    },
    {
        "user": "The VRP solver gives poor results. How can I improve them?",
        "assistant": """\
Several strategies to improve VRP results:

**1. Use ML solver selection:**
```bash
rmpca predict-solver --input stops.json --json
```

**2. Try different algorithms:**
```bash
# Savings (Clarke-Wright) — good for clustered stops
rmpca vrp ... --algo savings

# Local search (Two-Opt) — good for refinement
rmpca vrp ... --algo local-search
```

**3. Tune hyperparameters with AutoML:**
```bash
rmpca tune-hyperparams --input stops.json --json
```

**4. Check predicted quality before solving:**
```bash
rmpca predict-quality --input stops.json --json
```

**5. Submit feedback for improvement:**
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "submit_feedback", "arguments": {"solver_used": "clarke_wright", "num_stops": 50, "actual_distance_km": 142.5, "satisfaction": "poor"}}}
```

**6. Use real-road distances** instead of haversine:
```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_valhalla_matrix", "arguments": {"stops": [...]}}}
```"""
    },
    {
        "user": "How do I fix 'Path traversal denied' error?",
        "assistant": """\
This security error occurs when file paths go outside allowed directories. Make sure:

1. **Use relative paths** from the working directory:
   - ✅ `./montreal.geojson`
   - ✅ `data/maps/montreal.rmp`
   - ❌ `/etc/passwd`
   - ❌ `../../secret/data`

2. **Input/output files should be in the current directory tree**

3. **For MCP server**: The path validation canonicalizes and checks that files are within the server's allowed directory. Place your data files in the same directory where the server runs."""
    },
]


# ─── Dataset Builder ──────────────────────────────────────────────────────────

class V2RMPDatasetBuilder:
    """Builds the complete training dataset."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.examples: List[ConversationExample] = []

    def build(self) -> List[ConversationExample]:
        """Build all examples and return them."""
        self._build_cli_examples()
        self._build_mcp_tool_examples()
        self._build_agent_task_examples()
        self._build_multi_turn_examples()
        self._build_troubleshooting_examples()
        self._build_variations()
        return self.examples

    def _add(self, messages: List[Dict[str, str]]):
        self.examples.append(ConversationExample(messages=messages))

    def _build_cli_examples(self):
        """Section 1: CLI command knowledge."""
        for ex in CLI_EXAMPLES:
            self._add([
                make_system_message(SYSTEM_PROMPT),
                make_user_message(ex["user"]),
                make_assistant_message(ex["assistant"]),
            ])

    def _build_mcp_tool_examples(self):
        """Section 2: MCP tool calling."""
        for ex in MCP_TOOL_EXAMPLES:
            self._add([
                make_system_message(SYSTEM_PROMPT_MCP),
                make_user_message(ex["user"]),
                make_assistant_message(ex["assistant"]),
            ])

    def _build_agent_task_examples(self):
        """Section 3: Agent task plan execution."""
        for ex in AGENT_TASK_EXAMPLES:
            self._add([
                make_system_message(SYSTEM_PROMPT),
                make_user_message(ex["user"]),
                make_assistant_message(ex["assistant"]),
            ])

    def _build_multi_turn_examples(self):
        """Section 4: Multi-turn conversations."""
        for turns in MULTI_TURN_EXAMPLES:
            messages = [make_system_message(SYSTEM_PROMPT)]
            messages.extend(turns)
            self._add(messages)

    def _build_troubleshooting_examples(self):
        """Section 5: Error handling and debugging."""
        for ex in TROUBLESHOOTING_EXAMPLES:
            self._add([
                make_system_message(SYSTEM_PROMPT),
                make_user_message(ex["user"]),
                make_assistant_message(ex["assistant"]),
            ])

    def _build_variations(self):
        """Generate paraphrased variations for robustness."""
        # Short-form requests mapped to key commands
        short_forms = [
            ("extract roads", "To extract road network data:\n\n```bash\nrmpca extract --source overture --bbox \"MIN_LON,MIN_LAT,MAX_LON,MAX_LAT\" --output roads.geojson\n```\n\nReplace the bbox coordinates with your area of interest."),
            ("compile my map", "```bash\nrmpca compile --input roads.geojson --output map.rmp --clean --prune-disconnected\n```"),
            ("optimize a route", "```bash\nrmpca optimize --input map.rmp --output route.gpx --left-penalty 1.0 --uturn-penalty 5.0\n```"),
            ("clean geojson", "```bash\nrmpca clean --input raw.geojson --output cleaned.geojson --min-length-m 0.5 --node-snap-m 2.0 --max-components 1\n```"),
            ("solve vrp", "```bash\nrmpca vrp --input map.rmp --output-dir routes/ --vehicles 5 --capacity 1000 --depot 45.505,-73.565 --coordinates stops.csv --algo savings\n```"),
            ("list maps", "```bash\nrmpca list maps\n```\n\nLists all `.rmp` files in the current directory."),
            ("check elevation", "```bash\nrmpca elevation point --dem elevation.tif --lon -73.565 --lat 45.505\n```"),
            ("best vrp solver", "```bash\nrmpca predict-solver --input stops.json --json\n```\n\nUses ML to recommend the optimal VRP solver based on instance features."),
            ("generate embeddings", "```bash\nrmpca graph-embed -i map.rmp -m fastrp -d 64 -o embeddings.json\n```\n\nMethods: `fastrp` (<1s), `spatial` (~2s), `line` (~3s), `node2vec` (~25s)."),
            ("full pipeline", "```bash\nrmpca pipeline --bbox \"-73.59,45.49,-73.55,45.52\" --output-dir ./output --depot 45.505,-73.565\n```\n\nRuns extract → clean → compile → optimize in sequence."),
        ]
        for query, answer in short_forms:
            self._add([
                make_system_message(SYSTEM_PROMPT),
                make_user_message(query),
                make_assistant_message(answer),
            ])

        # Concept questions
        concepts = [
            ("What is the .rmp format?",
             "The `.rmp` format is a compact binary representation of a road network:\n\n```\n[4 bytes]  Magic \"RMP1\"\n[4 bytes]  Node count (u32 LE)\n[4 bytes]  Edge count (u32 LE)\n[N * 16]   Nodes: lat(f64) + lon(f64)\n[E * 17]   Edges: from(u32) + to(u32) + weight_m(f64) + oneway(u8)\n[4 bytes]  CRC32 checksum\n```\n\nBenefits:\n- ~90% compression vs GeoJSON\n- CRC32 integrity checking\n- Node deduplication at 1-meter precision\n- Fast random access for route optimization"),
            ("What VRP solvers are available?",
             "Available VRP solvers:\n\n| Solver | Description | Complexity |\n|--------|-------------|------------|\n| `greedy` | Nearest neighbor insertion | O(n²) |\n| `clarke_wright` | Savings algorithm | O(n² log n) |\n| `sweep` | Angular clustering | O(n log n) |\n| `two_opt` | Local search improvement | O(n²) per iteration |\n| `or_opt` | Simulated annealing metaheuristic | Varies |\n| `neural_guided` | ML-guided heuristic | O(n²) |\n| `neural_onnx` | Direct neural inference | O(1) forward pass |\n\nUse `rmpca predict-solver` to get an ML recommendation for your instance."),
            ("What data sources does v2rmp support?",
             "v2rmp supports 4 data sources for road network extraction:\n\n1. **Overture Maps** — AWS S3 public bucket, Parquet with WKB geometry. No auth required.\n2. **OpenStreetMap** — Local `.osm.pbf` files or Overpass API fallback.\n3. **PMTiles** — Vector tile archives with layer filtering and zoom selection.\n4. **PostGIS** — Direct extraction from PostgreSQL/PostGIS `road_edges` tables.\n\nAll sources output GeoJSON that can be compiled to .rmp binary format."),
            ("How do turn penalties work?",
             "Turn penalties add extra cost for specific turn types during route optimization:\n\n- **Left turn** (±45-135° left bearing change): Default 1.0m\n- **Right turn** (±45-135° right bearing change): Default 0.0m\n- **U-turn** (>135° bearing change): Default 5.0m\n- **Straight** (±45° bearing change): No penalty\n\nHigher penalties discourage certain maneuvers. Useful for:\n- Urban safety (avoid left turns across traffic)\n- Large vehicle routing (avoid U-turns)\n- Time-aware routing (left turns take longer at intersections)\n\nSet via: `--left-penalty 1.0 --right-penalty 0.0 --uturn-penalty 5.0`"),
            ("What is the drone energy model?",
             "The drone VRP uses the Dorling et al. (2017) physics-based energy model:\n\n- **Power**: P = P_empty + (P_loaded - P_empty) × (payload / max_payload)\n- **Energy** (Wh): (Power × Distance / Speed) / 3600\n- **Wind impact**: ground_speed = max(cruise_speed - wind_speed, 1.0)\n\nSupported drone models: FlyCart30, Wing\n\nConfigure via TUI Drone VRP view or CLI:\n```bash\nrmpca-drone --model FlyCart30 --depot 45.505,-73.565 --customers '[[45.506,-73.566],[45.507,-73.567]]' --demands '[1.0,2.0]' --wind-speed 2.5\n```"),
        ]
        for question, answer in concepts:
            self._add([
                make_system_message(SYSTEM_PROMPT),
                make_user_message(question),
                make_assistant_message(answer),
            ])


def main():
    parser = argparse.ArgumentParser(description="Build v2rmp SFT training dataset")
    parser.add_argument("--output", "-o", default="dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    builder = V2RMPDatasetBuilder(seed=args.seed)
    examples = builder.build()

    # Write dataset
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_jsonl() + "\n")

    # Statistics
    if args.stats:
        total = len(examples)
        system_prompts = set()
        total_user_msgs = 0
        total_asst_msgs = 0
        total_chars = 0
        for ex in examples:
            for msg in ex.messages:
                if msg["role"] == "system":
                    system_prompts.add(msg["content"][:50])
                elif msg["role"] == "user":
                    total_user_msgs += 1
                elif msg["role"] == "assistant":
                    total_asst_msgs += 1
                    total_chars += len(msg["content"])

        print(f"Dataset Statistics:")
        print(f"  Total conversations: {total}")
        print(f"  Unique system prompts: {len(system_prompts)}")
        print(f"  User messages: {total_user_msgs}")
        print(f"  Assistant messages: {total_asst_msgs}")
        print(f"  Total chars in assistant responses: {total_chars:,}")
        print(f"  Avg chars per response: {total_chars // max(total_asst_msgs, 1):,}")
        print(f"\nOutput: {args.output}")

    print(f"✅ Generated {len(examples)} training examples → {args.output}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
v2rmp Agent SFT Training — Self-contained for HF Jobs
======================================================
Fine-tunes Qwen2.5-7B-Instruct on the v2rmp route optimization agent dataset
using QLoRA (4-bit + LoRA adapters).

The dataset is embedded directly in this script so it works in HF Jobs
which has no access to local files.
"""

import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
# V2RMPDatasetBuilder is defined above in this file

# ─── Build dataset ────────────────────────────────────────────────────────────

print("📦 Building v2rmp training dataset...")
builder = V2RMPDatasetBuilder(seed=42)
examples = builder.build()

# Convert to HF Dataset format
data = {"messages": [ex.messages for ex in examples]}
dataset = Dataset.from_dict(data)
print(f"   {len(dataset)} examples loaded")

# Train/validation split
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"   Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ─── LoRA config ──────────────────────────────────────────────────────────────

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
print(f"🔧 LoRA: r=16, alpha=32, targets={peft_config.target_modules}")

# ─── QLoRA 4-bit quantization ────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print("⚡ QLoRA: 4-bit NF4 with double quantization")

# ─── Training config ─────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./v2rmp-agent-sft"
HUB_MODEL_ID = "aerialblancaservices/v2rmp-agent-7b"

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Core hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,        # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Sequence handling
    max_length=4096,
    packing=True,

    # Precision & memory
    bf16=True,
    gradient_checkpointing=True,

    # Logging
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    disable_tqdm=True,

    # Saving
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,

    # Evaluation
    eval_strategy="steps",
    eval_steps=50,

    # Hub
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,

    # Seed
    seed=42,
)

# ─── Summary ──────────────────────────────────────────────────────────────────

effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
print(f"\n{'='*60}")
print(f"v2rmp Agent SFT Training")
print(f"{'='*60}")
print(f"  Model:              {MODEL_ID}")
print(f"  Method:             QLoRA (4-bit NF4 + LoRA r=16)")
print(f"  Dataset:            {len(train_dataset)} train / {len(eval_dataset)} eval")
print(f"  Learning rate:      {training_args.learning_rate:.1e}")
print(f"  Epochs:             {training_args.num_train_epochs}")
print(f"  Effective batch:    {effective_batch}")
print(f"  Max seq length:     {training_args.max_length}")
print(f"  Packing:            {training_args.packing}")
print(f"  Output:             {OUTPUT_DIR}")
print(f"  Hub:                {HUB_MODEL_ID}")
print(f"{'='*60}\n")

# ─── Train ────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=MODEL_ID,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    quantization_config=bnb_config,
)

print("🚀 Starting training...")
trainer.train()

# ── Save & push ───────────────────────────────────────────────────────────
print(f"\n💾 Saving model to {OUTPUT_DIR}...")
trainer.save_model()

print(f"📤 Pushing to Hub: {HUB_MODEL_ID}")
trainer.push_to_hub()

print(f"\n✅ Training complete!")
print(f"   Model: https://huggingface.co/{HUB_MODEL_ID}")
