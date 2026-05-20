# rmpca - Route Optimization Engine

[![Crates.io](https://img.shields.io/crates/v/v2rmp.svg)](https://crates.io/crates/v2rmp)
[![License](https://img.shields.io/crates/l/v2rmp.svg)](LICENSE)

High-performance route optimization for road networks. Solve Chinese Postman Problem (CPP) and Vehicle Routing Problem (VRP).

## Features

- **CPP Solvers**: Internal Rust solver + external `rust-optimizer` (5-8x faster, 20-60% better efficiency)
- **VRP Solvers**: Greedy, Clarke-Wright, Sweep, Two-Opt, OR-Tools, Neural
- **Data Sources**: Overture Maps, OSM PBF, PMTiles, PostGIS
- **Binary Format**: `.rmp` - compact binary road network storage (~90% compression)
- **MCP Server**: 25+ tools for AI agent integration
- **ML Features**: Solver selection, quality prediction, graph embeddings

## Installation

```bash
cargo install v2rmp
```

For better CPP results, install the external optimizer:
```bash
cargo install rust-optimizer
```

## Quick Start

```bash
# Extract road network
rmpca extract --source overture --bbox "-73.6,45.5,-73.5,45.6" --output roads.geojson

# Compile to binary
rmpca compile --input roads.geojson --output map.rmp

# Optimize route (use external engine for best results)
rmpca optimize --input map.rmp --output route.gpx --cpp-engine external-rust-optimizer

# Or use the full pipeline
rmpca pipeline --bbox "..." --output-prefix my_route
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `extract` | Download road data from Overture/OSM/PMTiles |
| `compile` | Convert GeoJSON to .rmp binary |
| `optimize` | Solve CPP on .rmp map |
| `vrp` | Solve Vehicle Routing Problem |
| `serve` | Start MCP JSON-RPC server |
| `clean` | Repair GeoJSON road networks |
| `partition` | Split graph into zones |
| `elevation` | Query DEM elevation data |

## MCP Server

Connect AI agents to v2rmp with 25+ tools:

```json
{
  "mcpServers": {
    "rmpca": {
      "command": "rmpca",
      "args": ["serve"]
    }
  }
}
```

Tools: `extract_overture`, `compile`, `optimize`, `v2rmp_rust_optimizer`, `vrp_solve`, `partition`, `list_r2_bucket`, etc.

## Performance

| Metric | Internal Engine | External rust-optimizer |
|--------|-----------------|------------------------|
| Efficiency | ~23-53% | ~72-87% |
| Speed | ~900-1400ms | ~160-180ms |
| Route Length | Longer | 20-60% shorter |

## License

MIT OR Apache-2.0
