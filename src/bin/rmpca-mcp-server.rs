//! rmpca-mcp-server — MCP server exposing the route optimization pipeline.
//!
//! Tools: extract_overture, extract_osm, compile, optimize,
//!        clean, vrp_solve, elevation_query, elevation_profile,
//!        predict_solver, score_route, route_embedding, pipeline,
//!        haversine_distance, get_valhalla_matrix, inspect_rmp,
//!        list_solvers, elevation_stats, dem_info, fuel_estimate
//!
//! Runs over stdio with JSON-RPC 2.0 framing (one line per message).
//!
//! Connect from any MCP client by adding to the client config:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "rmpca": {
//!       "command": "cargo",
//!       "args": ["run", "--bin", "rmpca-mcp-server", "--release"]
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::Path;
use v2rmp::core::clean::{clean_geojson, CleanOptions};
use v2rmp::core::compile::{CompileRequest, CompileResult};
use v2rmp::core::elevation::local::LocalDem;
use v2rmp::core::extract::{BBoxRequest, ExtractRequest, ExtractResult, ExtractSource, RoadClass};
use v2rmp::core::optimize::{
    OnewayMode, OptimizeRequest, OptimizeResult, SolverMode, TurnPenalties,
};
use v2rmp::core::vrp::registry::solve_with;
use v2rmp::core::vrp::types::{
    VRPSolverInput, VRPSolverOutput, VRPSolverStop, VrpObjective,
};
use v2rmp::core::vrp::utils::{build_haversine_matrix, get_valhalla_matrix};
use v2rmp::core::elevation::{FuelCalculator};
#[cfg(feature = "ml")]
use v2rmp::core::ml::features::InstanceFeatures;
#[cfg(feature = "ml")]
use v2rmp::core::ml::selector::{predict_solver, default_model_path};
#[cfg(feature = "ml")]
use v2rmp::core::ml::quality_predictor::{predict_quality};
#[cfg(feature = "ml")]
use v2rmp::core::ml::automl::{predict_hyperparams};
use v2rmp::core::ml_legacy::{RouteFeatures, score_route, route_feature_vector};
#[cfg(feature = "ml")]
use v2rmp::core::nlp::{parse_query, to_vrp_json};
#[cfg(feature = "ml")]
use v2rmp::core::nlp::QwenNLParser;

// ── JSON-RPC / MCP types ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Request {
    jsonrpc: String,
    method: String,
    #[serde(default)]
    params: Value,
    #[serde(default)]
    id: Value,
}

#[derive(Debug)]
struct ToolDef {
    name: &'static str,
    description: &'static str,
    input_schema: Value,
}

// ── Protocol helpers ────────────────────────────────────────────────────────

fn send(id: &Value, result: Value) {
    let out = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    });
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{out}");
    let _ = stdout.flush();
}

fn send_err(id: &Value, code: i64, msg: &str) {
    let out = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": msg }
    });
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{out}");
    let _ = stdout.flush();
}

fn tool_success(id: &Value, result: &Result<Value>) {
    match result {
        Ok(val) => send(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(val).unwrap_or_default()
                }]
            }),
        ),
        Err(e) => send(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error: {e:#}")
                }],
                "isError": true
            }),
        ),
    }
}

// ── Tool definitions ────────────────────────────────────────────────────────

fn tool_definitions() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "extract_overture",
            description: "Extract road network data from Overture Maps S3 Parquet files. \
                Downloads road segments within a bounding box and writes a GeoJSON file. \
                Can take significant time for large bounding boxes.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "description": "Bounding box: {min_lon, min_lat, max_lon, max_lat}",
                        "properties": {
                            "min_lon": { "type": "number" },
                            "min_lat": { "type": "number" },
                            "max_lon": { "type": "number" },
                            "max_lat": { "type": "number" }
                        },
                        "required": ["min_lon", "min_lat", "max_lon", "max_lat"]
                    },
                    "road_classes": {
                        "type": "array",
                        "description": "Road classes to include (e.g. ['residential','tertiary','secondary']). Default: all vehicle-accessible roads.",
                        "items": { "type": "string" }
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output GeoJSON file path (default: extract-output.geojson)",
                        "default": "extract-output.geojson"
                    }
                },
                "required": ["bbox"]
            }),
        },
        ToolDef {
            name: "extract_osm",
            description: "Extract road network data from OpenStreetMap. Uses a local PBF file \
                if available, otherwise falls back to the Overpass API. Writes a GeoJSON file.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "description": "Bounding box: {min_lon, min_lat, max_lon, max_lat}",
                        "properties": {
                            "min_lon": { "type": "number" },
                            "min_lat": { "type": "number" },
                            "max_lon": { "type": "number" },
                            "max_lat": { "type": "number" }
                        },
                        "required": ["min_lon", "min_lat", "max_lon", "max_lat"]
                    },
                    "road_classes": {
                        "type": "array",
                        "description": "Road classes to include. Default: all vehicle-accessible roads.",
                        "items": { "type": "string" }
                    },
                    "pbf_path": {
                        "type": "string",
                        "description": "Path to local OSM PBF file. If omitted, falls back to Overpass API."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output GeoJSON file path (default: extract-output.geojson)",
                        "default": "extract-output.geojson"
                    }
                },
                "required": ["bbox"]
            }),
        },
        ToolDef {
            name: "compile",
            description: "Compile a GeoJSON road network file into the binary .rmp format. \
                Optionally runs a cleaning pipeline and prunes disconnected subgraphs.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Path to input GeoJSON file"
                    },
                    "output": {
                        "type": "string",
                        "description": "Path to output .rmp binary file"
                    },
                    "clean": {
                        "type": "boolean",
                        "description": "Run cleaning pipeline before compilation (default: false)",
                        "default": false
                    },
                    "prune_disconnected": {
                        "type": "boolean",
                        "description": "Prune disconnected subgraphs, keeping only the largest (default: false)",
                        "default": false
                    }
                },
                "required": ["input", "output"]
            }),
        },
        ToolDef {
            name: "optimize",
            description: "Run route optimization on a .rmp binary network file. \
                Supports CPP (Chinese Postman — edge coverage) and VRP (Vehicle Routing Problem) modes. \
                Outputs route statistics and optionally writes a GPX file.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Path to input .rmp binary network file"
                    },
                    "output": {
                        "type": "string",
                        "description": "Optional output GPX route file path"
                    },
                    "depot": {
                        "type": "object",
                        "description": "Depot coordinates {lat, lon}. Solver snaps to nearest node.",
                        "properties": {
                            "lat": { "type": "number" },
                            "lon": { "type": "number" }
                        }
                    },
                    "oneway_mode": {
                        "type": "string",
                        "enum": ["respect", "ignore", "reverse"],
                        "description": "How to handle one-way streets (default: respect)",
                        "default": "respect"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["cpp", "vrp"],
                        "description": "Solver: cpp for edge coverage, vrp for stop visits (default: cpp)",
                        "default": "cpp"
                    },
                    "left_penalty": {
                        "type": "number",
                        "description": "Left turn penalty (default: 1.0)",
                        "default": 1.0
                    },
                    "right_penalty": {
                        "type": "number",
                        "description": "Right turn penalty (default: 0.0)",
                        "default": 0.0
                    },
                    "uturn_penalty": {
                        "type": "number",
                        "description": "U-turn penalty (default: 5.0)",
                        "default": 5.0
                    },
                    "num_vehicles": {
                        "type": "integer",
                        "description": "Number of vehicles — VRP mode only (default: 1)",
                        "default": 1
                    },
                    "solver_id": {
                        "type": "string",
                        "description": "VRP solver algorithm: default, clarke_wright, sweep, two_opt, or_opt",
                        "default": "default"
                    }
                },
                "required": ["input"]
            }),
        },
        // ── New tools ──────────────────────────────────────────────────────────
        ToolDef {
            name: "clean",
            description: "Clean a GeoJSON road network with full control over all cleaning parameters. \
                Runs the 11-stage cleaning pipeline: repair → build graph → remove self-loops → \
                remove short edges → merge nearby nodes → deduplicate edges → remove edges missing attrs → \
                merge parallel edges → remove isolates → keep largest components → export. \
                Returns cleaned GeoJSON plus detailed statistics.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Path to input GeoJSON file"
                    },
                    "output": {
                        "type": "string",
                        "description": "Path to output cleaned GeoJSON file"
                    },
                    "make_valid": {
                        "type": "boolean",
                        "description": "Repair invalid geometries (default: true)"
                    },
                    "drop_invalid": {
                        "type": "boolean",
                        "description": "Drop features that cannot be repaired (default: true)"
                    },
                    "remove_selfloops": {
                        "type": "boolean",
                        "description": "Remove self-loop edges (default: true)"
                    },
                    "min_length_m": {
                        "type": "number",
                        "description": "Remove edges shorter than this in metres (default: 0.1)"
                    },
                    "node_snap_m": {
                        "type": "number",
                        "description": "Merge nodes within this distance in metres (default: 1.0)"
                    },
                    "node_precision_decimals": {
                        "type": "integer",
                        "description": "Decimal places for node ID generation (default: 6)"
                    },
                    "merge_node_positions": {
                        "type": "boolean",
                        "description": "Average coordinates when merging nodes (default: true)"
                    },
                    "dedupe_edges": {
                        "type": "boolean",
                        "description": "Remove duplicate edges (default: true)"
                    },
                    "remove_isolates": {
                        "type": "boolean",
                        "description": "Remove isolated nodes (default: true)"
                    },
                    "max_components": {
                        "type": "integer",
                        "description": "Keep only the N largest connected components (0 = keep all, default: 1)"
                    },
                    "required_attrs": {
                        "type": "array",
                        "description": "Remove edges missing any of these property names",
                        "items": { "type": "string" }
                    },
                    "merge_parallel_edges": {
                        "type": "boolean",
                        "description": "Merge parallel edges between same nodes (default: false)"
                    },
                    "merge_parallel_edge_properties": {
                        "type": "boolean",
                        "description": "When merging parallel edges, also merge their properties (default: false)"
                    },
                    "property_merge_strategy": {
                        "type": "string",
                        "enum": ["first", "merge"],
                        "description": "How to merge properties on parallel edges: 'first' keeps first edge's props, 'merge' combines them (default: first)"
                    },
                    "simplify_tolerance_m": {
                        "type": "number",
                        "description": "Douglas-Peucker simplification tolerance in metres (0 = no simplification, default: 0)"
                    },
                    "include_polygons": {
                        "type": "boolean",
                        "description": "Include polygon features in output (default: false)"
                    },
                    "include_points": {
                        "type": "boolean",
                        "description": "Include point features in output (default: false)"
                    }
                },
                "required": ["input", "output"]
            }),
        },
        ToolDef {
            name: "vrp_solve",
            description: "Solve a Vehicle Routing Problem (VRP) with explicit stop coordinates. \
                Unlike the 'optimize' tool (which operates on .rmp files), this tool takes an array \
                of stop coordinates, builds a haversine distance matrix, and dispatches to the chosen \
                solver algorithm. Returns per-vehicle routes with stop assignments and statistics.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "description": "Array of stops. The first stop is the depot.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": { "type": "number", "description": "Latitude" },
                                "lon": { "type": "number", "description": "Longitude" },
                                "label": { "type": "string", "description": "Label for this stop (optional)" },
                                "demand": { "type": "number", "description": "Demand at this stop (default: 1.0)" }
                            },
                            "required": ["lat", "lon"]
                        }
                    },
                    "num_vehicles": {
                        "type": "integer",
                        "description": "Number of vehicles (default: 1)"
                    },
                    "vehicle_capacity": {
                        "type": "number",
                        "description": "Vehicle capacity (default: 100.0)"
                    },
                    "solver_id": {
                        "type": "string",
                        "description": "VRP solver algorithm: default, clarke_wright, sweep, two_opt, or_opt",
                        "default": "default"
                    },
                    "avg_speed_kmh": {
                        "type": "number",
                        "description": "Average speed in km/h for time estimation (default: 40.0)"
                    },
                    "objective": {
                        "type": "string",
                        "enum": ["min_distance", "min_time", "balance_load", "min_vehicles"],
                        "description": "Optimization objective (default: min_distance)"
                    }
                },
                "required": ["stops"]
            }),
        },
        ToolDef {
            name: "elevation_query",
            description: "Query elevation at one or more lat/lon points from a local DEM GeoTIFF file. \
                Uses bilinear interpolation with nearest-neighbor fallback for nodata pixels. \
                Returns an array of elevations (or null for points outside coverage).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "dem_path": {
                        "type": "string",
                        "description": "Path to the DEM GeoTIFF file"
                    },
                    "points": {
                        "type": "array",
                        "description": "Array of {lon, lat} points to query",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lon": { "type": "number" },
                                "lat": { "type": "number" }
                            },
                            "required": ["lon", "lat"]
                        }
                    }
                },
                "required": ["dem_path", "points"]
            }),
        },
        ToolDef {
            name: "elevation_profile",
            description: "Sample elevation along a route at fixed intervals from a local DEM GeoTIFF file. \
                Returns per-sample points with distance and elevation, plus total ascent, descent, \
                min/max/avg elevation, and total distance.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "dem_path": {
                        "type": "string",
                        "description": "Path to the DEM GeoTIFF file"
                    },
                    "route": {
                        "type": "array",
                        "description": "Ordered array of {lon, lat} waypoints forming the route",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lon": { "type": "number" },
                                "lat": { "type": "number" }
                            },
                            "required": ["lon", "lat"]
                        }
                    },
                    "sample_interval_m": {
                        "type": "number",
                        "description": "Distance between elevation samples in metres (default: 100.0)"
                    }
                },
                "required": ["dem_path", "route"]
            }),
        },
        ToolDef {
            name: "list_solvers",
            description: "List available VRP solvers and their labels.",
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
        // ── Medium-priority tools ──────────────────────────────────────────
        ToolDef {
            name: "elevation_stats",
            description: "Compute elevation statistics (min, max, avg, coverage %) within a bounding box \
                from a DEM GeoTIFF file. Samples on a grid with configurable step size.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "dem_path": {
                        "type": "string",
                        "description": "Path to the DEM GeoTIFF file"
                    },
                    "bbox": {
                        "type": "object",
                        "description": "Bounding box: {min_lon, min_lat, max_lon, max_lat}",
                        "properties": {
                            "min_lon": { "type": "number" },
                            "min_lat": { "type": "number" },
                            "max_lon": { "type": "number" },
                            "max_lat": { "type": "number" }
                        },
                        "required": ["min_lon", "min_lat", "max_lon", "max_lat"]
                    },
                    "grid_step": {
                        "type": "integer",
                        "description": "Pixel skip for sampling (1 = every pixel, 10 = every 10th). Default: 1",
                        "default": 1
                    }
                },
                "required": ["dem_path", "bbox"]
            }),
        },
        ToolDef {
            name: "dem_info",
            description: "Return metadata about a DEM GeoTIFF file: width, height, bounding box, \
                nodata value, and pixel size in degrees.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "dem_path": {
                        "type": "string",
                        "description": "Path to the DEM GeoTIFF file"
                    }
                },
                "required": ["dem_path"]
            }),
        },
        ToolDef {
            name: "fuel_estimate",
            description: "Calculate fuel consumption from an elevation profile. Takes an array of \
                {distance_m, elevation_m} samples and a base consumption rate (L/km). Applies grade-based \
                adjustments: +15% per 1% uphill grade, -5% per 1% downhill grade (capped at -20%).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "samples": {
                        "type": "array",
                        "description": "Array of {distance_m, elevation_m} points along the route",
                        "items": {
                            "type": "object",
                            "properties": {
                                "distance_m": { "type": "number", "description": "Cumulative distance in metres" },
                                "elevation_m": { "type": "number", "description": "Elevation in metres" }
                            },
                            "required": ["distance_m", "elevation_m"]
                        }
                    },
                    "base_consumption": {
                        "type": "number",
                        "description": "Base fuel consumption in L/km on flat terrain (default: 0.08, ~8L/100km)",
                        "default": 0.08
                    }
                },
                "required": ["samples"]
            }),
        },
        ToolDef {
            name: "inspect_rmp",
            description: "Parse a .rmp binary network file and return node count, edge count, and \
                bounding box without running optimization. Useful for validating files and inspecting \
                network geometry.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Path to the .rmp binary file"
                    }
                },
                "required": ["input"]
            }),
        },
        ToolDef {
            name: "predict_solver",
            description: "Recommend the best VRP solver algorithm for a given instance based on \
                geometric and capacity features. Returns the recommended solver id, confidence \
                score, runner-up, and per-solver fit scores.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "description": "Array of stops. The first stop is the depot.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": { "type": "number", "description": "Latitude" },
                                "lon": { "type": "number", "description": "Longitude" },
                                "label": { "type": "string", "description": "Label for this stop (optional)" },
                                "demand": { "type": "number", "description": "Demand at this stop (default: 1.0)" }
                            },
                            "required": ["lat", "lon"]
                        }
                    },
                    "num_vehicles": {
                        "type": "integer",
                        "description": "Number of vehicles (default: 1)"
                    },
                    "vehicle_capacity": {
                        "type": "number",
                        "description": "Vehicle capacity (default: 100.0)"
                    },
                    "objective": {
                        "type": "string",
                        "enum": ["min_distance", "min_time", "balance_load", "min_vehicles"],
                        "description": "Optimization objective (default: min_distance)"
                    }
                },
                "required": ["stops"]
            }),
        },
        ToolDef {
            name: "score_route",
            description: "Score a solved VRP route on multiple quality dimensions: \
                distance efficiency, load balance, turn quality, and coverage. \
                Returns an overall composite score (0-100) plus per-dimension breakdown.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "description": "Array of stops. The first stop is the depot.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": { "type": "number" },
                                "lon": { "type": "number" },
                                "label": { "type": "string" },
                                "demand": { "type": "number" }
                            },
                            "required": ["lat", "lon"]
                        }
                    },
                    "routes": {
                        "type": "array",
                        "description": "Per-vehicle routes as arrays of stop indices",
                        "items": {
                            "type": "array",
                            "items": { "type": "integer" }
                        }
                    },
                    "total_distance_km": {
                        "type": "string",
                        "description": "Total distance string, e.g. '42.50'"
                    },
                    "num_vehicles": {
                        "type": "integer",
                        "description": "Number of vehicles used in the input"
                    },
                    "vehicle_capacity": {
                        "type": "number",
                        "description": "Vehicle capacity"
                    },
                    "objective": {
                        "type": "string",
                        "enum": ["min_distance", "min_time", "balance_load", "min_vehicles"]
                    }
                },
                "required": ["stops", "routes", "total_distance_km"]
            }),
        },
        ToolDef {
            name: "route_embedding",
            description: "Generate a 12-dimensional feature vector for a VRP instance \
                suitable for similarity search, clustering, or learned-model input. \
                Values are normalized to [0,1].",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "description": "Array of stops. The first stop is the depot.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": { "type": "number" },
                                "lon": { "type": "number" },
                                "label": { "type": "string" },
                                "demand": { "type": "number" }
                            },
                            "required": ["lat", "lon"]
                        }
                    },
                    "num_vehicles": { "type": "integer" },
                    "vehicle_capacity": { "type": "number" },
                    "objective": {
                        "type": "string",
                        "enum": ["min_distance", "min_time", "balance_load", "min_vehicles"]
                    }
                },
                "required": ["stops"]
            }),
        },
        ToolDef {
            name: "pipeline",
            description: "End-to-end route optimization pipeline: extract road network → clean → \
                compile to .rmp binary → optimize. Runs all four stages in sequence and returns \
                combined results. Can take significant time for large bounding boxes.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "description": "Bounding box: {min_lon, min_lat, max_lon, max_lat}",
                        "properties": {
                            "min_lon": { "type": "number" },
                            "min_lat": { "type": "number" },
                            "max_lon": { "type": "number" },
                            "max_lat": { "type": "number" }
                        },
                        "required": ["min_lon", "min_lat", "max_lon", "max_lat"]
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory for intermediate and output files (default: ./pipeline-output)",
                        "default": "./pipeline-output"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["overture", "osm"],
                        "description": "Data source for extraction (default: overture)"
                    },
                    "pbf_path": {
                        "type": "string",
                        "description": "Path to local OSM PBF file (OSM source only)"
                    },
                    "depot": {
                        "type": "object",
                        "description": "Depot coordinates {lat, lon}",
                        "properties": {
                            "lat": { "type": "number" },
                            "lon": { "type": "number" }
                        }
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["cpp", "vrp"],
                        "description": "Solver mode: cpp for edge coverage, vrp for stop visits (default: cpp)"
                    },
                    "prune_disconnected": {
                        "type": "boolean",
                        "description": "Prune disconnected subgraphs during compilation (default: false)"
                    }
                },
                "required": ["bbox"]
            }),
        },
        ToolDef {
            name: "haversine_distance",
            description: "Calculate the great-circle distance between two WGS-84 lat/lon points \
                using the haversine formula. Returns distance in metres and kilometres.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "from": {
                        "type": "object",
                        "description": "Origin point",
                        "properties": {
                            "lat": { "type": "number", "description": "Latitude in degrees" },
                            "lon": { "type": "number", "description": "Longitude in degrees" }
                        },
                        "required": ["lat", "lon"]
                    },
                    "to": {
                        "type": "object",
                        "description": "Destination point",
                        "properties": {
                            "lat": { "type": "number", "description": "Latitude in degrees" },
                            "lon": { "type": "number", "description": "Longitude in degrees" }
                        },
                        "required": ["lat", "lon"]
                    }
                },
                "required": ["from", "to"]
            }),
        },
        ToolDef {
            name: "get_valhalla_matrix",
            description: "Fetch a real-road distance/time matrix from the public Valhalla/OSRM API. \
                Takes an array of stop coordinates and returns an NxN matrix with distance (km) and \
                time (seconds) for each pair. Requires internet connectivity.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "description": "Array of {lat, lon} coordinates",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": { "type": "number" },
                                "lon": { "type": "number" }
                            },
                            "required": ["lat", "lon"]
                        }
                    }
                },
                "required": ["locations"]
            }),
        },
        ToolDef {
            name: "predict_quality",
            description: "Predict the expected route quality (gap to optimal and estimated tour length) \
                before actually solving the VRP instance. Uses a learned model (or heuristic fallback) \
                on 28-dimensional instance features.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "description": "Array of stops. The first stop is the depot.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": { "type": "number" },
                                "lon": { "type": "number" },
                                "label": { "type": "string" },
                                "demand": { "type": "number" }
                            },
                            "required": ["lat", "lon"]
                        }
                    },
                    "num_vehicles": { "type": "integer", "default": 1 },
                    "vehicle_capacity": { "type": "number", "default": 100.0 },
                    "objective": {
                        "type": "string",
                        "enum": ["min_distance", "min_time", "balance_load", "min_vehicles"],
                        "default": "min_distance"
                    }
                },
                "required": ["stops"]
            }),
        },
        ToolDef {
            name: "tune_hyperparams",
            description: "Predict instance-aware solver hyperparameters (max iterations, temperature, \
                cooling rate, tabu tenure, neighbourhood radius) from geometric and graph features. \
                Falls back to sensible defaults if no learned model is available.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "stops": {
                        "type": "array",
                        "description": "Array of stops. The first stop is the depot.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": { "type": "number" },
                                "lon": { "type": "number" },
                                "label": { "type": "string" },
                                "demand": { "type": "number" }
                            },
                            "required": ["lat", "lon"]
                        }
                    },
                    "num_vehicles": { "type": "integer", "default": 1 },
                    "vehicle_capacity": { "type": "number", "default": 100.0 },
                    "objective": {
                        "type": "string",
                        "enum": ["min_distance", "min_time", "balance_load", "min_vehicles"],
                        "default": "min_distance"
                    }
                },
                "required": ["stops"]
            }),
        },
        ToolDef {
            name: "parse_routing_query",
            description: "Convert a natural-language routing request into a structured VRP JSON config. \
                Extracts entities such as number of packages, vehicles, depot coordinates, deadlines, \
                capacity, speed, and optimization objective. Returns a JSON object ready for the vrp_solve tool.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language routing request, e.g. 'Route 50 packages with 5 vans starting at 45.5,-73.6 by 5pm'"
                    },
                    "use_llm": {
                        "type": "boolean",
                        "description": "Use a local LLM (Qwen2.5-0.5B) for complex/ambiguous queries. Requires the 'ml' feature and ~1GB RAM. (default: false)",
                        "default": false
                    }
                },
                "required": ["query"]
            }),
        },
    ]
}

// ── Argument parsing helpers ────────────────────────────────────────────────

fn parse_bbox(args: &Value) -> anyhow::Result<BBoxRequest> {
    let bbox = args
        .get("bbox")
        .ok_or_else(|| anyhow::anyhow!("Missing 'bbox' parameter"))?;
    Ok(BBoxRequest {
        min_lon: bbox
            .get("min_lon")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.min_lon required"))?,
        min_lat: bbox
            .get("min_lat")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.min_lat required"))?,
        max_lon: bbox
            .get("max_lon")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.max_lon required"))?,
        max_lat: bbox
            .get("max_lat")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.max_lat required"))?,
    })
}

fn parse_road_classes(args: &Value) -> Vec<RoadClass> {
    let Some(arr) = args.get("road_classes").and_then(|v| v.as_array()) else {
        return RoadClass::all_vehicle();
    };
    if arr.is_empty() {
        return RoadClass::all_vehicle();
    }
    arr.iter()
        .filter_map(|v| v.as_str())
        .filter_map(|s| match s {
            "residential" => Some(RoadClass::Residential),
            "tertiary" => Some(RoadClass::Tertiary),
            "secondary" => Some(RoadClass::Secondary),
            "primary" => Some(RoadClass::Primary),
            "trunk" => Some(RoadClass::Trunk),
            "motorway" => Some(RoadClass::Motorway),
            "unclassified" => Some(RoadClass::Unclassified),
            "living_street" => Some(RoadClass::LivingStreet),
            "service" => Some(RoadClass::Service),
            "secondary_link" => Some(RoadClass::SecondaryLink),
            "primary_link" => Some(RoadClass::PrimaryLink),
            "trunk_link" => Some(RoadClass::TrunkLink),
            "motorway_link" => Some(RoadClass::MotorwayLink),
            _ => None,
        })
        .collect()
}

fn parse_oneway_mode(args: &Value) -> OnewayMode {
    match args
        .get("oneway_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("respect")
    {
        "ignore" => OnewayMode::Ignore,
        "reverse" => OnewayMode::Reverse,
        _ => OnewayMode::Respect,
    }
}

fn parse_solver_mode(args: &Value) -> SolverMode {
    match args
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("cpp")
    {
        "vrp" => SolverMode::Vrp,
        _ => SolverMode::Cpp,
    }
}

// ── Tool handlers ───────────────────────────────────────────────────────────

async fn handle_extract_overture(args: &Value) -> Result<Value> {
    let bbox = parse_bbox(args)?;
    let road_classes = parse_road_classes(args);
    let output_path = args
        .get("output_path")
        .and_then(|v| v.as_str())
        .unwrap_or("extract-output.geojson")
        .to_string();

    tracing::info!("extract_overture: bbox={:.4},{:.4},{:.4},{:.4}",
        bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat);

    let req = ExtractRequest {
        source: ExtractSource::Overture,
        bbox,
        road_classes,
        output_path,
        pbf_path: None,
    };

    let result: ExtractResult = v2rmp::core::extract::run_extract(&req).await?;
    Ok(serde_json::to_value(result)?)
}

async fn handle_extract_osm(args: &Value) -> Result<Value> {
    let bbox = parse_bbox(args)?;
    let road_classes = parse_road_classes(args);
    let pbf_path = args.get("pbf_path").and_then(|v| v.as_str()).map(String::from);
    let output_path = args
        .get("output_path")
        .and_then(|v| v.as_str())
        .unwrap_or("extract-output.geojson")
        .to_string();

    tracing::info!("extract_osm: bbox={:.4},{:.4},{:.4},{:.4}",
        bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat);

    let req = ExtractRequest {
        source: ExtractSource::Osm,
        bbox,
        road_classes,
        output_path,
        pbf_path,
    };

    let result: ExtractResult = v2rmp::core::extract::run_extract(&req).await?;
    Ok(serde_json::to_value(result)?)
}

fn handle_compile(args: &Value) -> Result<Value> {
    let input = args
        .get("input")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'input' parameter"))?
        .to_string();
    let output = args
        .get("output")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'output' parameter"))?
        .to_string();
    let clean = args
        .get("clean")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let prune = args
        .get("prune_disconnected")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let clean_options = if clean {
        Some(v2rmp::core::clean::CleanOptions::default())
    } else {
        None
    };

    tracing::info!("compile: {} -> {}", input, output);

    let req = CompileRequest {
        input_geojson: input,
        output_rmp: output,
        compress: false,
        road_classes: vec![],
        clean_options,
        prune_disconnected: prune,
    };

    let result: CompileResult = v2rmp::core::compile::run_compile(&req)?;
    Ok(serde_json::to_value(result)?)
}

async fn handle_optimize(args: &Value) -> Result<Value> {
    let input = args
        .get("input")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'input' parameter"))?
        .to_string();
    let route_file = args.get("output").and_then(|v| v.as_str()).map(String::from);
    let depot = args
        .get("depot")
        .and_then(|d| {
            let lat = d.get("lat")?.as_f64()?;
            let lon = d.get("lon")?.as_f64()?;
            Some((lat, lon))
        });
    let oneway_mode = parse_oneway_mode(args);
    let mode = parse_solver_mode(args);
    let left = args
        .get("left_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let right = args
        .get("right_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let u_turn = args
        .get("uturn_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(5.0);
    let num_vehicles = args
        .get("num_vehicles")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let solver_id = args
        .get("solver_id")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    tracing::info!("optimize: input={}", input);
let req = OptimizeRequest {
    cache_file: input,
    route_file,
    turn_penalties: TurnPenalties {
        left,
        right,
        u_turn,
    },
    depot,
    oneway_mode,
    mode,
    num_vehicles,
    solver_id,
    coordinates: None,
    };

    let result: OptimizeResult = v2rmp::core::optimize::run_optimize(&req).await?;
    Ok(serde_json::to_value(result)?)
}

// ── Clean handler ─────────────────────────────────────────────────────────

fn handle_clean(args: &Value) -> Result<Value> {
    let input = args
        .get("input")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'input' parameter"))?
        .to_string();
    let output = args
        .get("output")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'output' parameter"))?
        .to_string();

    // Build CleanOptions from args, using defaults where not specified
    let defaults = CleanOptions::default();
    let options = CleanOptions {
        make_valid: args.get("make_valid").and_then(|v| v.as_bool()).unwrap_or(defaults.make_valid),
        drop_invalid: args.get("drop_invalid").and_then(|v| v.as_bool()).unwrap_or(defaults.drop_invalid),
        remove_selfloops: args.get("remove_selfloops").and_then(|v| v.as_bool()).unwrap_or(defaults.remove_selfloops),
        min_length_m: args.get("min_length_m").and_then(|v| v.as_f64()).unwrap_or(defaults.min_length_m),
        node_snap_m: args.get("node_snap_m").and_then(|v| v.as_f64()).unwrap_or(defaults.node_snap_m),
        node_precision_decimals: args.get("node_precision_decimals").and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(defaults.node_precision_decimals),
        merge_node_positions: args.get("merge_node_positions").and_then(|v| v.as_bool()).unwrap_or(defaults.merge_node_positions),
        dedupe_edges: args.get("dedupe_edges").and_then(|v| v.as_bool()).unwrap_or(defaults.dedupe_edges),
        remove_isolates: args.get("remove_isolates").and_then(|v| v.as_bool()).unwrap_or(defaults.remove_isolates),
        max_components: args.get("max_components").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(defaults.max_components),
        required_attrs: args.get("required_attrs").and_then(|v| v.as_array()).map(|arr| {
            arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()
        }),
        merge_parallel_edges: args.get("merge_parallel_edges").and_then(|v| v.as_bool()).unwrap_or(defaults.merge_parallel_edges),
        merge_parallel_edge_properties: args.get("merge_parallel_edge_properties").and_then(|v| v.as_bool()).unwrap_or(defaults.merge_parallel_edge_properties),
        property_merge_strategy: args.get("property_merge_strategy").and_then(|v| v.as_str()).map(String::from).unwrap_or(defaults.property_merge_strategy),
        simplify_tolerance_m: args.get("simplify_tolerance_m").and_then(|v| v.as_f64()).unwrap_or(defaults.simplify_tolerance_m),
        include_polygons: args.get("include_polygons").and_then(|v| v.as_bool()).unwrap_or(defaults.include_polygons),
        include_points: args.get("include_points").and_then(|v| v.as_bool()).unwrap_or(defaults.include_points),
    };

    tracing::info!("clean: {} -> {} (min_length={}, node_snap={}, max_components={})",
        input, output, options.min_length_m, options.node_snap_m, options.max_components);

    // Read input GeoJSON
    let input_str = std::fs::read_to_string(&input)
        .map_err(|e| anyhow::anyhow!("Failed to read input file '{}': {}", input, e))?;
    let geojson: geojson::FeatureCollection = serde_json::from_str(&input_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse GeoJSON: {}", e))?;

    // Run clean pipeline
    let (cleaned, stats, warnings) = clean_geojson(&geojson, &options)
        .map_err(|e| anyhow::anyhow!("Clean pipeline failed: {}", e))?;

    // Write output
    let output_str = serde_json::to_string_pretty(&cleaned)
        .map_err(|e| anyhow::anyhow!("Failed to serialize output: {}", e))?;
    std::fs::write(&output, &output_str)
        .map_err(|e| anyhow::anyhow!("Failed to write output file '{}': {}", output, e))?;

    Ok(json!({
        "output_file": output,
        "input_features": stats.input_features,
        "output_features": stats.output_features,
        "invalid_dropped": stats.invalid_dropped,
        "selfloops_removed": stats.selfloops_removed,
        "short_edges_removed": stats.short_edges_removed,
        "nodes_merged": stats.nodes_merged,
        "duplicate_edges_removed": stats.duplicate_edges_removed,
        "incomplete_edges_removed": stats.incomplete_edges_removed,
        "parallel_edges_merged": stats.parallel_edges_merged,
        "isolates_removed": stats.isolates_removed,
        "components_removed": stats.components_removed,
        "total_removed": stats.total_removed(),
        "warnings": warnings,
    }))
}

// ── VRP Solve handler ────────────────────────────────────────────────────

async fn handle_vrp_solve(args: &Value) -> Result<Value> {
    let stops_val = args
        .get("stops")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'stops' parameter"))?;

    if stops_val.is_empty() {
        anyhow::bail!("At least one stop (the depot) is required");
    }

    let stops: Vec<VRPSolverStop> = stops_val
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let lat = s.get("lat").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lat'", i))?;
            let lon = s.get("lon").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lon'", i))?;
            let label = s.get("label").and_then(|v| v.as_str())
                .unwrap_or_else(|| "")
                .to_string();
            let demand = s.get("demand").and_then(|v| v.as_f64());
            Ok(VRPSolverStop {
                lat,
                lon,
                label: if label.is_empty() { format!("Stop {}", i) } else { label },
                demand,
                arrival_time: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_vehicles = args.get("num_vehicles").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let vehicle_capacity = args.get("vehicle_capacity").and_then(|v| v.as_f64()).unwrap_or(100.0);
    let solver_id = args.get("solver_id").and_then(|v| v.as_str()).unwrap_or("default").to_string();
    let avg_speed_kmh = args.get("avg_speed_kmh").and_then(|v| v.as_f64()).unwrap_or(40.0);
    let objective = match args.get("objective").and_then(|v| v.as_str()).unwrap_or("min_distance") {
        "min_time" => VrpObjective::MinTime,
        "balance_load" => VrpObjective::BalanceLoad,
        "min_vehicles" => VrpObjective::MinVehicles,
        _ => VrpObjective::MinDistance,
    };

    // Build haversine distance matrix
    let matrix = build_haversine_matrix(&stops, avg_speed_kmh);

    #[cfg(feature = "ml")]
    let mut input = VRPSolverInput {
        locations: stops,
        num_vehicles,
        vehicle_capacity,
        objective,
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };

    #[cfg(not(feature = "ml"))]
    let input = VRPSolverInput {
        locations: stops,
        num_vehicles,
        vehicle_capacity,
        objective,
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };

    // AutoML: predict best hyperparameters for this instance
    #[cfg(feature = "ml")]
    let automl_used = {
        let features = InstanceFeatures::from_input(&input);
        let params = predict_hyperparams(&features);
        let used = params.model_used;
        input.hyperparams = Some(params);
        used
    };
    #[cfg(not(feature = "ml"))]
    let automl_used = false;

    tracing::info!("vrp_solve: {} stops, {} vehicles, solver={}, AutoML={}", 
        input.locations.len(), num_vehicles, solver_id, automl_used);

    let output = solve_with(&solver_id, &input).await
        .map_err(|e| anyhow::anyhow!("VRP solver failed: {}", e))?;

    // Convert routes to JSON
    let routes_json: Vec<Value> = output.routes.iter().flatten().enumerate().map(|(vi, route)| {
        let stops_json: Vec<Value> = route.iter().map(|s| {
            json!({
                "lat": s.lat,
                "lon": s.lon,
                "label": s.label,
                "demand": s.demand,
            })
        }).collect();
        json!({
            "vehicle": vi,
            "stops": stops_json,
        })
    }).collect();

    Ok(json!({
        "total_distance_km": output.total_distance_km,
        "total_time_min": output.total_time_min,
        "routes": routes_json,
        "unassigned": output.unassigned,
    }))
}

// ── Elevation Query handler ──────────────────────────────────────────────

fn handle_elevation_query(args: &Value) -> Result<Value> {
    let dem_path = args
        .get("dem_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'dem_path' parameter"))?;
    let points_val = args
        .get("points")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'points' parameter"))?;

    let dem = LocalDem::open(Path::new(dem_path))
        .map_err(|e| anyhow::anyhow!("Failed to open DEM file '{}': {}", dem_path, e))?;

    let points: Vec<(f64, f64)> = points_val
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let lon = p.get("lon").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Point {} missing 'lon'", i))?;
            let lat = p.get("lat").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Point {} missing 'lat'", i))?;
            Ok((lon, lat))
        })
        .collect::<Result<Vec<_>>>()?;

    let elevations = dem.get_elevations(&points)
        .map_err(|e| anyhow::anyhow!("Elevation query failed: {}", e))?;

    let results: Vec<Value> = points.iter().zip(elevations.iter()).enumerate().map(|(i, ((lon, lat), elev))| {
        json!({
            "index": i,
            "lon": *lon,
            "lat": *lat,
            "elevation_m": elev,
        })
    }).collect();

    Ok(json!({
        "dem_path": dem_path,
        "results": results,
    }))
}

// ── Elevation Profile handler ───────────────────────────────────────────

fn handle_elevation_profile(args: &Value) -> Result<Value> {
    let dem_path = args
        .get("dem_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'dem_path' parameter"))?;
    let route_val = args
        .get("route")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'route' parameter"))?;
    let sample_interval_m = args.get("sample_interval_m").and_then(|v| v.as_f64()).unwrap_or(100.0);

    let dem = LocalDem::open(Path::new(dem_path))
        .map_err(|e| anyhow::anyhow!("Failed to open DEM file '{}': {}", dem_path, e))?;

    let route: Vec<(f64, f64)> = route_val
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let lon = p.get("lon").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Route point {} missing 'lon'", i))?;
            let lat = p.get("lat").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Route point {} missing 'lat'", i))?;
            Ok((lon, lat))
        })
        .collect::<Result<Vec<_>>>()?;

    if route.len() < 2 {
        anyhow::bail!("Route must have at least 2 waypoints");
    }

    let profile = dem.route_profile(&route, sample_interval_m)
        .map_err(|e| anyhow::anyhow!("Elevation profile failed: {}", e))?;

    let points_json: Vec<Value> = profile.points.iter().map(|p| {
        json!({
            "distance_m": p.distance_m,
            "elevation_m": p.elevation_m,
            "lon": p.point.lon,
            "lat": p.point.lat,
        })
    }).collect();

    Ok(json!({
        "total_ascent_m": profile.total_ascent,
        "total_descent_m": profile.total_descent,
        "max_elevation_m": profile.max_elevation,
        "min_elevation_m": profile.min_elevation,
        "avg_elevation_m": profile.avg_elevation,
        "distance_km": profile.distance_km,
        "sample_interval_m": sample_interval_m,
        "num_samples": profile.points.len(),
        "points": points_json,
    }))
}

// ── List Solvers handler ───────────────────────────────────────────────────

fn handle_list_solvers(_args: &Value) -> Result<Value> {
    let options = v2rmp::core::vrp::registry::get_algorithm_options();
    let results: Vec<Value> = options.into_iter().map(|(id, label)| {
        json!({
            "id": id,
            "label": label
        })
    }).collect();

    Ok(json!({
        "solvers": results
    }))
}

// ── Haversine Distance handler ────────────────────────────────────────────

fn handle_haversine_distance(args: &Value) -> Result<Value> {
    let from = args
        .get("from")
        .ok_or_else(|| anyhow::anyhow!("Missing 'from' parameter"))?;
    let to = args
        .get("to")
        .ok_or_else(|| anyhow::anyhow!("Missing 'to' parameter"))?;

    let lat1 = from.get("lat").and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow::anyhow!("from.lat required"))?;
    let lon1 = from.get("lon").and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow::anyhow!("from.lon required"))?;
    let lat2 = to.get("lat").and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow::anyhow!("to.lat required"))?;
    let lon2 = to.get("lon").and_then(|v| v.as_f64())
        .ok_or_else(|| anyhow::anyhow!("to.lon required"))?;

    let dist_m = v2rmp::core::haversine_m(lat1, lon1, lat2, lon2);

    Ok(json!({
        "from": { "lat": lat1, "lon": lon1 },
        "to":   { "lat": lat2, "lon": lon2 },
        "distance_m": dist_m,
        "distance_km": dist_m / 1000.0,
    }))
}

// ── Elevation Stats handler ──────────────────────────────────────────────

fn handle_elevation_stats(args: &Value) -> Result<Value> {
    let dem_path = args
        .get("dem_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'dem_path' parameter"))?;

    let bbox_val = args
        .get("bbox")
        .ok_or_else(|| anyhow::anyhow!("Missing 'bbox' parameter"))?;
    let bbox = v2rmp::core::geo_types::BBox {
        min_lon: bbox_val.get("min_lon").and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.min_lon required"))?,
        min_lat: bbox_val.get("min_lat").and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.min_lat required"))?,
        max_lon: bbox_val.get("max_lon").and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.max_lon required"))?,
        max_lat: bbox_val.get("max_lat").and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow::anyhow!("bbox.max_lat required"))?,
    };

    let grid_step = args.get("grid_step").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

    let dem = LocalDem::open(Path::new(dem_path))
        .map_err(|e| anyhow::anyhow!("Failed to open DEM file '{}': {}", dem_path, e))?;

    let stats = dem.bbox_stats(bbox, grid_step)
        .map_err(|e| anyhow::anyhow!("Elevation stats failed: {}", e))?;

    Ok(json!({
        "min_elevation_m": stats.min_elevation,
        "max_elevation_m": stats.max_elevation,
        "avg_elevation_m": stats.avg_elevation,
        "coverage_percent": stats.coverage_percent,
        "pixel_count": stats.pixel_count,
    }))
}

// ── DEM Info handler ─────────────────────────────────────────────────────

fn handle_dem_info(args: &Value) -> Result<Value> {
    let dem_path = args
        .get("dem_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'dem_path' parameter"))?;

    let dem = LocalDem::open(Path::new(dem_path))
        .map_err(|e| anyhow::anyhow!("Failed to open DEM file '{}': {}", dem_path, e))?;

    let info = dem.info();

    Ok(json!({
        "width": info.width,
        "height": info.height,
        "bbox": {
            "min_lon": info.bbox.min_lon,
            "min_lat": info.bbox.min_lat,
            "max_lon": info.bbox.max_lon,
            "max_lat": info.bbox.max_lat,
        },
        "nodata": info.nodata,
        "pixel_size_x": info.pixel_size_x,
        "pixel_size_y": info.pixel_size_y,
    }))
}

// ── Fuel Estimate handler ────────────────────────────────────────────────

fn handle_fuel_estimate(args: &Value) -> Result<Value> {
    let samples_val = args
        .get("samples")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'samples' parameter"))?;

    if samples_val.len() < 2 {
        anyhow::bail!("At least 2 samples are required for fuel estimation");
    }

    let base_consumption = args
        .get("base_consumption")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.08);

    // Build an ElevationProfile from the samples
    let points: Vec<v2rmp::core::elevation::RouteElevationPoint> = samples_val
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let distance_m = s.get("distance_m").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Sample {} missing 'distance_m'", i))?;
            let elevation_m = s.get("elevation_m").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Sample {} missing 'elevation_m'", i))?;
            Ok(v2rmp::core::elevation::RouteElevationPoint {
                distance_m,
                elevation_m: Some(elevation_m),
                point: v2rmp::core::elevation::Point { lon: 0.0, lat: 0.0 },
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let total_distance_km = points.last().map(|p| p.distance_m / 1000.0).unwrap_or(0.0);

    let profile = v2rmp::core::elevation::ElevationProfile {
        points,
        total_ascent: 0.0, // Not needed for fuel calc
        total_descent: 0.0,
        max_elevation: 0.0,
        min_elevation: 0.0,
        avg_elevation: 0.0,
        distance_km: total_distance_km,
    };

    let consumption = FuelCalculator::calculate(&profile, base_consumption);

    Ok(json!({
        "total_fuel_l": consumption.total_fuel_l,
        "avg_consumption_l_per_km": consumption.avg_consumption_l_per_km,
        "elevation_penalty_l": consumption.elevation_penalty_l,
        "elevation_benefit_l": consumption.elevation_benefit_l,
        "base_consumption_l_per_km": base_consumption,
        "distance_km": total_distance_km,
    }))
}

// ── Inspect RMP handler ──────────────────────────────────────────────────

fn handle_inspect_rmp(args: &Value) -> Result<Value> {
    let input = args
        .get("input")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'input' parameter"))?;

    let file_data = std::fs::read(input)
        .map_err(|e| anyhow::anyhow!("Failed to read .rmp file: {}", e))?;

    let (nodes, edges) = v2rmp::core::optimize::read_rmp_file(&file_data)?;

    // Compute bounding box from nodes
    let (min_lon, min_lat, max_lon, max_lat) = if nodes.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        nodes.iter().fold(
            (f64::MAX, f64::MAX, f64::MIN, f64::MIN),
            |(mn_lon, mn_lat, mx_lon, mx_lat), n| {
                (mn_lon.min(n.lon), mn_lat.min(n.lat), mx_lon.max(n.lon), mx_lat.max(n.lat))
            },
        )
    };

    let total_edge_length_km: f64 = edges.iter().map(|e| e.weight_m / 1000.0).sum();

    Ok(json!({
        "node_count": nodes.len(),
        "edge_count": edges.len(),
        "bbox": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        },
        "total_edge_length_km": total_edge_length_km,
        "file_size_bytes": file_data.len(),
    }))
}

// ── Predict Solver handler ───────────────────────────────────────────────

fn handle_predict_solver(args: &Value) -> Result<Value> {
    #[cfg(not(feature = "ml"))]
    {
        let _ = args;
        anyhow::bail!("ML feature is not enabled. Cannot use predict_solver.");
    }

    #[cfg(feature = "ml")]
    {
        let stops_val = args
            .get("stops")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing 'stops' parameter"))?;

        let stops: Vec<VRPSolverStop> = stops_val
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let lat = s.get("lat").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lat'", i))?;
                let lon = s.get("lon").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lon'", i))?;
                let label = s.get("label").and_then(|v| v.as_str())
                    .unwrap_or_else(|| "")
                    .to_string();
                let demand = s.get("demand").and_then(|v| v.as_f64());
                Ok(VRPSolverStop {
                    lat,
                    lon,
                    label: if label.is_empty() { format!("Stop {}", i) } else { label },
                    demand,
                    arrival_time: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let num_vehicles = args.get("num_vehicles").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let vehicle_capacity = args.get("vehicle_capacity").and_then(|v| v.as_f64()).unwrap_or(100.0);
        let objective = match args.get("objective").and_then(|v| v.as_str()).unwrap_or("min_distance") {
            "min_time" => VrpObjective::MinTime,
            "balance_load" => VrpObjective::BalanceLoad,
            "min_vehicles" => VrpObjective::MinVehicles,
            _ => VrpObjective::MinDistance,
        };

        let input = VRPSolverInput {
            locations: stops,
            num_vehicles,
            vehicle_capacity,
            objective,
            matrix: None,
            service_time_secs: None,
            use_time_windows: false,
            window_open: None,
            window_close: None, hyperparams: None,
        };

        let model_path = default_model_path();
        let pred = predict_solver(&input, Some(&model_path))?;

        let all_scores_json: Vec<Value> = pred.all_scores.iter().map(|(id, score)| {
            json!({"solver_id": id, "score": score})
        }).collect();

        let inst_features = InstanceFeatures::from_input(&input);

        Ok(json!({
            "recommended": pred.recommended,
            "confidence": pred.confidence,
            "runner_up": pred.runner_up.as_ref().map(|(id, score)| json!({"solver_id": id, "score": score})),
            "all_scores": all_scores_json,
            "features": {
                "num_stops": input.locations.len().saturating_sub(1),
                "num_vehicles": input.num_vehicles,
                "objective": format!("{:?}", input.objective),
            },
            "instance_feature_vector": inst_features.to_vector(),
            "model_loaded": model_path.exists(),
        }))
    }
}

// ── Score Route handler ──────────────────────────────────────────────────

fn handle_score_route(args: &Value) -> Result<Value> {
    let stops_val = args
        .get("stops")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'stops' parameter"))?;

    let stops: Vec<VRPSolverStop> = stops_val
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let lat = s.get("lat").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lat'", i))?;
            let lon = s.get("lon").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lon'", i))?;
            let label = s.get("label").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let demand = s.get("demand").and_then(|v| v.as_f64());
            Ok(VRPSolverStop { lat, lon, label, demand, arrival_time: None })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_vehicles = args.get("num_vehicles").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let vehicle_capacity = args.get("vehicle_capacity").and_then(|v| v.as_f64()).unwrap_or(100.0);
    let objective = match args.get("objective").and_then(|v| v.as_str()).unwrap_or("min_distance") {
        "min_time" => VrpObjective::MinTime,
        "balance_load" => VrpObjective::BalanceLoad,
        "min_vehicles" => VrpObjective::MinVehicles,
        _ => VrpObjective::MinDistance,
    };

    let input = VRPSolverInput {
        locations: stops.clone(),
        num_vehicles,
        vehicle_capacity,
        objective,
        matrix: None,
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None, hyperparams: None,
    };

    let routes_indices = args
        .get("routes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'routes' parameter"))?;

    let routes: Vec<Vec<VRPSolverStop>> = routes_indices
        .iter()
        .map(|route_val| {
            let indices = route_val.as_array()
                .ok_or_else(|| anyhow::anyhow!("Each route must be an array of indices"))?;
            let route_stops: Vec<VRPSolverStop> = indices
                .iter()
                .filter_map(|iv| iv.as_u64().and_then(|idx| stops.get(idx as usize).cloned()))
                .collect();
            Ok(route_stops)
        })
        .collect::<Result<Vec<_>>>()?;

    let total_distance_km = args
        .get("total_distance_km")
        .and_then(|v| v.as_str())
        .unwrap_or("0.0")
        .to_string();

    let output = VRPSolverOutput {
        stops: routes.iter().flatten().cloned().collect(),
        routes: Some(routes),
        total_distance_km,
        total_time_min: 0,
        route_stats: None,
        route_metrics: None,
        unassigned: None,
    };

    let score = score_route(&input, &output);

    Ok(json!({
        "overall": score.overall,
        "distance_efficiency": score.distance_efficiency,
        "load_balance": score.load_balance,
        "turn_quality": score.turn_quality,
        "coverage": score.coverage,
    }))
}

// ── Route Embedding handler ──────────────────────────────────────────────

fn handle_route_embedding(args: &Value) -> Result<Value> {
    let stops_val = args
        .get("stops")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'stops' parameter"))?;

    let stops: Vec<VRPSolverStop> = stops_val
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let lat = s.get("lat").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lat'", i))?;
            let lon = s.get("lon").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lon'", i))?;
            let label = s.get("label").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let demand = s.get("demand").and_then(|v| v.as_f64());
            Ok(VRPSolverStop { lat, lon, label, demand, arrival_time: None })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_vehicles = args.get("num_vehicles").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let vehicle_capacity = args.get("vehicle_capacity").and_then(|v| v.as_f64()).unwrap_or(100.0);
    let objective = match args.get("objective").and_then(|v| v.as_str()).unwrap_or("min_distance") {
        "min_time" => VrpObjective::MinTime,
        "balance_load" => VrpObjective::BalanceLoad,
        "min_vehicles" => VrpObjective::MinVehicles,
        _ => VrpObjective::MinDistance,
    };

    let input = VRPSolverInput {
        locations: stops,
        num_vehicles,
        vehicle_capacity,
        objective,
        matrix: None,
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None, hyperparams: None,
    };

    let features = RouteFeatures::from_input(&input);
    let vector = route_feature_vector(&features);

    Ok(json!({
        "vector": vector,
        "dimension": vector.len(),
        "features": {
            "num_stops": features.num_stops,
            "num_vehicles": features.num_vehicles,
            "avg_pairwise_km": features.avg_pairwise_km,
            "lat_spread": features.lat_spread,
            "lon_spread": features.lon_spread,
            "density": features.density,
            "tight_capacity": features.tight_capacity,
            "objective": format!("{:?}", features.objective),
        }
    }))
}

// ── Pipeline handler ─────────────────────────────────────────────────────

async fn handle_pipeline(args: &Value) -> Result<Value> {
    let bbox = parse_bbox(args)?;
    let output_dir = args
        .get("output_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("./pipeline-output")
        .to_string();
    let source = match args.get("source").and_then(|v| v.as_str()).unwrap_or("overture") {
        "osm" => ExtractSource::Osm,
        _ => ExtractSource::Overture,
    };
    let pbf_path = args.get("pbf_path").and_then(|v| v.as_str()).map(String::from);
    let depot = args.get("depot").and_then(|d| {
        let lat = d.get("lat")?.as_f64()?;
        let lon = d.get("lon")?.as_f64()?;
        Some((lat, lon))
    });
    let mode = parse_solver_mode(args);
    let prune_disconnected = args.get("prune_disconnected").and_then(|v| v.as_bool()).unwrap_or(false);

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    let extract_path = format!("{}/extract.geojson", output_dir);
    let cleaned_path = format!("{}/cleaned.geojson", output_dir);
    let rmp_path = format!("{}/network.rmp", output_dir);
    let route_path = format!("{}/route.gpx", output_dir);

    // Stage 1: Extract
    tracing::info!("pipeline stage 1: extract");
    let extract_req = ExtractRequest {
        source,
        bbox,
        road_classes: RoadClass::all_vehicle(),
        output_path: extract_path.clone(),
        pbf_path,
    };
    let extract_result = v2rmp::core::extract::run_extract(&extract_req).await?;

    // Stage 2: Clean
    tracing::info!("pipeline stage 2: clean");
    let input_data = std::fs::read_to_string(&extract_path)?;
    let geojson: geojson::FeatureCollection = serde_json::from_str(&input_data)?;
    let (cleaned, clean_stats, _warnings) = clean_geojson(&geojson, &CleanOptions::default())?;
    let cleaned_str = serde_json::to_string_pretty(&cleaned)?;
    std::fs::write(&cleaned_path, &cleaned_str)?;

    // Stage 3: Compile
    tracing::info!("pipeline stage 3: compile");
    let compile_req = CompileRequest {
        input_geojson: cleaned_path.clone(),
        output_rmp: rmp_path.clone(),
        compress: false,
        road_classes: vec![],
        clean_options: None,
        prune_disconnected,
    };
    let compile_result = v2rmp::core::compile::run_compile(&compile_req)?;

    // Stage 4: Optimize
    tracing::info!("pipeline stage 4: optimize");
    let num_vehicles = args.get("num_vehicles").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let solver_id = args.get("solver_id").and_then(|v| v.as_str()).unwrap_or("clarke_wright").to_string();
let optimize_req = OptimizeRequest {
    cache_file: rmp_path.clone(),
    route_file: Some(route_path.clone()),
    turn_penalties: TurnPenalties::default(),
    depot,
    oneway_mode: OnewayMode::default(),
    mode,
    num_vehicles,
    solver_id,
    coordinates: None,
};
    let optimize_result = v2rmp::core::optimize::run_optimize(&optimize_req).await?;

    Ok(json!({
        "output_dir": output_dir,
        "extract": {
            "nodes": extract_result.nodes,
            "edges": extract_result.edges,
            "total_km": extract_result.total_km,
        },
        "clean": {
            "input_features": clean_stats.input_features,
            "output_features": clean_stats.output_features,
            "total_removed": clean_stats.total_removed(),
        },
        "compile": {
            "node_count": compile_result.node_count,
            "edge_count": compile_result.edge_count,
            "output_size_bytes": compile_result.output_size_bytes,
        },
        "optimize": {
            "total_distance_km": optimize_result.total_distance_km,
            "total_segments": optimize_result.total_segments,
            "deadhead_distance_km": optimize_result.deadhead_distance_km,
            "efficiency_pct": optimize_result.efficiency_pct,
            "num_routes": optimize_result.num_routes,
        },
    }))
}

// ── Predict Quality handler ────────────────────────────────────────────

fn handle_predict_quality(args: &Value) -> Result<Value> {
    #[cfg(not(feature = "ml"))]
    {
        let _ = args;
        anyhow::bail!("ML feature is not enabled. Cannot use predict_quality.");
    }

    #[cfg(feature = "ml")]
    {
        let stops_val = args
            .get("stops")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing 'stops' parameter"))?;

        let stops: Vec<VRPSolverStop> = stops_val
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let lat = s.get("lat").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lat'", i))?;
                let lon = s.get("lon").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lon'", i))?;
                let label = s.get("label").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let demand = s.get("demand").and_then(|v| v.as_f64());
                Ok(VRPSolverStop { lat, lon, label, demand, arrival_time: None })
            })
            .collect::<Result<Vec<_>>>()?;

        let num_vehicles = args.get("num_vehicles").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let vehicle_capacity = args.get("vehicle_capacity").and_then(|v| v.as_f64()).unwrap_or(100.0);
        let objective = match args.get("objective").and_then(|v| v.as_str()).unwrap_or("min_distance") {
            "min_time" => VrpObjective::MinTime,
            "balance_load" => VrpObjective::BalanceLoad,
            "min_vehicles" => VrpObjective::MinVehicles,
            _ => VrpObjective::MinDistance,
        };

        let input = VRPSolverInput {
            locations: stops,
            num_vehicles,
            vehicle_capacity,
            objective,
            matrix: None,
            service_time_secs: None,
            use_time_windows: false,
            window_open: None,
            window_close: None, hyperparams: None,
        };

        let features = InstanceFeatures::from_input(&input);
        let pred = predict_quality(&features);

        Ok(json!({
            "predicted_gap_pct": pred.predicted_gap_pct,
            "predicted_tour_length_km": pred.predicted_tour_length_km,
            "confidence": pred.confidence,
            "feature_vector": features.to_vector(),
        }))
    }
}

// ── Tune Hyperparams handler ─────────────────────────────────────────────

fn handle_tune_hyperparams(args: &Value) -> Result<Value> {
    #[cfg(not(feature = "ml"))]
    {
        let _ = args;
        anyhow::bail!("ML feature is not enabled. Cannot use tune_hyperparams.");
    }

    #[cfg(feature = "ml")]
    {
        let stops_val = args
            .get("stops")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing 'stops' parameter"))?;

        let stops: Vec<VRPSolverStop> = stops_val
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let lat = s.get("lat").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lat'", i))?;
                let lon = s.get("lon").and_then(|v| v.as_f64())
                    .ok_or_else(|| anyhow::anyhow!("Stop {} missing 'lon'", i))?;
                let label = s.get("label").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let demand = s.get("demand").and_then(|v| v.as_f64());
                Ok(VRPSolverStop { lat, lon, label, demand, arrival_time: None })
            })
            .collect::<Result<Vec<_>>>()?;

        let num_vehicles = args.get("num_vehicles").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let vehicle_capacity = args.get("vehicle_capacity").and_then(|v| v.as_f64()).unwrap_or(100.0);
        let objective = match args.get("objective").and_then(|v| v.as_str()).unwrap_or("min_distance") {
            "min_time" => VrpObjective::MinTime,
            "balance_load" => VrpObjective::BalanceLoad,
            "min_vehicles" => VrpObjective::MinVehicles,
            _ => VrpObjective::MinDistance,
        };

        let input = VRPSolverInput {
            locations: stops,
            num_vehicles,
            vehicle_capacity,
            objective,
            matrix: None,
            service_time_secs: None,
            use_time_windows: false,
            window_open: None,
            window_close: None, hyperparams: None,
        };

        let features = InstanceFeatures::from_input(&input);
        let params = predict_hyperparams(&features);

        Ok(json!({
            "max_iterations": params.max_iterations,
            "temperature": params.temperature,
            "tabu_tenure": params.tabu_tenure,
            "cooling_rate": params.cooling_rate,
            "neighbourhood_radius": params.neighbourhood_radius,
            "feature_vector": features.to_vector(),
        }))
    }
}

// ── Parse Routing Query handler ──────────────────────────────────────────

fn handle_parse_routing_query(args: &Value) -> Result<Value> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing 'query' parameter"))?;

    let use_llm = args.get("use_llm").and_then(|v| v.as_bool()).unwrap_or(false);

    #[cfg(feature = "ml")]
    {
        if use_llm {
            tracing::info!("parse_routing_query: using LLM for query='{}'", query);
            let mut parser = QwenNLParser::new()?;
            let json_str = parser.parse_llm(query)?;
            
            // Try to parse the LLM output as JSON. If it fails, fallback to regex.
            match serde_json::from_str::<Value>(&json_str) {
                Ok(json) => {
                    return Ok(json!({
                        "variant": json.get("variant").and_then(|v| v.as_str()).unwrap_or("cvrp"),
                        "config": json,
                        "method": "llm",
                    }));
                }
                Err(e) => {
                    tracing::warn!("LLM output was not valid JSON: {}. Falling back to regex.", e);
                }
            }
        }

        let parsed = parse_query(query);
        let json = to_vrp_json(&parsed);

        return Ok(json!({
            "variant": parsed.variant,
            "config": json,
            "entities": parsed.entities,
            "method": "regex",
        }));
    }

    #[cfg(not(feature = "ml"))]
    {
        if use_llm {
            anyhow::bail!("ML feature is not enabled. Cannot use LLM parser.");
        }
        anyhow::bail!("ML feature is not enabled. NLP requires the 'ml' feature.");
    }
}

// ── Get Valhalla Matrix handler ──────────────────────────────────────────

async fn handle_get_valhalla_matrix(args: &Value) -> Result<Value> {
    let locs_val = args
        .get("locations")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("Missing 'locations' parameter"))?;

    if locs_val.is_empty() {
        anyhow::bail!("At least one location is required");
    }

    let locations: Vec<VRPSolverStop> = locs_val
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let lat = l.get("lat").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Location {} missing 'lat'", i))?;
            let lon = l.get("lon").and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("Location {} missing 'lon'", i))?;
            Ok(VRPSolverStop {
                lat,
                lon,
                label: format!("Loc {}", i),
                demand: None,
                arrival_time: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    tracing::info!("get_valhalla_matrix: {} locations", locations.len());

    let matrix = get_valhalla_matrix(&locations).await
        .map_err(|e| anyhow::anyhow!("Valhalla matrix fetch failed: {}", e))?;

    let matrix_json: Vec<Vec<Value>> = matrix.iter().map(|row| {
        row.iter().map(|cell| {
            json!({
                "distance_km": cell.distance,
                "time_seconds": cell.time,
            })
        }).collect()
    }).collect();

    Ok(json!({
        "size": locations.len(),
        "matrix": matrix_json,
    }))
}

// ── Main loop ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!(
        "rmpca-mcp-server starting (v{})",
        env!("CARGO_PKG_VERSION")
    );

    let tools = tool_definitions();
    let tool_list_json: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema
            })
        })
        .collect();

    let stdin = std::io::stdin().lock();
    for line in stdin.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                send_err(&Value::Null, -32700, &format!("Parse error: {e}"));
                continue;
            }
        };

        if req.jsonrpc != "2.0" {
            send_err(&req.id, -32600, "Invalid Request: jsonrpc must be 2.0");
            continue;
        }

        match req.method.as_str() {
            "initialize" => {
                send(
                    &req.id,
                    json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": {
                            "name": "rmpca-mcp-server",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }),
                );
            }

            "tools/list" => {
                send(&req.id, json!({ "tools": tool_list_json }));
            }

            "tools/call" => {
                let name = req
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args = req
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::Null);

                let result = match name {
                    "extract_overture" => handle_extract_overture(&args).await,
                    "extract_osm" => handle_extract_osm(&args).await,
                    "compile" => handle_compile(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "optimize" => handle_optimize(&args).await,
                    "clean" => handle_clean(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "vrp_solve" => handle_vrp_solve(&args).await,
                    "elevation_query" => handle_elevation_query(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "elevation_profile" => handle_elevation_profile(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "list_solvers" => handle_list_solvers(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "haversine_distance" => handle_haversine_distance(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "elevation_stats" => handle_elevation_stats(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "dem_info" => handle_dem_info(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "fuel_estimate" => handle_fuel_estimate(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "inspect_rmp" => handle_inspect_rmp(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "pipeline" => handle_pipeline(&args).await,
                    "get_valhalla_matrix" => handle_get_valhalla_matrix(&args).await,
                    "predict_solver" => handle_predict_solver(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "score_route" => handle_score_route(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "route_embedding" => handle_route_embedding(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "predict_quality" => handle_predict_quality(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "tune_hyperparams" => handle_tune_hyperparams(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    "parse_routing_query" => handle_parse_routing_query(&args)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .map(|v| v),
                    other => {
                        send_err(&req.id, -32602, &format!("Unknown tool: {other}"));
                        continue;
                    }
                };

                tool_success(&req.id, &result);
            }

            "notifications/initialized" | "initialized" => {}

            other => {
                send_err(&req.id, -32601, &format!("Method not found: {other}"));
            }
        }
    }

    Ok(())
}
