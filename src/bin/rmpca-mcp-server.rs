//! rmpca-mcp-server — MCP server exposing the route optimization pipeline.
//!
//! Tools: extract_overture, extract_osm, compile, optimize
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
use v2rmp::core::compile::{CompileRequest, CompileResult};
use v2rmp::core::extract::{BBoxRequest, ExtractRequest, ExtractResult, ExtractSource, RoadClass};
use v2rmp::core::optimize::{
    OnewayMode, OptimizeRequest, OptimizeResult, SolverMode, TurnPenalties,
};

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
    };

    let result: OptimizeResult = v2rmp::core::optimize::run_optimize(&req).await?;
    Ok(serde_json::to_value(result)?)
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
