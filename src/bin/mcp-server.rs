use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use v2rmp::core::compile::{run_compile, CompileRequest};
#[cfg(feature = "extract")]
use v2rmp::core::extract::{BBoxRequest, ExtractRequest, ExtractSource, RoadClass};
use v2rmp::core::optimize::{run_optimize, OnewayMode, OptimizeRequest, SolverMode, TurnPenalties};
use v2rmp::core::postgis_cpp::{run_postgis_cpp, PostGisCppRequest};
use v2rmp::core::r2::R2Storage;

fn main() -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };

        if let Some(response) = handle_message(message) {
            let payload = serde_json::to_string(&response)?;
            writeln!(stdout, "{}", payload)?;
            stdout.flush()?;
        }
    }

    Ok(())
}

fn handle_message(message: Value) -> Option<Value> {
    let method = message.get("method").and_then(Value::as_str)?;
    let id = message.get("id").cloned();

    match method {
        "initialize" => respond(
            id.as_ref(),
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "v2rmp-mcp-server",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "capabilities": {
                    "tools": {},
                }
            }),
        ),
        "tools/list" => respond(
            id.as_ref(),
            json!({
                "tools": [
                    {
                        "name": "list_r2_bucket",
                        "description": "List all objects available in the configured Cloudflare R2 bucket.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "bucket": { "type": "string" },
                                "prefix": { "type": "string" }
                            }
                        }
                    },
                    {
                        "name": "upload_to_r2",
                        "description": "Upload a local file to the R2 bucket.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "local_path": { "type": "string" },
                                "r2_path": { "type": "string" },
                                "bucket": { "type": "string" }
                            },
                            "required": ["local_path", "r2_path"]
                        }
                    },
                    {
                        "name": "download_from_r2",
                        "description": "Download an object from the R2 bucket to a local file.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "r2_path": { "type": "string" },
                                "local_path": { "type": "string" },
                                "bucket": { "type": "string" }
                            },
                            "required": ["r2_path", "local_path"]
                        }
                    },
                    {
                        "name": "query_supabase",
                        "description": "Execute a SQL query against the Supabase database.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "string" }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "v2rmp_extract",
                        "description": "Extract road network data from Overture, OSM, or Postgres.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "source": { "type": "string", "enum": ["overture", "osm", "postgres", "r2"] },
                                "min_lon": { "type": "number" },
                                "min_lat": { "type": "number" },
                                "max_lon": { "type": "number" },
                                "max_lat": { "type": "number" },
                                "output_path": { "type": "string" }
                            },
                            "required": ["source", "min_lon", "min_lat", "max_lon", "max_lat", "output_path"]
                        }
                    },
                    {
                        "name": "v2rmp_compile",
                        "description": "Compile GeoJSON into .rmp binary format with optional cleaning.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "input_geojson": { "type": "string" },
                                "output_rmp": { "type": "string" },
                                "remove_isolates": { "type": "boolean" }
                            },
                            "required": ["input_geojson", "output_rmp"]
                        }
                    },
                    {
                        "name": "v2rmp_optimize",
                        "description": "Optimize a route on an .rmp map.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "map_path": { "type": "string" },
                                "u_turn_penalty": { "type": "number" },
                                "depot_lat": { "type": "number" },
                                "depot_lon": { "type": "number" },
                                "output_route": { "type": "string" },
                                "db_export_table": { "type": "string" }
                            },
                            "required": ["map_path", "output_route"]
                        }
                    },
                    {
                        "name": "v2rmp_postgis_cpp",
                        "description": "Solve Chinese Postman Problem directly from a PostGIS road_edges table. Extracts graph topology via SQL, runs optimal Blossom V matching for deadheading, and returns a GeoJSON route with turn-by-turn instructions.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "bbox_min_lon": { "type": "number", "description": "Bounding box west longitude" },
                                "bbox_min_lat": { "type": "number", "description": "Bounding box south latitude" },
                                "bbox_max_lon": { "type": "number", "description": "Bounding box east longitude" },
                                "bbox_max_lat": { "type": "number", "description": "Bounding box north latitude" },
                                "road_classes": { "type": "array", "items": { "type": "string" }, "description": "Road classes to include (default: residential, tertiary, secondary, unclassified, living_street, road)" },
                                "oneway_mode": { "type": "string", "enum": ["ignore", "respect", "reverse"], "description": "How to handle one-way streets (default: respect)" },
                                "left_turn_penalty": { "type": "number", "description": "Left turn penalty in meters (default: 50)" },
                                "right_turn_penalty": { "type": "number", "description": "Right turn penalty in meters (default: 0)" },
                                "u_turn_penalty": { "type": "number", "description": "U-turn penalty in meters (default: 500)" },
                                "depot_lat": { "type": "number", "description": "Optional depot/start latitude" },
                                "depot_lon": { "type": "number", "description": "Optional depot/start longitude" },
                                "database_url": { "type": "string", "description": "PostgreSQL connection URL (falls back to DATABASE_URL env var)" },
                                "table_name": { "type": "string", "description": "PostGIS table name (default: road_edges)" },
                                "output_path": { "type": "string", "description": "Optional file path to write JSON result" }
                            },
                            "required": ["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]
                        }
                    },
                    {
                        "name": "v2rmp_neural_optimize",
                        "description": "Optimize a route using a Neural Network model (ONNX). Best for complex VRP problems with capacity constraints.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "model_path": { "type": "string", "description": "Path to the .onnx model file" },
                                "locations": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": { "type": "number" },
                                        "minItems": 2,
                                        "maxItems": 2
                                    },
                                    "description": "List of [lat, lon] coordinates. Index 0 is the depot."
                                },
                                "demands": { "type": "array", "items": { "type": "number" }, "description": "Demands for each location (0 for depot)" },
                                "capacity": { "type": "number", "description": "Vehicle capacity" }
                            },
                            "required": ["model_path", "locations", "demands", "capacity"]
                        }
                    },
                    {
                        "name": "v2rmp_generate_osmand_link",
                        "description": "Generate an OsmAnd deep link for importing a GPX file from a public URL.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "gpx_url": { "type": "string", "description": "Publicly accessible URL of the GPX file" }
                            },
                            "required": ["gpx_url"]
                        }
                    }
                ]
            }),
        ),
        "resources/list" => respond(id.as_ref(), json!({ "resources": [] })),
        "resources/templates/list" => respond(id.as_ref(), json!({ "resourceTemplates": [] })),
        "notifications/initialized" => None,
        "tools/call" => {
            let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
            match handle_tool_call(params) {
                Ok(result) => respond(id.as_ref(), result),
                Err(err) => respond_error(id.as_ref(), -32000, err.to_string()),
            }
        }
        "ping" => respond(id.as_ref(), json!({})),
        _ => respond_error(id.as_ref(), -32601, format!("Method not found: {}", method)),
    }
}

fn handle_tool_call(params: Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .context("Missing tool name")?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    match name {
        "list_r2_bucket" => {
            let bucket = arguments
                .get("bucket")
                .and_then(Value::as_str)
                .unwrap_or("v2rmp");
            let prefix = arguments.get("prefix").and_then(Value::as_str);
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage = R2Storage::from_env(bucket)?;
                let objects = storage.list_objects(prefix).await?;
                let val = json!({ "objects": objects });
                Ok(json!({
                    "content": [{ "type": "text", "text": format!("Objects:\n{}", objects.join("\n")) }],
                    "structured": val,
                    "isError": false
                }))
            })
        }
        "upload_to_r2" => {
            let local_path = arguments
                .get("local_path")
                .and_then(Value::as_str)
                .context("Missing local_path")?;
            let r2_path = arguments
                .get("r2_path")
                .and_then(Value::as_str)
                .context("Missing r2_path")?;
            let bucket = arguments
                .get("bucket")
                .and_then(Value::as_str)
                .unwrap_or("v2rmp");
            let data = std::fs::read(local_path)?;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage = R2Storage::from_env(bucket)?;
                storage.upload_object(r2_path, data).await?;
                let val = json!({ "status": "success", "r2_path": r2_path });
                Ok(json!({
                    "content": [{ "type": "text", "text": format!("Uploaded to {}", r2_path) }],
                    "structured": val,
                    "isError": false
                }))
            })
        }
        "download_from_r2" => {
            let r2_path = arguments
                .get("r2_path")
                .and_then(Value::as_str)
                .context("Missing r2_path")?;
            let local_path = arguments
                .get("local_path")
                .and_then(Value::as_str)
                .context("Missing local_path")?;
            let bucket = arguments
                .get("bucket")
                .and_then(Value::as_str)
                .unwrap_or("v2rmp");
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                let storage = R2Storage::from_env(bucket)?;
                let data = storage.download_object(r2_path).await?;
                std::fs::write(local_path, data)?;
                let val = json!({ "status": "success", "local_path": local_path });
                Ok(json!({
                    "content": [{ "type": "text", "text": format!("Downloaded to {}", local_path) }],
                    "structured": val,
                    "isError": false
                }))
            })
        }
        "query_supabase" => {
            let sql_query = arguments
                .get("query")
                .and_then(Value::as_str)
                .context("Missing SQL query")?;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                dotenvy::dotenv().ok();
                let db_url = std::env::var("SUPABASE_DB_URL")?;
                let pool = sqlx::PgPool::connect(&db_url).await?;
                let mut tx = pool.begin().await?;
                sqlx::query("SET TRANSACTION READ ONLY")
                    .execute(&mut *tx)
                    .await?;
                let rows = sqlx::query(sql_query).fetch_all(&mut *tx).await?;
                let mut results = Vec::new();
                for row in rows {
                    use sqlx::{Column, Row, TypeInfo};
                    let mut res_row = serde_json::Map::new();
                    for col in row.columns() {
                        let name = col.name();
                        let val: Value = match col.type_info().name() {
                            "TEXT" | "VARCHAR" | "NAME" => row
                                .get::<Option<String>, _>(name)
                                .map(Value::String)
                                .unwrap_or(Value::Null),
                            "INT4" | "INTEGER" => row
                                .get::<Option<i32>, _>(name)
                                .map(|n| json!(n))
                                .unwrap_or(Value::Null),
                            "INT8" | "BIGINT" => row
                                .get::<Option<i64>, _>(name)
                                .map(|n| json!(n))
                                .unwrap_or(Value::Null),
                            _ => json!("<type not displayed>"),
                        };
                        res_row.insert(name.to_string(), val);
                    }
                    results.push(Value::Object(res_row));
                }
                let _ = tx.rollback().await;
                let val = json!({ "results": results });
                Ok(json!({
                    "content": [{ "type": "text", "text": serde_json::to_string(&results)? }],
                    "structured": val,
                    "isError": false
                }))
            })
        }
        "v2rmp_extract" => {
            #[cfg(feature = "extract")]
            {
                let source = match arguments
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or("overture")
                {
                    "osm" => ExtractSource::Osm,
                    "postgres" => ExtractSource::Postgres,
                    "r2" => ExtractSource::R2,
                    _ => ExtractSource::Overture,
                };
                let req = ExtractRequest {
                    source,
                    bbox: BBoxRequest {
                        min_lon: arguments
                            .get("min_lon")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0),
                        min_lat: arguments
                            .get("min_lat")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0),
                        max_lon: arguments
                            .get("max_lon")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0),
                        max_lat: arguments
                            .get("max_lat")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0),
                    },
                    road_classes: RoadClass::all_vehicle(),
                    output_path: arguments
                        .get("output_path")
                        .and_then(Value::as_str)
                        .unwrap_or("out.geojson")
                        .to_string(),
                    database_url: None,
                    table_name: None,
                    r2_bucket: None,
                    r2_access_key_id: None,
                    r2_secret_access_key: None,
                    r2_endpoint: None,
                };
                let res = v2rmp::core::extract::run_extract(&req)?;
                let val = json!(res);
                Ok(json!({
                    "content": [{ "type": "text", "text": format!("Extracted {} nodes", res.nodes) }],
                    "structured": val,
                    "isError": false
                }))
            }
            #[cfg(not(feature = "extract"))]
            {
                anyhow::bail!("The 'extract' feature is not enabled in this build of the server.")
            }
        }
        "v2rmp_compile" => {
            let opts = v2rmp::core::clean::CleanOptions {
                remove_isolates: arguments
                    .get("remove_isolates")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                ..Default::default()
            };
            let req = CompileRequest {
                input_geojson: arguments
                    .get("input_geojson")
                    .and_then(Value::as_str)
                    .context("Missing input")?
                    .to_string(),
                output_rmp: arguments
                    .get("output_rmp")
                    .and_then(Value::as_str)
                    .context("Missing output")?
                    .to_string(),
                compress: true,
                road_classes: vec![],
                clean_options: Some(opts),
                prune_disconnected: arguments
                    .get("prune_disconnected")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            };
            let res = run_compile(&req)?;
            let val = json!(res);
            Ok(json!({
                "content": [{ "type": "text", "text": format!("Compiled {} nodes", res.node_count) }],
                "structured": val,
                "isError": false
            }))
        }
        "v2rmp_optimize" => {
            let penalties = TurnPenalties {
                u_turn: arguments
                    .get("u_turn_penalty")
                    .and_then(Value::as_f64)
                    .unwrap_or(10.0),
                ..Default::default()
            };
            let depot = if let (Some(lat), Some(lon)) = (
                arguments.get("depot_lat").and_then(Value::as_f64),
                arguments.get("depot_lon").and_then(Value::as_f64),
            ) {
                Some((lat, lon))
            } else {
                None
            };

            let req = OptimizeRequest {
                cache_file: arguments
                    .get("map_path")
                    .and_then(Value::as_str)
                    .context("Missing map")?
                    .to_string(),
                route_file: Some(
                    arguments
                        .get("output_route")
                        .and_then(Value::as_str)
                        .context("Missing output")?
                        .to_string(),
                ),
                turn_penalties: penalties,
                depot,
                oneway_mode: OnewayMode::Respect,
                mode: SolverMode::Cpp,
                num_vehicles: 1,
                solver_id: "default".to_string(),
                coordinates: None,
            };

            let rt = tokio::runtime::Runtime::new()?;
            let res = rt.block_on(async { run_optimize(&req).await })?;
            let val = json!(res);
            Ok(json!({
                "content": [{ "type": "text", "text": format!("Optimized: {:.2} km", res.total_distance_km) }],
                "structured": val,
                "isError": false
            }))
        }
        "v2rmp_postgis_cpp" => {
            let depot = if let (Some(lat), Some(lon)) = (
                arguments.get("depot_lat").and_then(Value::as_f64),
                arguments.get("depot_lon").and_then(Value::as_f64),
            ) {
                Some((lat, lon))
            } else {
                None
            };

            let road_classes: Vec<String> = arguments
                .get("road_classes")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let oneway_mode = match arguments
                .get("oneway_mode")
                .and_then(Value::as_str)
                .unwrap_or("respect")
            {
                "ignore" => v2rmp::core::postgis_cpp::OneWayMode::Ignore,
                "reverse" => v2rmp::core::postgis_cpp::OneWayMode::Reverse,
                _ => v2rmp::core::postgis_cpp::OneWayMode::Respect,
            };

            let req = PostGisCppRequest {
                bbox: [
                    arguments
                        .get("bbox_min_lon")
                        .and_then(Value::as_f64)
                        .context("Missing bbox_min_lon")?,
                    arguments
                        .get("bbox_min_lat")
                        .and_then(Value::as_f64)
                        .context("Missing bbox_min_lat")?,
                    arguments
                        .get("bbox_max_lon")
                        .and_then(Value::as_f64)
                        .context("Missing bbox_max_lon")?,
                    arguments
                        .get("bbox_max_lat")
                        .and_then(Value::as_f64)
                        .context("Missing bbox_max_lat")?,
                ],
                road_classes,
                oneway_mode,
                turn_penalties: v2rmp::core::postgis_cpp::TurnPenalties {
                    left: arguments
                        .get("left_turn_penalty")
                        .and_then(Value::as_f64)
                        .unwrap_or(50.0),
                    right: arguments
                        .get("right_turn_penalty")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0),
                    u_turn: arguments
                        .get("u_turn_penalty")
                        .and_then(Value::as_f64)
                        .unwrap_or(500.0),
                },
                depot,
                database_url: arguments
                    .get("database_url")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                table_name: arguments
                    .get("table_name")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                output_path: arguments
                    .get("output_path")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
            };

            let res = run_postgis_cpp(&req)?;
            let val = json!(res);
            Ok(json!({
                "content": [{ "type": "text", "text": serde_json::to_string(&res)? }],
                "structured": val,
                "isError": false
            }))
        }
        "v2rmp_neural_optimize" => {
            use v2rmp::core::neural_routing::{solve_neural, NeuralRouteRequest};

            let locations_raw = arguments
                .get("locations")
                .and_then(Value::as_array)
                .context("Missing locations")?;
            let mut locations = Vec::with_capacity(locations_raw.len());
            for loc in locations_raw {
                let coords = loc.as_array().context("Invalid coordinate format")?;
                let lat = coords[0].as_f64().context("Invalid latitude")?;
                let lon = coords[1].as_f64().context("Invalid longitude")?;
                // NeuralRouteRequest expects [lat, lon, elevation]
                locations.push([lat, lon, 0.0]);
            }

            let demands = arguments
                .get("demands")
                .and_then(Value::as_array)
                .context("Missing demands")?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();

            let req = NeuralRouteRequest {
                model_path: arguments
                    .get("model_path")
                    .and_then(Value::as_str)
                    .context("Missing model_path")?
                    .to_string(),
                locations,
                demands,
                capacity: arguments
                    .get("capacity")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0),
            };

            let res = solve_neural(&req)?;
            let val = json!(res);
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(&res)?
                }],
                "structured": val,
                "isError": false
            }))
        }
        "v2rmp_generate_osmand_link" => {
            let gpx_url = arguments
                .get("gpx_url")
                .and_then(Value::as_str)
                .context("Missing gpx_url")?;
            let link = v2rmp::core::vrp::utils::generate_osmand_import_url(gpx_url);
            let val = json!({ "osmand_link": link });
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": link
                }],
                "structured": val,
                "isError": false
            }))
        }

        _ => anyhow::bail!("Tool not found"),
    }
}

fn respond(id: Option<&Value>, result: Value) -> Option<Value> {
    id.map(|id| json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

fn respond_error(id: Option<&Value>, code: i64, message: String) -> Option<Value> {
    id.map(
        |id| json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }),
    )
}
