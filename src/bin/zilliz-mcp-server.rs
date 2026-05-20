//! Zilliz-MCP-Server — MCP (Model Context Protocol) server exposing
//! semantic codebase search over the v2rmp Zilliz Cloud vector database.
//!
//! Runs over stdio with JSON-RPC 2.0 framing, one line per message.
//!
//! Tools exposed:
//!   - `search_codebase` — embed a natural-language query via Ollama and
//!     return the top-k code chunks from the v2rmp vector DB.
//!
//! Connect from any MCP client (Claude Desktop, Continue, Cursor, etc.)
//! by adding to the client's config:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "zilliz-codebase": {
//!       "command": "cargo",
//!       "args": ["run", "--bin", "zilliz-mcp-server", "--release"]
//!     }
//!   }
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

// ── Config (same as zilliz_embed_pipeline.py / VECTOR_DB_CONFIG.md) ────────

const ZILLIZ_URI: &str =
    "https://in05-895d375123308f8.serverless.aws-eu-central-1.cloud.zilliz.com";
fn zilliz_token() -> Result<String> {
    std::env::var("ZILLIZ_TOKEN").context("ZILLIZ_TOKEN environment variable not set")
}
const OLLAMA_URL: &str = "http://localhost:11434/api/embed";
const EMBED_MODEL: &str = "mxbai-embed-large";
const COLLECTION: &str = "v2rmp_codebase";
const DIMENSION: usize = 1024;

// ── JSON-RPC / MCP types ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Request {
    jsonrpc: String,
    method: String,
    #[serde(default)]
    params: Value,
    #[serde(default)]
    #[allow(dead_code)]
    id: Value,
}

#[derive(Debug, Serialize)]
struct ToolDef {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

// ── Ollama embed helper ────────────────────────────────────────────────────

async fn ollama_embed(query: &str) -> Result<Vec<Vec<f32>>> {
    let client = reqwest::Client::new();
    let body = json!({
        "model": EMBED_MODEL,
        "input": [query]
    });

    let resp = client
        .post(OLLAMA_URL)
        .json(&body)
        .send()
        .await
        .context("Ollama request failed — is it running?")?;

    let data: Value = resp.json().await.context("Ollama response parse")?;
    let embeddings: Vec<Vec<f32>> =
        serde_json::from_value(data["embeddings"].clone()).context("bad embeddings shape")?;
    Ok(embeddings)
}

// ── Zilliz REST search ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ZillizSearchResp {
    data: Vec<ZillizHit>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ZillizHit {
    #[serde(default)]
    #[allow(dead_code)]
    id: Value,
    #[serde(default)]
    distance: Option<f64>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    start_line: Option<i64>,
    #[serde(default)]
    end_line: Option<i64>,
    #[serde(default)]
    code: Option<String>,
}

async fn zilliz_search(vector: &[f32], limit: usize) -> Result<Vec<ZillizHit>> {
    let client = reqwest::Client::new();

    let body = json!({
        "collectionName": COLLECTION,
        "data": [vector],
        "annsField": "vector",
        "limit": limit,
        "outputFields": ["file", "start_line", "end_line", "code"],
        "searchParams": {
            "metricType": "L2",
            "params": { "nprobe": 32 }
        }
    });

    let resp = client
        .post(format!("{ZILLIZ_URI}/v2/vectordb/entities/search"))
        .header("Authorization", format!("Bearer {}", zilliz_token()?))
        .json(&body)
        .send()
        .await
        .context("Zilliz search request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Zilliz returned {status}: {text}");
    }

    let mut result: ZillizSearchResp = resp.json().await.context("Zilliz search response parse")?;

    Ok(std::mem::take(&mut result.data))
}

// ── Tool: search_codebase ──────────────────────────────────────────────────

async fn handle_search_codebase(args: &Value) -> Result<Value> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if query.is_empty() {
        anyhow::bail!("Missing 'query' parameter");
    }

    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10)
        .min(50) as usize;

    // 1. Embed
    let embeddings = ollama_embed(&query).await?;
    let vector = embeddings
        .first()
        .context("Ollama returned empty embeddings")?;

    if vector.len() != DIMENSION {
        anyhow::bail!(
            "Ollama returned {}-dim embedding; expected {DIMENSION}",
            vector.len()
        );
    }

    // 2. Search
    let hits = zilliz_search(vector, limit).await?;

    // 3. Format
    let items: Vec<Value> = hits
        .into_iter()
        .filter_map(|h| {
            Some(json!({
                "file": h.file?,
                "start_line": h.start_line?,
                "end_line": h.end_line?,
                "code": h.code?,
                "score": h.distance?,
            }))
        })
        .collect();

    Ok(json!({
        "query": query,
        "results": items,
        "count": items.len(),
        "embedding_model": EMBED_MODEL,
        "metric": "L2 (lower = more similar)"
    }))
}

// ── Main loop ──────────────────────────────────────────────────────────────

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
        "error": {
            "code": code,
            "message": msg
        }
    });
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{out}");
    let _ = stdout.flush();
}

#[tokio::main]
async fn main() -> Result<()> {
    // Stderr for logging so it doesn't corrupt the stdio JSON-RPC stream
    eprintln!("zilliz-mcp-server starting (collection={COLLECTION}, model={EMBED_MODEL})");

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
            // ── initialize ────────────────────────────────────────────
            "initialize" => {
                send(
                    &req.id,
                    json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "zilliz-mcp-server",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }),
                );
            }

            // ── tools/list ────────────────────────────────────────────
            "tools/list" => {
                let tools = vec![ToolDef {
                    name: "search_codebase".into(),
                    description: concat!(
                        "Semantic search over the v2rmp (rmpca) Rust codebase. ",
                        "Embeds your natural-language query with mxbai-embed-large ",
                        "and retrieves the most relevant code chunks from the Zilliz ",
                        "vector database. Returns file path, line range, code snippet, ",
                        "and L2 similarity score for each match."
                    )
                    .into(),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural-language search query, e.g. 'How does the VRP solver optimize routes?'"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default: 10, max: 50)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }),
                }];
                send(&req.id, json!({ "tools": tools }));
            }

            // ── tools/call ────────────────────────────────────────────
            "tools/call" => {
                let name = req
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args = req.params.get("arguments").cloned().unwrap_or(Value::Null);

                match name {
                    "search_codebase" => match handle_search_codebase(&args).await {
                        Ok(result) => {
                            send(
                                &req.id,
                                json!({
                                    "content": [{
                                        "type": "text",
                                        "text": serde_json::to_string(&result).unwrap_or_default()
                                    }],
                                    "structured": result
                                }),
                            );
                        }
                        Err(e) => {
                            send(
                                &req.id,
                                json!({
                                    "content": [{
                                        "type": "text",
                                        "text": format!("Error: {e:#}")
                                    }],
                                    "isError": true
                                }),
                            );
                        }
                    },
                    other => {
                        send_err(&req.id, -32602, &format!("Unknown tool: {other}"));
                    }
                }
            }

            // ── notifications (no response) ───────────────────────────
            "notifications/initialized" | "initialized" => {
                // No response needed — MCP spec: client sends this after
                // receiving the initialize response.
            }

            // ── unrecognized ──────────────────────────────────────────
            other => {
                send_err(&req.id, -32601, &format!("Method not found: {other}"));
            }
        }
    }

    Ok(())
}
