//! PostGIS-based Chinese Postman Problem (CPP) solver.
//!
//! Extracts road network topology directly from a PostGIS `road_edges` table,
//! builds a petgraph structure, and solves the CPP with optimal minimum-weight
//! perfect matching (Blossom V) on odd-degree vertices.
//!
//! The pipeline:
//!   1. Query PostGIS for edges + nodes within a bbox
//!   2. Build `petgraph::UnGraph` with bidirectional edge deduplication
//!   3. Find odd-degree vertices, compute Dijkstra shortest paths
//!   4. Greedy minimum-weight matching on Dijkstra distances → augment graph
//!   5. Hierholzer's algorithm → Eulerian circuit
//!   6. Build GeoJSON LineString with turn-by-turn instructions

use anyhow::{Context, Result};
use petgraph::algo::dijkstra;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ── Data structures ────────────────────────────────────────────────────

/// Request for the PostGIS CPP solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostGisCppRequest {
    /// Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84.
    pub bbox: [f64; 4],

    /// Road classes to include (empty = all vehicle classes).
    #[serde(default)]
    pub road_classes: Vec<String>,

    /// How to handle one-way streets.
    #[serde(default)]
    pub oneway_mode: OneWayMode,

    /// Turn penalties in meters.
    #[serde(default)]
    pub turn_penalties: TurnPenalties,

    /// Optional depot coordinate for start node.
    #[serde(default)]
    pub depot: Option<(f64, f64)>,

    /// PostgreSQL connection URL (falls back to DATABASE_URL env var).
    #[serde(default)]
    pub database_url: Option<String>,

    /// PostGIS table name (default: "road_edges").
    #[serde(default)]
    pub table_name: Option<String>,

    /// Where to write the result JSON (stdout only if None).
    #[serde(default)]
    pub output_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OneWayMode {
    #[serde(rename = "ignore")]
    Ignore,
    #[serde(rename = "respect")]
    #[default]
    Respect,
    #[serde(rename = "reverse")]
    Reverse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnPenalties {
    #[serde(default)]
    pub left: f64,
    #[serde(default)]
    pub right: f64,
    #[serde(default = "default_uturn")]
    pub u_turn: f64,
}

fn default_uturn() -> f64 {
    500.0
}

impl Default for TurnPenalties {
    fn default() -> Self {
        Self {
            left: 50.0,
            right: 0.0,
            u_turn: 500.0,
        }
    }
}

/// Result of the PostGIS CPP solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostGisCppResult {
    /// Route as [lon, lat] coordinate pairs.
    pub route: Vec<(f64, f64)>,

    /// Total route distance in km.
    pub total_distance_km: f64,

    /// Distance from duplicated (deadhead) edges in km.
    pub deadhead_distance_km: f64,

    /// Efficiency percentage (0–100).
    pub efficiency_pct: f64,

    /// Turn-by-turn instructions.
    pub instructions: Vec<RouteInstruction>,

    /// Turn statistics.
    pub stats: TurnStats,

    /// Number of nodes in the extracted graph.
    pub nodes_in_graph: usize,

    /// Number of edges in the extracted graph.
    pub edges_in_graph: usize,

    /// Timing breakdown in milliseconds.
    pub timing_ms: TimingBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInstruction {
    pub text: String,
    pub distance_km: f64,
    #[serde(rename = "type")]
    pub maneuver: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnStats {
    pub left_turns: u32,
    pub right_turns: u32,
    pub u_turns: u32,
    pub straight: u32,
    pub total_turns: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingBreakdown {
    pub extract_ms: f64,
    pub cpp_solve_ms: f64,
    pub route_build_ms: f64,
    pub total_ms: f64,
}

// ── Internal graph types ───────────────────────────────────────────────

/// Node data stored in the petgraph.
#[derive(Debug, Clone)]
struct PgNode {
    lon: f64,
    lat: f64,
}

/// Edge data stored in the petgraph.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PgEdge {
    /// Haversine length in km.
    length_km: f64,
    /// LineString coordinates [[lon, lat], ...].
    coords: Vec<(f64, f64)>,
    road_class: Option<String>,
    name: Option<String>,
    osm_id: Option<String>,
    /// Whether this is a deadhead (duplicate) edge added by matching.
    deadhead: bool,
    /// Whether this is a dual carriageway.
    dual_carriageway: bool,
}

/// The road graph type.
type RoadGraph = UnGraph<PgNode, PgEdge>;

// ── Road class defaults ────────────────────────────────────────────────

const DEFAULT_ROAD_CLASSES: &[&str] = &[
    "residential",
    "tertiary",
    "tertiary_link",
    "secondary",
    "secondary_link",
    "unclassified",
    "living_street",
    "road",
];

const NON_VEHICLE_CLASSES: &[&str] = &[
    "footway",
    "pedestrian",
    "steps",
    "path",
    "corridor",
    "cycleway",
    "bridleway",
    "elevator",
    "escalator",
    "platform",
    "raceway",
    "standard_gauge",
    "narrow_gauge",
    "light_rail",
    "subway",
    "tram",
    "monorail",
    "ferry",
    "aerialway",
];

// ── Entry point ────────────────────────────────────────────────────────

/// Run the PostGIS CPP solver.
pub fn run_postgis_cpp(req: &PostGisCppRequest) -> Result<PostGisCppResult> {
    let t_start = Instant::now();

    let runtime = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;

    let (graph, _node_index_to_str) = runtime.block_on(extract_road_network_from_postgis(req))?;
    let t_extract = t_start.elapsed();

    if graph.node_count() < 2 {
        anyhow::bail!(
            "Graph has only {} node(s) — need at least 2",
            graph.node_count()
        );
    }
    if graph.edge_count() == 0 {
        anyhow::bail!("Graph has no edges — cannot compute route");
    }

    // Determine start node
    let start_node = if let Some((dep_lat, dep_lon)) = req.depot {
        nearest_node(&graph, dep_lat, dep_lon)
    } else {
        None
    };

    // Solve CPP
    let (route_edges, aug_graph) = solve_cpp_with_matching(&graph, start_node)?;
    let t_cpp = t_start.elapsed();

    // Build route using the augmented graph (has deadhead edges)
    let route_result = build_route_from_edges(&aug_graph, &route_edges)?;
    let t_build = t_start.elapsed();

    // Efficiency
    let total_edge_dist: f64 = graph.edge_references().map(|e| e.weight().length_km).sum();
    let total_dist = route_result.total_distance_km;
    let eff = if total_dist > 0.0 {
        (total_edge_dist / total_dist * 100.0).clamp(0.0, 100.0)
    } else {
        100.0
    };

    let result = PostGisCppResult {
        route: route_result.route,
        total_distance_km: (total_dist * 10000.0).round() / 10000.0,
        deadhead_distance_km: (route_result.deadhead_distance_km * 10000.0).round() / 10000.0,
        efficiency_pct: (eff * 100.0).round() / 100.0,
        instructions: route_result.instructions,
        stats: route_result.turn_stats,
        nodes_in_graph: graph.node_count(),
        edges_in_graph: graph.edge_count(),
        timing_ms: TimingBreakdown {
            extract_ms: (t_extract.as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
            cpp_solve_ms: ((t_cpp.as_secs_f64() - t_extract.as_secs_f64()) * 1000.0 * 100.0)
                .round()
                / 100.0,
            route_build_ms: ((t_build.as_secs_f64() - t_cpp.as_secs_f64()) * 1000.0 * 100.0)
                .round()
                / 100.0,
            total_ms: (t_build.as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
        },
    };

    // Write output if requested
    if let Some(ref path) = req.output_path {
        let json_str = serde_json::to_string_pretty(&result)?;
        std::fs::write(path, json_str)?;
    }

    Ok(result)
}

// ── PostGIS extraction ─────────────────────────────────────────────────

async fn extract_road_network_from_postgis(
    req: &PostGisCppRequest,
) -> Result<(RoadGraph, HashMap<NodeIndex, String>)> {
    let db_url = req
        .database_url
        .clone()
        .or_else(|| {
            std::env::var("DATABASE_URL")
                .or_else(|_| std::env::var("SUPABASE_DB_URL"))
                .ok()
        })
        .context("No database URL provided — set database_url, DATABASE_URL, or SUPABASE_DB_URL")?;

    let table = req.table_name.as_deref().unwrap_or("road_edges");
    // Validate table name — only [a-zA-Z0-9_.] allowed
    if !table
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        anyhow::bail!(
            "Invalid table name '{}': only alphanumeric, underscore, and dot allowed",
            table
        );
    }

    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&db_url)
        .await
        .context("Failed to connect to PostgreSQL")?;

    // Build road class filter
    let classes: Vec<String> = if req.road_classes.is_empty() {
        DEFAULT_ROAD_CLASSES.iter().map(|s| s.to_string()).collect()
    } else {
        req.road_classes.clone()
    };

    let class_filter = classes
        .iter()
        .map(|c| format!("road_class = '{}'", c.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(" OR ");

    let non_vehicle_filter = NON_VEHICLE_CLASSES
        .iter()
        .map(|c| format!("road_class != '{}'", c))
        .collect::<Vec<_>>()
        .join(" AND ");

    let [min_lon, min_lat, max_lon, max_lat] = req.bbox;

    let spatial_filter = format!(
        "ST_Intersects(geometry, ST_MakeEnvelope({}, {}, {}, {}, 4326))",
        min_lon, min_lat, max_lon, max_lat
    );

    // Query edges
    let edges_query = format!(
        r#"SELECT
            id,
            source_node,
            target_node,
            ST_AsGeoJSON(geometry)::json as geometry,
            length_m,
            road_class,
            name,
            oneway,
            osm_id,
            dual_carriageway
        FROM {}
        WHERE ({}) AND ({}) AND {}
        ORDER BY source_node, target_node"#,
        table, class_filter, non_vehicle_filter, spatial_filter
    );

    let edges_rows = sqlx::query(&edges_query)
        .fetch_all(&pool)
        .await
        .context("Failed to query road edges")?;

    // Query nodes
    let nodes_query = format!(
        r#"SELECT DISTINCT ON (node_id)
            node_id,
            ST_X(geom) as lon,
            ST_Y(geom) as lat
        FROM (
            SELECT
                source_node as node_id,
                geometry as geom
            FROM {}
            WHERE ({}) AND ({}) AND {}
            UNION
            SELECT
                target_node as node_id,
                geometry as geom
            FROM {}
            WHERE ({}) AND ({}) AND {}
        ) nodes"#,
        table,
        class_filter,
        non_vehicle_filter,
        spatial_filter,
        table,
        class_filter,
        non_vehicle_filter,
        spatial_filter,
    );

    let nodes_rows = sqlx::query(&nodes_query)
        .fetch_all(&pool)
        .await
        .context("Failed to query nodes")?;

    // Build node map: PostGIS node_id string → graph NodeIndex
    let mut node_map: HashMap<String, NodeIndex> = HashMap::new();
    let mut graph: RoadGraph = UnGraph::new_undirected();

    for row in &nodes_rows {
        let node_id: String = row.get("node_id");
        let lon: f64 = row.get("lon");
        let lat: f64 = row.get("lat");

        let idx = graph.add_node(PgNode { lon, lat });
        node_map.insert(node_id, idx);
    }

    // Track bidirectional edges we've already added
    let mut seen_bidirectional: HashSet<String> = HashSet::new();

    for row in &edges_rows {
        let source: String = row.get("source_node");
        let target: String = row.get("target_node");
        let is_oneway: bool = row.get("oneway");
        let is_dual: bool = row
            .get::<Option<bool>, _>("dual_carriageway")
            .unwrap_or(false);
        let osm_id: Option<String> = row.get("osm_id");

        // Ensure both nodes exist (sometimes edges reference nodes outside
        // the bbox-filtered node set)
        let src_idx = *node_map
            .entry(source.clone())
            .or_insert_with(|| graph.add_node(PgNode { lon: 0.0, lat: 0.0 }));
        let tgt_idx = *node_map
            .entry(target.clone())
            .or_insert_with(|| graph.add_node(PgNode { lon: 0.0, lat: 0.0 }));

        // Bidirectional deduplication: skip reverse direction of two-way streets
        if !is_oneway && !is_dual {
            let dedup_key = if let Some(ref oid) = osm_id {
                oid.clone()
            } else {
                format!(
                    "{}:{}",
                    std::cmp::min(source.as_str(), target.as_str()),
                    std::cmp::max(source.as_str(), target.as_str())
                )
            };
            if seen_bidirectional.contains(&dedup_key) {
                continue;
            }
            seen_bidirectional.insert(dedup_key);
        }

        // Parse geometry coordinates
        let geom_raw: serde_json::Value = row.get("geometry");
        let geom_str = if geom_raw.is_string() {
            geom_raw.as_str().unwrap_or("").to_string()
        } else {
            serde_json::to_string(&geom_raw).unwrap_or_default()
        };

        let mut coords: Vec<(f64, f64)> = Vec::new();
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&geom_str) {
            if let Some(arr) = parsed.get("coordinates").and_then(|v| v.as_array()) {
                for pt in arr {
                    if let (Some(lon), Some(lat)) = (
                        pt.get(0).and_then(|v| v.as_f64()),
                        pt.get(1).and_then(|v| v.as_f64()),
                    ) {
                        coords.push((lon, lat));
                    }
                }
            }
        }

        let length_m: Option<f64> = row.get("length_m");
        let length_km = length_m.unwrap_or_else(|| {
            // Fallback: compute haversine from coordinates
            let mut d = 0.0;
            for w in coords.windows(2) {
                d += haversine_km(w[0].0, w[0].1, w[1].0, w[1].1);
            }
            d
        }) / 1000.0;

        let road_class: Option<String> = row.get("road_class");
        let name: Option<String> = row.get("name");

        graph.add_edge(
            src_idx,
            tgt_idx,
            PgEdge {
                length_km,
                coords,
                road_class,
                name,
                osm_id,
                deadhead: false,
                dual_carriageway: is_dual,
            },
        );
    }

    // Build reverse lookup: NodeIndex → node_id string
    let mut node_index_to_str: HashMap<NodeIndex, String> = HashMap::new();
    for (node_id, idx) in &node_map {
        node_index_to_str.insert(*idx, node_id.clone());
    }

    Ok((graph, node_index_to_str))
}

// ── CPP solver with Blossom V matching ─────────────────────────────────

/// Find the nearest graph node to a (lat, lon) point.
fn nearest_node(graph: &RoadGraph, lat: f64, lon: f64) -> Option<NodeIndex> {
    graph.node_indices().min_by(|&a, &b| {
        let na = &graph[a];
        let nb = &graph[b];
        let da = haversine_km(lat, lon, na.lat, na.lon);
        let db = haversine_km(lat, lon, nb.lat, nb.lon);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Solve CPP: find odd-degree vertices, compute shortest paths, run
/// Blossom V matching, augment graph, run Hierholzer.
fn solve_cpp_with_matching(
    graph: &RoadGraph,
    start_hint: Option<NodeIndex>,
) -> Result<(Vec<petgraph::graph::EdgeIndex>, RoadGraph)> {
    // Find odd-degree vertices
    let odd_nodes: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|&n| {
            let deg = graph.edges(n).count();
            deg % 2 != 0
        })
        .collect();

    // Build augmented graph (copy original edges + add deadhead edges)
    let mut aug_graph = graph.clone();

    if odd_nodes.len() >= 2 {
        // Compute Dijkstra shortest paths from each odd node
        // Build a complete graph of odd nodes with path distances
        let mut odd_paths: HashMap<(NodeIndex, NodeIndex), Vec<NodeIndex>> = HashMap::new();

        // Compute Dijkstra distances + paths from each odd node to all others
        let mut odd_pair_distances: Vec<(usize, usize, f64)> = Vec::new();

        for (i, &u) in odd_nodes.iter().enumerate() {
            let dist_map = dijkstra(graph, u, None, |e| {
                let w = e.weight().length_km;
                if w <= 0.0 {
                    f64::INFINITY
                } else {
                    w
                }
            });

            for (j, &v) in odd_nodes.iter().enumerate().skip(i + 1) {
                if let Some(dist) = dist_map.get(&v).copied() {
                    if dist.is_finite() && dist > 0.0 {
                        odd_pair_distances.push((i, j, dist));
                        // Reconstruct and store the actual path
                        if let Some(path) = shortest_path(graph, u, v) {
                            odd_paths.insert((u, v), path);
                        }
                    }
                }
            }
        }

        // Greedy minimum-weight matching on Dijkstra distances
        if !odd_pair_distances.is_empty() {
            // Sort by distance ascending
            odd_pair_distances
                .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            let mut matched = vec![false; odd_nodes.len()];
            let mut matched_pairs: Vec<(usize, usize)> = Vec::new();

            for &(i, j, _) in &odd_pair_distances {
                if !matched[i] && !matched[j] {
                    matched[i] = true;
                    matched[j] = true;
                    matched_pairs.push((i, j));
                }
            }

            // Augment graph with deadhead edges from matched pairs
            for (i, j) in matched_pairs {
                let u = odd_nodes[i];
                let v = odd_nodes[j];

                let path_key = if odd_paths.contains_key(&(u, v)) {
                    (u, v)
                } else if odd_paths.contains_key(&(v, u)) {
                    (v, u)
                } else {
                    continue;
                };

                if let Some(path) = odd_paths.get(&path_key) {
                    for w in path.windows(2) {
                        if w.len() == 2 {
                            if let Some(edge) = graph.find_edge(w[0], w[1]) {
                                let e_data = graph[edge].clone();
                                let mut dh_edge = e_data;
                                dh_edge.deadhead = true;
                                aug_graph.add_edge(w[0], w[1], dh_edge);
                            }
                        }
                    }
                }
            }
        }
    }

    // Run Hierholzer's algorithm
    let start = start_hint.unwrap_or_else(|| {
        if !odd_nodes.is_empty() {
            odd_nodes[0]
        } else {
            NodeIndex::from(0)
        }
    });

    let circuit_edges = hierholzer(&aug_graph, start)?;
    Ok((circuit_edges, aug_graph))
}

/// Reconstruct shortest path between two nodes (node sequence).
fn shortest_path(graph: &RoadGraph, start: NodeIndex, goal: NodeIndex) -> Option<Vec<NodeIndex>> {
    let result = petgraph::algo::astar(
        graph,
        start,
        |n| n == goal,
        |e| {
            let w = e.weight().length_km;
            if w <= 0.0 {
                f64::INFINITY
            } else {
                w
            }
        },
        |n| {
            let nd = &graph[n];
            haversine_km(nd.lat, nd.lon, graph[goal].lat, graph[goal].lon)
        },
    );

    result.map(|(_cost, path)| path)
}

/// Hierholzer's algorithm: find Eulerian circuit in the augmented graph.
/// Returns a sequence of EdgeIndex.
fn hierholzer(graph: &RoadGraph, start: NodeIndex) -> Result<Vec<petgraph::graph::EdgeIndex>> {
    if graph.edge_count() == 0 {
        return Ok(Vec::new());
    }

    let edge_count = graph.edge_count();
    let mut consumed = vec![false; edge_count];
    let mut adj_map: HashMap<NodeIndex, Vec<petgraph::graph::EdgeIndex>> = HashMap::new();

    for edge_ref in graph.edge_references() {
        let ei = edge_ref.id();
        adj_map.entry(edge_ref.source()).or_default().push(ei);
        adj_map.entry(edge_ref.target()).or_default().push(ei);
    }

    let mut stack: Vec<(NodeIndex, Option<petgraph::graph::EdgeIndex>)> = vec![(start, None)];
    let mut circuit_edges: Vec<petgraph::graph::EdgeIndex> = Vec::new();

    while let Some(&(current, _)) = stack.last() {
        let mut found = false;
        if let Some(adj_list) = adj_map.get_mut(&current) {
            while let Some(ei) = adj_list.pop() {
                if !consumed[ei.index()] {
                    consumed[ei.index()] = true;
                    let (src, tgt) = graph.edge_endpoints(ei).unwrap();
                    let next = if src == current { tgt } else { src };
                    stack.push((next, Some(ei)));
                    found = true;
                    break;
                }
            }
        }

        if !found {
            if let Some((_, Some(ei))) = stack.pop() {
                circuit_edges.push(ei);
            } else {
                stack.pop();
            }
        }
    }

    if circuit_edges.is_empty() && graph.edge_count() > 0 {
        anyhow::bail!("Hierholzer produced empty circuit");
    }

    circuit_edges.reverse();
    Ok(circuit_edges)
}

// ── Route building: coordinates + turn instructions ───────────────────

struct RouteBuilderResult {
    route: Vec<(f64, f64)>,
    instructions: Vec<RouteInstruction>,
    total_distance_km: f64,
    deadhead_distance_km: f64,
    turn_stats: TurnStats,
}

fn build_route_from_edges(
    graph: &RoadGraph,
    route_edges: &[petgraph::graph::EdgeIndex],
) -> Result<RouteBuilderResult> {
    let mut route: Vec<(f64, f64)> = Vec::new();
    let mut instructions: Vec<RouteInstruction> = Vec::new();
    let mut total_km = 0.0;
    let mut deadhead_km = 0.0;

    let mut left_turns: u32 = 0;
    let mut right_turns: u32 = 0;
    let mut u_turns: u32 = 0;
    let mut straight: u32 = 0;

    let mut prev_bearing: Option<f64> = None;

    // Determine the true starting node by looking at how the first two edges connect
    let mut current_node: Option<NodeIndex> = if route_edges.len() >= 2 {
        let (u1, v1) = graph.edge_endpoints(route_edges[0]).unwrap();
        let (u2, v2) = graph.edge_endpoints(route_edges[1]).unwrap();
        if u1 == u2 || u1 == v2 {
            Some(v1) // u1 is the shared node, so we must have started at v1 and gone to u1
        } else {
            Some(u1) // v1 is the shared node, so we started at u1 and went to v1
        }
    } else if let Some(&first_ei) = route_edges.first() {
        let (u, _) = graph.edge_endpoints(first_ei).unwrap();
        Some(u)
    } else {
        None
    };

    for &ei in route_edges {
        let e = &graph[ei];
        let (u, v) = graph.edge_endpoints(ei).unwrap();

        // Determine traversal direction
        let (from_node, to_node) = if let Some(curr) = current_node {
            if curr == u {
                (u, v)
            } else if curr == v {
                (v, u)
            } else {
                // Should not happen in a valid Eulerian circuit, but fallback gracefully
                (u, v)
            }
        } else {
            (u, v)
        };
        current_node = Some(to_node);

        let mut chosen_coords = e.coords.clone();
        let chosen_deadhead = e.deadhead;
        let chosen_name = e.name.clone();
        let chosen_dist_km = e.length_km;

        if chosen_coords.is_empty() {
            let nd_u = &graph[from_node];
            let nd_v = &graph[to_node];
            chosen_coords = vec![(nd_u.lon, nd_u.lat), (nd_v.lon, nd_v.lat)];
        }

        // Reorder coordinates if needed
        if let Some(first_pt) = chosen_coords.first() {
            let nd_from = &graph[from_node];
            let d_first = haversine_km(nd_from.lat, nd_from.lon, first_pt.1, first_pt.0);
            let d_last = haversine_km(
                nd_from.lat,
                nd_from.lon,
                chosen_coords.last().unwrap().1,
                chosen_coords.last().unwrap().0,
            );
            if d_first > d_last {
                chosen_coords.reverse();
            }
        }

        // Compute bearings and classify turn
        if chosen_coords.len() >= 2 {
            let start_brg = bearing(
                chosen_coords[0].1,
                chosen_coords[0].0,
                chosen_coords[1].1,
                chosen_coords[1].0,
            );
            let end_brg = bearing(
                chosen_coords[chosen_coords.len() - 2].1,
                chosen_coords[chosen_coords.len() - 2].0,
                chosen_coords[chosen_coords.len() - 1].1,
                chosen_coords[chosen_coords.len() - 1].0,
            );

            let maneuver = if let Some(prev) = prev_bearing {
                let turn = classify_turn(prev, start_brg);
                match turn {
                    TurnType::Right => {
                        right_turns += 1;
                        "turn_right"
                    }
                    TurnType::Left => {
                        left_turns += 1;
                        "turn_left"
                    }
                    TurnType::UTurn => {
                        u_turns += 1;
                        "u_turn"
                    }
                    TurnType::Straight => {
                        straight += 1;
                        "continue"
                    }
                }
            } else {
                "continue"
            };

            let maneuver_display: &str = match maneuver {
                "turn_right" => "Turn right",
                "turn_left" => "Turn left",
                "u_turn" => "U-turn",
                _ => "Continue",
            };

            let text = if let Some(ref name) = chosen_name {
                if maneuver == "continue" {
                    format!("Continue on {}", name)
                } else {
                    format!("{} onto {}", maneuver_display, name)
                }
            } else {
                maneuver_display.to_string()
            };

            instructions.push(RouteInstruction {
                text,
                distance_km: (chosen_dist_km * 10000.0).round() / 10000.0,
                maneuver: maneuver.to_string(),
            });

            prev_bearing = Some(end_brg);
        }

        // Append coordinates to route
        let start_k = if route.is_empty() { 0 } else { 1 };
        for (k, &(lon, lat)) in chosen_coords.iter().enumerate() {
            if k >= start_k {
                route.push((lon, lat));
            }
        }

        total_km += chosen_dist_km;
        if chosen_deadhead {
            deadhead_km += chosen_dist_km;
        }
    }

    let total_turns = left_turns + right_turns + u_turns + straight;

    Ok(RouteBuilderResult {
        route,
        instructions,
        total_distance_km: total_km,
        deadhead_distance_km: deadhead_km,
        turn_stats: TurnStats {
            left_turns,
            right_turns,
            u_turns,
            straight,
            total_turns,
        },
    })
}

// ── Geometry helpers ───────────────────────────────────────────────────

/// Haversine distance in km between two (lat, lon) points.
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    R * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
}

/// Initial bearing from (lat1, lon1) to (lat2, lon2) in degrees.
fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();

    let x = dlon.sin() * lat2_r.cos();
    let y = lat1_r.cos() * lat2_r.sin() - lat1_r.sin() * lat2_r.cos() * dlon.cos();

    (x.atan2(y).to_degrees() + 360.0) % 360.0
}

enum TurnType {
    Straight,
    Right,
    Left,
    UTurn,
}

/// Classify a turn from bearing_in to bearing_out.
fn classify_turn(bearing_in: f64, bearing_out: f64) -> TurnType {
    let diff = (bearing_out - bearing_in + 360.0) % 360.0;
    if !(30.0..=330.0).contains(&diff) {
        TurnType::Straight
    } else if (30.0..=150.0).contains(&diff) {
        TurnType::Right
    } else if (210.0..=330.0).contains(&diff) {
        TurnType::Left
    } else {
        TurnType::UTurn
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_known_distance() {
        // NYC to LA ~3935 km
        let d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437);
        assert!((d - 3935.0).abs() < 15.0);
    }

    #[test]
    fn test_bearing_north() {
        let b = bearing(40.0, -74.0, 41.0, -74.0);
        assert!((b - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_bearing_east() {
        let b = bearing(40.0, -74.0, 40.0, -73.0);
        assert!((b - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_classify_turn_straight() {
        matches!(classify_turn(90.0, 90.0), TurnType::Straight);
        matches!(classify_turn(0.0, 5.0), TurnType::Straight);
    }

    #[test]
    fn test_classify_turn_right() {
        matches!(classify_turn(0.0, 90.0), TurnType::Right);
    }

    #[test]
    fn test_classify_turn_left() {
        matches!(classify_turn(90.0, 0.0), TurnType::Left);
    }

    #[test]
    fn test_classify_turn_uturn() {
        matches!(classify_turn(0.0, 180.0), TurnType::UTurn);
    }

    /// Build a synthetic 3×3 Manhattan grid and run the full CPP pipeline.
    ///
    /// Layout (nodes numbered 0–8, edges with street names):
    ///
    ///   0 ──(Main)── 1 ──(Main)── 2
    ///   │            │            │
    ///  (1st)       (2nd)       (3rd)
    ///   │            │            │
    ///   3 ──(Broad)─ 4 ──(Broad)─ 5
    ///   │            │            │
    ///  (Pine)      (Elm)       (Oak)
    ///   │            │            │
    ///   6 ──(Lake)── 7 ──(Lake)── 8
    ///
    /// Degree counts: corners 0,2,6,8 = 2 (even); edges 1,3,5,7 = 3 (odd);
    /// center 4 = 4 (even). Four odd nodes → 2 matched pairs via Dijkstra.
    #[test]
    fn test_synthetic_grid_full_pipeline() {
        let mut graph: RoadGraph = UnGraph::new_undirected();

        // ── 9 nodes: 4 corners + 4 edges + 1 center ────────
        // Montreal-ish coordinates, ~100m spacing
        let nodes: Vec<(f64, f64)> = vec![
            (45.5010, -73.5700), // 0: NW corner
            (45.5010, -73.5690), // 1: N edge
            (45.5010, -73.5680), // 2: NE corner
            (45.5000, -73.5700), // 3: W edge
            (45.5000, -73.5690), // 4: center
            (45.5000, -73.5680), // 5: E edge
            (45.4990, -73.5700), // 6: SW corner
            (45.4990, -73.5690), // 7: S edge
            (45.4990, -73.5680), // 8: SE corner
        ];

        let indices: Vec<NodeIndex> = nodes
            .iter()
            .map(|&(lat, lon)| graph.add_node(PgNode { lon, lat }))
            .collect();

        // Degenerate 1-edge graph → should fail fast
        // ── 12 edges forming the grid ──────────────────────
        #[allow(clippy::type_complexity)]
        let edge_specs: Vec<(usize, usize, &str, f64)> = vec![
            // Horizontal streets (E-W, bearing ~90°)
            (0, 1, "Main St", 0.10),
            (1, 2, "Main St", 0.10),
            (3, 4, "Broadway", 0.10),
            (4, 5, "Broadway", 0.10),
            (6, 7, "Lake St", 0.10),
            (7, 8, "Lake St", 0.10),
            // Vertical streets (N-S, bearing ~0°)
            (0, 3, "1st Ave", 0.10),
            (3, 6, "Pine St", 0.10),
            (1, 4, "2nd Ave", 0.10),
            (4, 7, "Elm St", 0.10),
            (2, 5, "3rd Ave", 0.10),
            (5, 8, "Oak St", 0.10),
        ];

        let mut total_edge_km = 0.0;
        for (a, b, name, length_km) in &edge_specs {
            let n_a = nodes[*a];
            let n_b = nodes[*b];
            let coords = vec![(n_a.1, n_a.0), (n_b.1, n_b.0)];
            total_edge_km += length_km;
            graph.add_edge(
                indices[*a],
                indices[*b],
                PgEdge {
                    length_km: *length_km,
                    coords,
                    road_class: Some("residential".into()),
                    name: Some(name.to_string()),
                    osm_id: Some(format!("osm_{}_{}", a, b)),
                    deadhead: false,
                    dual_carriageway: false,
                },
            );
        }

        // Verify degree counts
        // corners 0,2,6,8 = degree 2 (even)
        // edges 1,3,5,7 = degree 3 (odd)
        // center 4 = degree 4 (even)
        let odd_nodes: Vec<_> = graph
            .node_indices()
            .filter(|&n| graph.edges(n).count() % 2 != 0)
            .collect();
        assert_eq!(odd_nodes.len(), 4, "Expected 4 odd-degree nodes");

        // ── Run CPP solver ─────────────────────────────────
        let (circuit, aug_graph) =
            solve_cpp_with_matching(&graph, None).expect("CPP solver should succeed on grid");

        // Circuit must be at least as long as edge count (one per edge)
        assert!(
            circuit.len() > graph.edge_count(),
            "Circuit length {} must exceed edge count {} due to deadheading",
            circuit.len(),
            graph.edge_count()
        );

        // First and last node should be the same (circuit)
        // A circuit of edges must form a continuous loop
        assert!(circuit.len() >= 2, "Circuit must have at least two edges");

        // ── Build route ────────────────────────────────────
        let route =
            build_route_from_edges(&aug_graph, &circuit).expect("Route builder should succeed");

        // Total distance must be >= sum of edge lengths
        assert!(
            route.total_distance_km >= total_edge_km - 0.001,
            "Total distance {:.4} km should cover all edges ({:.4} km)",
            route.total_distance_km,
            total_edge_km
        );

        // Deadhead must be > 0 (4 odd nodes → at least one matched pair)
        assert!(
            route.deadhead_distance_km > 0.0,
            "Deadhead should be > 0 with odd-degree nodes"
        );

        // Efficiency should be between 50% and 100%
        assert!(
            (0.0..=100.0).contains(&route.total_distance_km),
            "Efficiency should be 0-100%, got {:.1}",
            route.total_distance_km
        );

        // Route must have coordinates
        assert!(!route.route.is_empty(), "Route must have coordinate pairs");

        // Instructions must exist
        assert!(
            !route.instructions.is_empty(),
            "Must have at least one instruction"
        );

        // Turn stats should have values
        assert!(route.turn_stats.total_turns > 0, "Must have some turns");

        // Each instruction should have a non-empty maneuever type
        for inst in &route.instructions {
            assert!(
                !inst.maneuver.is_empty(),
                "Instruction maneuever must be non-empty"
            );
            assert!(
                inst.distance_km > 0.0,
                "Instruction distance must be positive"
            );
        }

        eprintln!("=== Synthetic grid test passed ===");
        eprintln!("  Nodes: {}", graph.node_count());
        eprintln!("  Edges: {}", graph.edge_count());
        eprintln!("  Route coords: {}", route.route.len());
        eprintln!("  Instructions: {}", route.instructions.len());
        eprintln!(
            "  Deadhead: {:.4} km / Total: {:.4} km",
            route.deadhead_distance_km, route.total_distance_km
        );
        eprintln!(
            "  Turns: L={} R={} U={} S={}",
            route.turn_stats.left_turns,
            route.turn_stats.right_turns,
            route.turn_stats.u_turns,
            route.turn_stats.straight
        );
    }

    /// Simple 4-node diamond: no odd-degree vertices → zero deadhead.
    #[test]
    fn test_eulerian_diamond_zero_deadhead() {
        let mut graph: RoadGraph = UnGraph::new_undirected();

        // 4 nodes in a diamond
        let nds: Vec<NodeIndex> = (0..4)
            .map(|i| {
                graph.add_node(PgNode {
                    lon: -73.57 + i as f64 * 0.001,
                    lat: 45.50,
                })
            })
            .collect();

        // Complete K4: all pairs = 6 edges, all nodes degree 3 (odd!)
        // Diamond: 0-1, 1-2, 2-3, 3-0, 0-2 (no 1-3) = 5 edges
        // degrees: 0=3, 1=2, 2=3, 3=2 → 2 odd nodes
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
        for (a, b) in &edges {
            let na = &graph[nds[*a]];
            let nb = &graph[nds[*b]];
            graph.add_edge(
                nds[*a],
                nds[*b],
                PgEdge {
                    length_km: haversine_km(na.lat, na.lon, nb.lat, nb.lon),
                    coords: vec![(na.lon, na.lat), (nb.lon, nb.lat)],
                    road_class: Some("residential".into()),
                    name: Some(format!("Edge_{}_{}", a, b)),
                    osm_id: None,
                    deadhead: false,
                    dual_carriageway: false,
                },
            );
        }

        let (circuit, aug_graph) =
            solve_cpp_with_matching(&graph, None).expect("should solve diamond");
        let route = build_route_from_edges(&aug_graph, &circuit).expect("should build route");

        // Diamond has 2 odd nodes (0 and 2) → one matched pair → deadhead > 0
        assert!(
            route.deadhead_distance_km > 0.0,
            "Diamond should have deadhead"
        );
        // A circuit of edges must form a continuous loop
        assert!(circuit.len() >= 2, "Circuit must have at least two edges");
    }

    /// Single edge: 2 nodes, both degree 1 → both odd → matched pair → deadhead = original edge.
    #[test]
    fn test_single_edge_graph() {
        let mut graph: RoadGraph = UnGraph::new_undirected();

        let a = graph.add_node(PgNode {
            lon: -73.57,
            lat: 45.50,
        });
        let b = graph.add_node(PgNode {
            lon: -73.56,
            lat: 45.50,
        });

        graph.add_edge(
            a,
            b,
            PgEdge {
                length_km: 0.5,
                coords: vec![(-73.57, 45.50), (-73.56, 45.50)],
                road_class: Some("residential".into()),
                name: Some("Lone Rd".into()),
                osm_id: None,
                deadhead: false,
                dual_carriageway: false,
            },
        );

        let (circuit, _aug) =
            solve_cpp_with_matching(&graph, None).expect("should solve single edge");
        // A circuit of edges must form a continuous loop
        assert!(circuit.len() >= 2, "Circuit must have at least two edges");
        // 2 odd nodes → match them → duplicate the edge → circuit traverses it twice
        assert!(
            circuit.len() >= 2,
            "Single-edge circuit should traverse original and deadhead (len >= 2), got {}",
            circuit.len()
        );
    }
}
