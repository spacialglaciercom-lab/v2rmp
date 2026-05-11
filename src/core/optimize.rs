use crate::core::vrp::registry::solve_with;
use crate::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::time::Instant;

/// Filter nodes and edges to only those within a bounding box.
/// Returns remapped (nodes, edges) where edge indices are renumbered from 0.
/// If `bbox` is `None`, returns the originals unchanged.
#[allow(dead_code)]
pub fn filter_bbox(
    nodes: &[RmpNode],
    edges: &[RmpEdge],
    bbox: Option<(f64, f64, f64, f64)>, // (min_lat, max_lat, min_lon, max_lon)
) -> (Vec<RmpNode>, Vec<RmpEdge>) {
    let Some((min_lat, max_lat, min_lon, max_lon)) = bbox else {
        return (nodes.to_vec(), edges.to_vec());
    };

    // Mark which old-node indices are inside the bbox
    let inside: Vec<bool> = nodes
        .iter()
        .map(|n| n.lat >= min_lat && n.lat <= max_lat && n.lon >= min_lon && n.lon <= max_lon)
        .collect();

    // Build old->new index map
    let mut old_to_new = vec![u32::MAX; nodes.len()];
    let mut new_nodes = Vec::new();
    let mut next: u32 = 0;
    for (i, &is_inside) in inside.iter().enumerate() {
        if is_inside {
            old_to_new[i] = next;
            new_nodes.push(nodes[i]);
            next += 1;
        }
    }

    // Keep edges whose both endpoints are inside
    let new_edges: Vec<RmpEdge> = edges
        .iter()
        .filter(|e| {
            let from_inside = inside.get(e.from as usize).copied().unwrap_or(false);
            let to_inside = inside.get(e.to as usize).copied().unwrap_or(false);
            from_inside && to_inside
        })
        .map(|e| RmpEdge {
            from: old_to_new[e.from as usize],
            to: old_to_new[e.to as usize],
            weight_m: e.weight_m,
            oneway: e.oneway,
        })
        .collect();

    (new_nodes, new_edges)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TurnPenalties {
    pub left: f64,
    pub right: f64,
    pub u_turn: f64,
}

impl Default for TurnPenalties {
    fn default() -> Self {
        Self {
            left: 1.0,
            right: 0.0,
            u_turn: 5.0,
        }
    }
}

/// Which solver to use: CPP covers all edges (Chinese Postman), VRP optimises stop visits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum SolverMode {
    #[default]
    Cpp,
    Vrp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeRequest {
    pub cache_file: String,
    pub route_file: Option<String>,
    pub turn_penalties: TurnPenalties,
    pub depot: Option<(f64, f64)>,
    pub oneway_mode: OnewayMode,
    /// Solver mode: Cpp (default, edge coverage) or Vrp (stop visits).
    pub mode: SolverMode,
    /// VRP-only: number of vehicles.
    #[serde(default = "default_num_vehicles")]
    pub num_vehicles: usize,
    /// VRP-only: solver algorithm id (clarke_wright, sweep, two_opt, or_opt, default).
    #[serde(default = "default_solver_id")]
    pub solver_id: String,
}

fn default_num_vehicles() -> usize {
    1
}
fn default_solver_id() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum OnewayMode {
    Ignore,
    #[default]
    Respect,
    Reverse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeResult {
    pub total_distance_km: f64,
    pub total_segments: usize,
    pub deadhead_distance_km: f64,
    pub efficiency_pct: f64,
    pub turns: TurnSummary,
    pub elapsed_ms: u64,
    pub num_routes: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TurnSummary {
    pub left: u32,
    pub right: u32,
    pub u_turn: u32,
    pub straight: u32,
}

// ── Binary format structures ──────────────────────────────────────────

/// A node in the road network (lat/lon in WGS-84).
#[derive(Debug, Clone, Copy)]
pub struct RmpNode {
    pub lat: f64,
    pub lon: f64,
}

/// An edge in the road network.
#[derive(Debug, Clone, Copy)]
pub struct RmpEdge {
    pub from: u32,
    pub to: u32,
    pub weight_m: f64,
    pub oneway: u8,
}

// ── Turn classification ──────────────────────────────────

pub(crate) use super::haversine_m;

/// Classify a turn by bearing delta (degrees).
/// Returns "straight", "right", "left", or "u_turn".
pub fn classify_turn(bearing_delta: f64) -> &'static str {
    let d = bearing_delta.normalize(-180.0, 180.0);
    if d.abs() <= 45.0 {
        "straight"
    } else if d > 45.0 && d <= 135.0 {
        "right"
    } else if (-135.0..-45.0).contains(&d) {
        "left"
    } else {
        "u_turn"
    }
}

trait NormalizeAngle {
    fn normalize(self, lower: f64, upper: f64) -> f64;
}

impl NormalizeAngle for f64 {
    fn normalize(self, lower: f64, upper: f64) -> f64 {
        let width = upper - lower;
        let mut val = self;
        while val < lower {
            val += width;
        }
        while val >= upper {
            val -= width;
        }
        val
    }
}

// ── Binary parser ────────────────────────────────────────────────────

/// Magic bytes for the .rmp binary format.
const RMP_MAGIC: &[u8; 4] = b"RMP1";

/// Parse an `.rmp` binary file and return (nodes, edges).
pub fn read_rmp_file(data: &[u8]) -> anyhow::Result<(Vec<RmpNode>, Vec<RmpEdge>)> {
    // Validate magic
    if data.len() < 4 || &data[..4] != RMP_MAGIC {
        anyhow::bail!("Invalid .rmp file: missing RMP1 magic bytes");
    }

    if data.len() < 12 {
        anyhow::bail!("Invalid .rmp file: header too short");
    }

    let node_count = u32::from_le_bytes(data[4..8].try_into()?) as usize;
    let edge_count = u32::from_le_bytes(data[8..12].try_into()?) as usize;

    let nodes_end = 12 + node_count * 16;
    let edges_end = nodes_end + edge_count * 17;
    let expected_len = edges_end + 4;

    if data.len() < expected_len {
        anyhow::bail!(
            "Invalid .rmp file: expected {} bytes, got {}",
            expected_len,
            data.len()
        );
    }

    // Parse nodes
    let mut nodes = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let offset = 12 + i * 16;
        let lat = f64::from_le_bytes(data[offset..offset + 8].try_into()?);
        let lon = f64::from_le_bytes(data[offset + 8..offset + 16].try_into()?);
        nodes.push(RmpNode { lat, lon });
    }

    // Parse edges
    let mut edges = Vec::with_capacity(edge_count);
    for i in 0..edge_count {
        let offset = nodes_end + i * 17;
        let from = u32::from_le_bytes(data[offset..offset + 4].try_into()?);
        let to = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?);
        let weight_m = f64::from_le_bytes(data[offset + 8..offset + 16].try_into()?);
        let oneway = data[offset + 16];
        edges.push(RmpEdge {
            from,
            to,
            weight_m,
            oneway,
        });
    }

    // Verify CRC32
    let expected_crc = u32::from_le_bytes(data[edges_end..edges_end + 4].try_into()?);
    let actual_crc = crc32fast::hash(&data[..edges_end]);
    if expected_crc != actual_crc {
        anyhow::bail!(
            "Invalid .rmp file: CRC32 mismatch (expected {}, got {})",
            expected_crc,
            actual_crc
        );
    }

    Ok((nodes, edges))
}

// ── Bearing calculation ──────────────────────────────────────────────

/// Calculate the initial bearing from point 1 to point 2 in degrees.
fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();

    let x = dlon.cos() * lat2_r.sin();
    let y = lat1_r.cos() * lat2_r.sin() - lat1_r.sin() * lat2_r.cos() * dlon.cos();

    let bearing_rad = y.atan2(x);
    (bearing_rad.to_degrees() + 360.0) % 360.0
}

// ── CPP internals ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct AdjEntry {
    to: u32,
    weight_m: f64,
    edge_idx: usize,
}

/// In-memory CPP result: both the summary stats and the ordered node-IDs of the Eulerian circuit.
#[derive(Debug, Clone)]
pub struct CppOutput {
    /// Summary statistics (distance, turns, etc.)
    pub summary: OptimizeResult,
    /// Ordered node indices forming the Eulerian circuit.
    pub circuit: Vec<u32>,
}

/// Solve the Chinese Postman Problem directly from an in-memory graph.
///
/// This is the pure algorithm — no filesystem dependency.
/// - `depot` is an optional (lat, lon) — the solver snaps to the nearest node.
///
/// Returns a `CppOutput` with both summary statistics and the full circuit.
pub fn solve_cpp(
    nodes: &[RmpNode],
    edges: &[RmpEdge],
    oneway: OnewayMode,
    depot: Option<(f64, f64)>,
) -> anyhow::Result<CppOutput> {
    let start = Instant::now();

    if nodes.is_empty() || edges.is_empty() {
        return Ok(CppOutput {
            summary: OptimizeResult {
                total_distance_km: 0.0,
                total_segments: 0,
                deadhead_distance_km: 0.0,
                efficiency_pct: 100.0,
                turns: TurnSummary {
                    left: 0,
                    right: 0,
                    u_turn: 0,
                    straight: 0,
                },
                elapsed_ms: 0,
                num_routes: 1,
            },
            circuit: Vec::new(),
        });
    }

    // Build adjacency list
    let n = nodes.len();
    let mut adj: Vec<Vec<AdjEntry>> = vec![vec![]; n];

    for (idx, edge) in edges.iter().enumerate() {
        let from = edge.from as usize;
        let to = edge.to as usize;

        adj[from].push(AdjEntry {
            to: edge.to,
            weight_m: edge.weight_m,
            edge_idx: idx,
        });

        match oneway {
            OnewayMode::Ignore => {
                adj[to].push(AdjEntry {
                    to: edge.from,
                    weight_m: edge.weight_m,
                    edge_idx: idx,
                });
            }
            OnewayMode::Respect => {
                if edge.oneway == 0 {
                    adj[to].push(AdjEntry {
                        to: edge.from,
                        weight_m: edge.weight_m,
                        edge_idx: idx,
                    });
                }
            }
            OnewayMode::Reverse => {
                if edge.oneway == 1 {
                    adj[to].push(AdjEntry {
                        to: edge.from,
                        weight_m: edge.weight_m,
                        edge_idx: idx,
                    });
                    adj[from].retain(|e| e.edge_idx != idx);
                } else {
                    adj[to].push(AdjEntry {
                        to: edge.from,
                        weight_m: edge.weight_m,
                        edge_idx: idx,
                    });
                }
            }
        }
    }

    // Find odd-degree vertices
    let mut degrees = vec![0usize; n];
    for (i, adj_list) in adj.iter().enumerate() {
        degrees[i] = adj_list.len();
    }
    let odd_vertices: Vec<usize> = (0..n).filter(|&i| !degrees[i].is_multiple_of(2)).collect();

    // Minimum weight perfect matching (greedy nearest-neighbor using Dijkstra)
    let mut duplicate_edges: Vec<(usize, usize, f64, usize)> = Vec::new();
    let mut matched = vec![false; n];
    for i in 0..n {
        if degrees[i].is_multiple_of(2) {
            matched[i] = true;
        }
    }

    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Copy, Clone, PartialEq)]
    struct State {
        cost: f64,
        position: usize,
    }
    impl Eq for State {}
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .cost
                .partial_cmp(&self.cost)
                .unwrap_or(Ordering::Equal)
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    for &u in &odd_vertices {
        if matched[u] {
            continue;
        }

        let mut dists = vec![f64::MAX; n];
        let mut prev = vec![None; n];
        let mut heap = BinaryHeap::new();

        dists[u] = 0.0;
        heap.push(State {
            cost: 0.0,
            position: u,
        });

        let mut best_v = None;

        while let Some(State { cost, position }) = heap.pop() {
            if cost > dists[position] {
                continue;
            }

            if position != u && !matched[position] {
                best_v = Some(position);
                break;
            }

            for edge in &adj[position] {
                let next = State {
                    cost: cost + edge.weight_m,
                    position: edge.to as usize,
                };
                if next.cost < dists[next.position] {
                    dists[next.position] = next.cost;
                    prev[next.position] = Some((position, edge.weight_m, edge.edge_idx));
                    heap.push(next);
                }
            }
        }

        if let Some(v) = best_v {
            matched[u] = true;
            matched[v] = true;

            // Trace path back from v to u
            let mut curr = v;
            while let Some((p, weight, eidx)) = prev[curr] {
                duplicate_edges.push((p, curr, weight, eidx));
                curr = p;
            }
        } else {
            // Fallback: If disconnected, just skip (graph must be disconnected)
            matched[u] = true;
        }
    }

    // Add duplicate edges
    let deadhead_edge_idx = usize::MAX;
    for &(u, v, weight, eidx) in &duplicate_edges {
        adj[u].push(AdjEntry {
            to: v as u32,
            weight_m: weight,
            edge_idx: eidx,
        });
        adj[v].push(AdjEntry {
            to: u as u32,
            weight_m: weight,
            edge_idx: eidx,
        });
    }

    // Find Eulerian circuit using Hierholzer's algorithm
    let start_node = if let Some((dep_lat, dep_lon)) = depot {
        let mut best_node = 0;
        let mut best_dist = f64::MAX;
        for (i, node) in nodes.iter().enumerate() {
            let dist = haversine_m(dep_lat, dep_lon, node.lat, node.lon);
            if dist < best_dist {
                best_dist = dist;
                best_node = i;
            }
        }
        best_node
    } else if !odd_vertices.is_empty() {
        odd_vertices[0]
    } else {
        0
    };

    let mut stack = vec![(start_node as u32, None)];
    let mut circuit_with_edges: Vec<(u32, Option<AdjEntry>)> = Vec::new();

    while let Some(&(v_u32, _)) = stack.last() {
        let v = v_u32 as usize;
        if let Some(edge) = adj[v].pop() {
            if let Some(pos) = adj[edge.to as usize].iter().position(|e| {
                e.to == v as u32 && e.edge_idx == edge.edge_idx && e.weight_m == edge.weight_m
            }) {
                adj[edge.to as usize].swap_remove(pos);
            }
            stack.push((edge.to, Some(edge)));
        } else if let Some((v_u32, e)) = stack.pop() {
            circuit_with_edges.push((v_u32, e));
        }
    }

    circuit_with_edges.reverse();
    let circuit: Vec<u32> = circuit_with_edges.iter().map(|(v, _)| *v).collect();

    // Compute total distance, deadhead distance, and turn summary
    let mut total_distance_m = 0.0;
    let mut deadhead_distance_m = 0.0;
    let mut total_segments = 0usize;
    let mut turns = TurnSummary {
        left: 0,
        right: 0,
        u_turn: 0,
        straight: 0,
    };
    let mut edge_traversal_count = vec![0u32; edges.len()];

    for entry in circuit_with_edges.iter().skip(1) {
        if let Some(e) = &entry.1 {
            total_distance_m += e.weight_m;
            total_segments += 1;
            if e.edge_idx == deadhead_edge_idx {
                deadhead_distance_m += e.weight_m;
            } else {
                edge_traversal_count[e.edge_idx] += 1;
                if edge_traversal_count[e.edge_idx] > 1 {
                    deadhead_distance_m += e.weight_m;
                }
            }
        }
    }

    // Turn classification
    if circuit.len() > 2 {
        for i in 1..circuit.len().saturating_sub(1) {
            let prev = circuit[i - 1] as usize;
            let curr = circuit[i] as usize;
            let next = circuit[i + 1] as usize;
            if prev == curr || curr == next {
                continue;
            }
            let b_in = bearing(
                nodes[prev].lat,
                nodes[prev].lon,
                nodes[curr].lat,
                nodes[curr].lon,
            );
            let b_out = bearing(
                nodes[curr].lat,
                nodes[curr].lon,
                nodes[next].lat,
                nodes[next].lon,
            );
            let b_in_reverse = (b_in + 180.0).normalize(0.0, 360.0);
            let delta = b_out - b_in_reverse;
            match classify_turn(delta) {
                "left" => turns.left += 1,
                "right" => turns.right += 1,
                "u_turn" => turns.u_turn += 1,
                _ => turns.straight += 1,
            }
        }
    }

    // Compute efficiency
    let effective_distance_m = total_distance_m - deadhead_distance_m;
    let efficiency_pct = if total_distance_m > 0.0 {
        (effective_distance_m / total_distance_m) * 100.0
    } else {
        100.0
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(CppOutput {
        summary: OptimizeResult {
            total_distance_km: total_distance_m / 1000.0,
            total_segments,
            deadhead_distance_km: deadhead_distance_m / 1000.0,
            efficiency_pct,
            turns,
            elapsed_ms,
            num_routes: 1,
        },
        circuit,
    })
}

/// Run the Chinese Postman Problem route optimization (filesystem wrapper).
fn run_cpp_optimize(req: &OptimizeRequest) -> anyhow::Result<OptimizeResult> {
    let start = Instant::now();

    let mut file_data = Vec::new();
    {
        let mut file = std::fs::File::open(&req.cache_file)
            .map_err(|e| anyhow::anyhow!("Failed to open .rmp file '{}': {}", req.cache_file, e))?;
        file.read_to_end(&mut file_data)?;
    }
    let (nodes, edges) = read_rmp_file(&file_data)?;

    let output = solve_cpp(&nodes, &edges, req.oneway_mode, req.depot)?;

    if let Some(ref route_path) = req.route_file {
        if route_path.ends_with(".json") {
            let route_json = serde_json::json!({
                "route": output.circuit,
                "total_distance_km": output.summary.total_distance_km,
                "deadhead_distance_km": output.summary.deadhead_distance_km,
                "efficiency_pct": output.summary.efficiency_pct,
                "nodes": nodes.iter().enumerate().map(|(i, n)| serde_json::json!({ "id": i, "lat": n.lat, "lon": n.lon })).collect::<Vec<_>>(),
            });
            std::fs::write(route_path, serde_json::to_string_pretty(&route_json)?)?;
        } else {
            write_gpx_cpp(route_path, &nodes, &output.circuit)?;
        }
    }

    let mut result = output.summary;
    result.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(result)
}

// ── VRP route optimization ────────────────────────────────────────────

/// Run the Vehicle Routing Problem optimization.
async fn run_vrp_optimize(req: &OptimizeRequest) -> anyhow::Result<OptimizeResult> {
    let start = Instant::now();

    // 1. Read .rmp
    let mut file_data = Vec::new();
    std::fs::File::open(&req.cache_file)?.read_to_end(&mut file_data)?;
    let (nodes, _edges) = read_rmp_file(&file_data)?;

    if nodes.is_empty() {
        anyhow::bail!("No nodes found in .rmp file");
    }

    // 2. Build VRP Stops
    let mut stops: Vec<VRPSolverStop> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| VRPSolverStop {
            lat: n.lat,
            lon: n.lon,
            label: format!("Node {}", i),
            demand: Some(1.0),
            arrival_time: None,
        })
        .collect();

    // Add depot at start if specified
    if let Some((dlat, dlon)) = req.depot {
        stops.insert(
            0,
            VRPSolverStop {
                lat: dlat,
                lon: dlon,
                label: "Depot".into(),
                demand: Some(0.0),
                arrival_time: None,
            },
        );
    }

    // 3. Build Distance Matrix (Haversine for now)
    let matrix = super::vrp::utils::build_haversine_matrix(&stops, 40.0);

    // 4. Solve
    let vrp_input = VRPSolverInput {
        locations: stops.clone(),
        num_vehicles: req.num_vehicles,
        vehicle_capacity: (stops.len() as f64 / req.num_vehicles as f64).ceil() * 1.5,
        objective: VrpObjective::MinDistance,
        matrix: Some(matrix),
        service_time_secs: Some(30.0),
        use_time_windows: false,
        window_open: None,
        window_close: None,
    };

    let output = solve_with(&req.solver_id, &vrp_input)
        .await
        .map_err(|e| anyhow::anyhow!("VRP Solver error: {}", e))?;

    // 5. Compute Stats
    let mut turns = TurnSummary {
        left: 0,
        right: 0,
        u_turn: 0,
        straight: 0,
    };

    if let Some(ref routes) = output.routes {
        for route in routes {
            if route.len() > 2 {
                for i in 1..route.len() - 1 {
                    let prev = &route[i - 1];
                    let curr = &route[i];
                    let next = &route[i + 1];
                    let b_in = bearing(prev.lat, prev.lon, curr.lat, curr.lon);
                    let b_out = bearing(curr.lat, curr.lon, next.lat, next.lon);
                    let b_in_rev = (b_in + 180.0) % 360.0;
                    let delta = b_out - b_in_rev;
                    match classify_turn(delta) {
                        "left" => turns.left += 1,
                        "right" => turns.right += 1,
                        "u_turn" => turns.u_turn += 1,
                        _ => turns.straight += 1,
                    }
                }
            }
        }
    }

    let total_dist_km: f64 = output.total_distance_km.parse().unwrap_or(0.0);

    // 6. Write GPX
    if let Some(ref path) = req.route_file {
        if let Some(ref routes) = output.routes {
            write_gpx_multi(path, routes)?;
        }
    }

    Ok(OptimizeResult {
        total_distance_km: total_dist_km,
        total_segments: output.stops.len(),
        deadhead_distance_km: 0.0,
        efficiency_pct: 100.0,
        turns,
        elapsed_ms: start.elapsed().as_millis() as u64,
        num_routes: output.routes.as_ref().map(|r| r.len()).unwrap_or(1),
    })
}

pub(crate) fn write_gpx_multi(path: &str, routes: &[Vec<VRPSolverStop>]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(
        file,
        "<gpx version=\"1.1\" creator=\"rmpca\" xmlns=\"http://www.topografix.com/GPX/1/1\">"
    )?;
    for (i, route) in routes.iter().enumerate() {
        writeln!(file, "  <trk>")?;
        writeln!(file, "    <name>Optimized Route {}</name>", i + 1)?;
        writeln!(file, "    <trkseg>")?;
        for stop in route {
            writeln!(
                file,
                "      <trkpt lat=\"{:.7}\" lon=\"{:.7}\"></trkpt>",
                stop.lat, stop.lon
            )?;
        }
        writeln!(file, "    </trkseg>")?;
        writeln!(file, "  </trk>")?;
    }
    writeln!(file, "</gpx>")?;

    Ok(())
}

/// Write a GPX track file from a CPP circuit.
#[allow(dead_code)]
pub fn write_gpx_cpp(path: &str, nodes: &[RmpNode], circuit: &[u32]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(
        file,
        "<gpx version=\"1.1\" creator=\"rmpca\" xmlns=\"http://www.topografix.com/GPX/1/1\">"
    )?;
    writeln!(file, "  <trk>")?;
    writeln!(file, "    <name>CPP Route</name>")?;
    writeln!(file, "    <trkseg>")?;
    for &idx in circuit {
        if (idx as usize) < nodes.len() {
            writeln!(
                file,
                "      <trkpt lat=\"{:.7}\" lon=\"{:.7}\"></trkpt>",
                nodes[idx as usize].lat, nodes[idx as usize].lon
            )?;
        }
    }
    writeln!(file, "    </trkseg>")?;
    writeln!(file, "  </trk>")?;
    writeln!(file, "</gpx>")?;
    Ok(())
}

// ── Dispatcher ───────────────────────────────────────────────────────

/// Run route optimization, dispatching to CPP or VRP based on `req.mode`.
pub async fn run_optimize(req: &OptimizeRequest) -> anyhow::Result<OptimizeResult> {
    match req.mode {
        SolverMode::Cpp => run_cpp_optimize(req),
        SolverMode::Vrp => run_vrp_optimize(req).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_turn_straight() {
        assert_eq!(classify_turn(0.0), "straight");
    }
    #[test]
    fn test_classify_turn_left() {
        assert_eq!(classify_turn(-90.0), "left");
    }
    #[test]
    fn test_classify_turn_right() {
        assert_eq!(classify_turn(90.0), "right");
    }
    #[test]
    fn test_classify_turn_uturn() {
        assert_eq!(classify_turn(170.0), "u_turn");
        assert_eq!(classify_turn(-170.0), "u_turn");
    }
    #[test]
    fn test_haversine_m_known_distance() {
        let dist = haversine_m(40.7128, -74.0060, 34.0522, -118.2437);
        assert!((dist - 3_935_000.0).abs() < 10_000.0);
    }
    #[test]
    fn test_read_rmp_file_invalid() {
        assert!(read_rmp_file(&[]).is_err());
    }
    #[test]
    fn test_solver_mode_default_is_cpp() {
        assert_eq!(SolverMode::default(), SolverMode::Cpp);
    }
    #[test]
    fn test_optimize_simple_network_cpp() {
        let nodes: Vec<(f64, f64)> = vec![(40.7128, -74.006), (40.748, -73.985), (40.678, -73.944)];
        let edges: Vec<(u32, u32, f64, u8)> = vec![
            (0u32, 1u32, 5000.0, 0u8),
            (1u32, 2u32, 6000.0, 0u8),
            (2u32, 0u32, 7000.0, 0u8),
        ];
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RMP1");
        buf.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(edges.len() as u32).to_le_bytes());
        for (lat, lon) in &nodes {
            buf.extend_from_slice(&(*lat).to_le_bytes());
            buf.extend_from_slice(&(*lon).to_le_bytes());
        }
        for (from, to, weight, oneway) in &edges {
            buf.extend_from_slice(&(*from).to_le_bytes());
            buf.extend_from_slice(&(*to).to_le_bytes());
            buf.extend_from_slice(&(*weight).to_le_bytes());
            buf.push(*oneway);
        }
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        let temp_path = "/tmp/v2rmp_test.rmp";
        std::fs::write(temp_path, &buf).unwrap();
        let req = OptimizeRequest {
            cache_file: temp_path.to_string(),
            route_file: None,
            turn_penalties: TurnPenalties::default(),
            depot: None,
            oneway_mode: OnewayMode::Ignore,
            mode: SolverMode::Cpp,
            num_vehicles: 1,
            solver_id: "default".to_string(),
        };
        let _result = run_optimize(&req);
        // run_optimize is async but CPP is sync, so we need to use tokio
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(run_optimize(&req)).unwrap();
        assert!(result.total_distance_km > 0.0);
        let _ = std::fs::remove_file(temp_path);
    }

    // ── Mathematical correctness tests (equivalent to Lean 4 theorems) ────

    /// Helper: build RmpNode/RmpEdge vectors from raw tuples.
    fn make_graph(
        coords: &[(f64, f64)],
        edge_defs: &[(u32, u32, f64, u8)],
    ) -> (Vec<RmpNode>, Vec<RmpEdge>) {
        let nodes: Vec<RmpNode> = coords
            .iter()
            .map(|&(lat, lon)| RmpNode { lat, lon })
            .collect();
        let edges: Vec<RmpEdge> = edge_defs
            .iter()
            .map(|&(from, to, weight_m, oneway)| RmpEdge {
                from,
                to,
                weight_m,
                oneway,
            })
            .collect();
        (nodes, edges)
    }

    /// **Theorem (Eulerian circuit exists)**:
    /// After the CPP solver adds duplicate edges to make all vertex degrees even,
    /// an Eulerian circuit must exist. This is a direct consequence of Euler's theorem:
    /// A connected graph has an Eulerian circuit iff every vertex has even degree.
    ///
    /// Test: for any connected graph, after running solve_cpp, verify that the
    /// output circuit is a valid closed walk that traverses every edge.
    #[test]
    fn test_cpp_circuit_is_eulerian() {
        // Triangle graph - already Eulerian (every vertex degree 2)
        let (nodes, edges) = make_graph(
            &[(45.0, -73.0), (45.01, -73.0), (45.005, -73.01)],
            &[(0, 1, 1100.0, 0), (1, 2, 1100.0, 0), (2, 0, 1100.0, 0)],
        );
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        // Property 1: circuit is non-empty
        assert!(!out.circuit.is_empty(), "circuit must not be empty");

        // Property 2: circuit is a closed walk (first == last)
        assert_eq!(
            out.circuit.first(),
            out.circuit.last(),
            "circuit must be closed"
        );

        // Property 3: every node in the circuit is a valid node index
        for &v in &out.circuit {
            assert!(
                (v as usize) < nodes.len(),
                "circuit node {} out of range (max {})",
                v,
                nodes.len() - 1
            );
        }

        // Property 4: every consecutive pair in the circuit is connected by an edge
        let mut adj: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for e in &edges {
            adj.insert((e.from, e.to));
            if e.oneway == 0 {
                adj.insert((e.to, e.from));
            }
        }
        for window in out.circuit.windows(2) {
            let (a, b) = (window[0], window[1]);
            assert!(
                adj.contains(&(a, b)),
                "circuit edge ({}, {}) is not in the graph",
                a,
                b
            );
        }
    }

    /// **Theorem (Edge coverage)**:
    /// The CPP circuit must traverse every edge in the original graph at least once.
    /// This is the defining property of the Chinese Postman Problem.
    #[test]
    fn test_cpp_covers_all_edges() {
        // 4 nodes, 5 edges (node 0 has degree 3 = odd)
        let (nodes, edges) = make_graph(
            &[
                (45.0, -73.0),
                (45.01, -73.0),
                (45.0, -73.01),
                (45.01, -73.01),
            ],
            &[
                (0, 1, 1100.0, 0),
                (0, 2, 1100.0, 0),
                (1, 3, 1100.0, 0),
                (2, 3, 1100.0, 0),
                (0, 3, 1500.0, 0),
            ],
        );
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        // Build a multiset of traversed directed edges from the circuit
        let mut traversed: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for window in out.circuit.windows(2) {
            *traversed.entry((window[0], window[1])).or_insert(0) += 1;
        }

        // Every original undirected edge must appear at least once (in either direction)
        for (i, e) in edges.iter().enumerate() {
            let forward_count = traversed.get(&(e.from, e.to)).copied().unwrap_or(0);
            let reverse_count = if e.oneway == 0 {
                traversed.get(&(e.to, e.from)).copied().unwrap_or(0)
            } else {
                0
            };
            assert!(
                forward_count + reverse_count >= 1,
                "edge {} ({}, {}) not traversed in circuit",
                i,
                e.from,
                e.to
            );
        }
    }

    /// **Theorem (Handshaking lemma / Parity correction)**:
    /// In any graph, the number of odd-degree vertices is even.
    /// The CPP algorithm must add duplicate edges that pair up all odd-degree vertices,
    /// making every vertex even-degree. This is a necessary condition for Eulerian circuit.
    #[test]
    fn test_cpp_handshaking_lemma() {
        // Path graph: A-B-C-D (3 edges, 2 odd-degree endpoints)
        let (nodes, edges) = make_graph(
            &[
                (45.0, -73.0),
                (45.01, -73.0),
                (45.02, -73.0),
                (45.03, -73.0),
            ],
            &[(0, 1, 1100.0, 0), (1, 2, 1100.0, 0), (2, 3, 1100.0, 0)],
        );

        let n = nodes.len();
        let mut degree = vec![0u32; n];
        for e in &edges {
            degree[e.from as usize] += 1;
            if e.oneway == 0 {
                degree[e.to as usize] += 1;
            }
        }
        let odd_count = degree.iter().filter(|&&d| d % 2 == 1).count();
        assert_eq!(
            odd_count % 2,
            0,
            "handshaking lemma: odd-degree count must be even"
        );

        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        // In an Eulerian circuit, every node must have even degree in the walk.
        // Degree = number of times a node appears as "from" endpoint + "to" endpoint.
        // For circuit [v0, v1, ..., vk] with v0 == vk, edges are (vi, vi+1).
        let mut walk_degree = vec![0u32; n];
        for window in out.circuit.windows(2) {
            walk_degree[window[0] as usize] += 1; // "from" endpoint
            walk_degree[window[1] as usize] += 1; // "to" endpoint
        }
        for i in 0..n {
            assert_eq!(
                walk_degree[i] % 2,
                0,
                "node {} has odd walk-degree {} in Eulerian circuit - parity violation",
                i,
                walk_degree[i]
            );
        }
    }

    /// **Theorem (Optimality on Eulerian graphs)**:
    /// If the original graph is already Eulerian (all vertices even degree),
    /// the CPP solution must equal the Eulerian circuit with zero deadhead.
    /// No edges need to be duplicated.
    #[test]
    fn test_cpp_eulerian_graph_zero_deadhead() {
        // Square: A-B-C-D-A - all vertices degree 2 (Eulerian)
        let (nodes, edges) = make_graph(
            &[
                (45.0, -73.0),
                (45.01, -73.0),
                (45.01, -73.01),
                (45.0, -73.01),
            ],
            &[
                (0, 1, 1100.0, 0),
                (1, 2, 1100.0, 0),
                (2, 3, 1100.0, 0),
                (3, 0, 1100.0, 0),
            ],
        );
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        let sum_weights_km: f64 = edges.iter().map(|e| e.weight_m / 1000.0).sum();
        let tolerance = 0.01;

        assert!(
            (out.summary.deadhead_distance_km - 0.0).abs() < tolerance,
            "Eulerian graph must have zero deadhead, got {:.6} km",
            out.summary.deadhead_distance_km
        );
        assert!(
            (out.summary.total_distance_km - sum_weights_km).abs() < tolerance,
            "Eulerian graph: total distance {:.6} km should equal sum of weights {:.6} km",
            out.summary.total_distance_km,
            sum_weights_km
        );
        assert!(
            out.summary.efficiency_pct > 99.9,
            "Eulerian graph must have ~100% efficiency, got {:.1}%",
            out.summary.efficiency_pct
        );
    }

    /// **Theorem (Deadhead correctness on non-Eulerian graphs)**:
    /// If the graph is not Eulerian, the CPP solver must add deadhead edges.
    /// The total distance must equal the sum of original edge weights plus
    /// deadhead distance. This is a conservation law.
    #[test]
    fn test_cpp_distance_conservation() {
        // Path graph (non-Eulerian): A-B-C
        let (nodes, edges) = make_graph(
            &[(45.0, -73.0), (45.01, -73.0), (45.02, -73.0)],
            &[(0, 1, 1100.0, 0), (1, 2, 1100.0, 0)],
        );
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        let sum_weights_km: f64 = edges.iter().map(|e| e.weight_m / 1000.0).sum();
        let expected_total = sum_weights_km + out.summary.deadhead_distance_km;
        let tolerance = 0.01;

        assert!(
            (out.summary.total_distance_km - expected_total).abs() < tolerance,
            "distance conservation: total ({:.6}) should equal original ({:.6}) + deadhead ({:.6})",
            out.summary.total_distance_km,
            sum_weights_km,
            out.summary.deadhead_distance_km
        );

        assert!(
            out.summary.deadhead_distance_km > 0.0,
            "non-Eulerian graph must have positive deadhead, got 0"
        );
    }

    /// **Theorem (Circuit connectivity / Valid walk)**:
    /// Every consecutive pair in the circuit must correspond to an actual edge
    /// in the graph (original or duplicate). This is the "valid walk" property.
    #[test]
    fn test_cpp_circuit_is_valid_walk() {
        // Cycle graph (Eulerian): A-B-C-D-E-A - all vertices degree 2
        // Using an Eulerian graph means no deadhead edges are added,
        // so every circuit step must be an original edge.
        let (nodes, edges) = make_graph(
            &[
                (45.0, -73.0),
                (45.01, -73.0),
                (45.01, -73.01),
                (45.0, -73.02),
                (44.99, -73.01),
            ],
            &[
                (0, 1, 1100.0, 0),
                (1, 2, 1100.0, 0),
                (2, 3, 1100.0, 0),
                (3, 4, 1100.0, 0),
                (4, 0, 1100.0, 0),
            ],
        );
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        let mut adj: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for e in &edges {
            adj.insert((e.from, e.to));
            if e.oneway == 0 {
                adj.insert((e.to, e.from));
            }
        }

        for (i, window) in out.circuit.windows(2).enumerate() {
            let (a, b) = (window[0], window[1]);
            assert!(
                adj.contains(&(a, b)),
                "circuit step {}: ({}, {}) is not a valid edge",
                i,
                a,
                b
            );
        }

        assert_eq!(
            out.circuit.first(),
            out.circuit.last(),
            "circuit must start and end at the same node"
        );
    }

    /// **Theorem (Single edge graph)**:
    /// The CPP algorithm must handle the degenerate case of a single edge.
    /// The circuit should traverse it forward and back (deadhead = edge weight).
    #[test]
    fn test_cpp_single_edge() {
        let (nodes, edges) = make_graph(&[(45.0, -73.0), (45.01, -73.0)], &[(0, 1, 1100.0, 0)]);
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        assert!(
            out.circuit.len() >= 2,
            "circuit must have at least 2 nodes for 1 edge"
        );
        assert_eq!(
            out.circuit.first(),
            out.circuit.last(),
            "circuit must be closed"
        );
        assert!(out.circuit.contains(&0), "node 0 must appear in circuit");
        assert!(out.circuit.contains(&1), "node 1 must appear in circuit");

        let edge_km = 1100.0 / 1000.0;
        assert!(
            out.summary.total_distance_km >= edge_km - 0.01,
            "total distance must be at least the edge weight"
        );
    }

    /// **Theorem (One-way constraint)**:
    /// When one-way streets are respected, the circuit must never traverse
    /// a one-way edge in the reverse direction.
    #[test]
    fn test_cpp_oneway_respected() {
        let (nodes, edges) = make_graph(
            &[(45.0, -73.0), (45.01, -73.0)],
            &[(0, 1, 1100.0, 1)], // oneway = 1
        );
        let out = solve_cpp(&nodes, &edges, OnewayMode::Respect, None).unwrap();

        for window in out.circuit.windows(2) {
            let (a, b) = (window[0], window[1]);
            assert!(
                !(a == 1 && b == 0),
                "circuit traverses one-way edge backwards: (1, 0)"
            );
        }
    }

    /// **Theorem (Empty graph base case)**:
    /// The CPP algorithm must return an empty circuit for an empty graph
    /// without panicking.
    #[test]
    fn test_cpp_empty_graph() {
        let nodes: Vec<RmpNode> = vec![];
        let edges: Vec<RmpEdge> = vec![];
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        assert!(
            out.circuit.is_empty(),
            "empty graph must produce empty circuit"
        );
        assert_eq!(out.summary.total_distance_km, 0.0);
        assert_eq!(out.summary.total_segments, 0);
        assert_eq!(out.summary.deadhead_distance_km, 0.0);
    }

    /// **Theorem (Depot snapping)**:
    /// When a depot is specified, the CPP circuit must start at the node
    /// closest to the depot coordinates.
    #[test]
    fn test_cpp_depot_snapping() {
        let (nodes, edges) = make_graph(
            &[(45.0, -73.0), (45.01, -73.0), (45.02, -73.0)],
            &[(0, 1, 1100.0, 0), (1, 2, 1100.0, 0)],
        );

        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, Some((45.01, -73.0))).unwrap();

        assert_eq!(
            out.circuit.first().copied(),
            Some(1u32),
            "circuit must start at node closest to depot"
        );
        assert_eq!(
            out.circuit.last().copied(),
            Some(1u32),
            "circuit must end at depot node (closed tour)"
        );
    }

    /// **Theorem (No edge left behind / Completeness)**:
    /// For a graph with many edges, every single edge must be traversed at least once.
    /// This is the completeness guarantee of CPP.
    #[test]
    fn test_cpp_completeness_large() {
        // Grid: 3x3 nodes, 12 edges
        let coords: Vec<(f64, f64)> = (0..3)
            .flat_map(|r| (0..3).map(move |c| (45.0 + r as f64 * 0.01, -73.0 + c as f64 * 0.01)))
            .collect();
        let mut edge_defs: Vec<(u32, u32, f64, u8)> = Vec::new();
        for r in 0u32..3 {
            for c in 0u32..3 {
                let idx = r * 3 + c;
                if c < 2 {
                    edge_defs.push((idx, idx + 1, 1100.0, 0));
                }
                if r < 2 {
                    edge_defs.push((idx, idx + 3, 1100.0, 0));
                }
            }
        }

        let (nodes, edges) = make_graph(&coords, &edge_defs);
        let out = solve_cpp(&nodes, &edges, OnewayMode::Ignore, None).unwrap();

        let mut traversed: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for window in out.circuit.windows(2) {
            *traversed.entry((window[0], window[1])).or_insert(0) += 1;
        }

        for (i, e) in edges.iter().enumerate() {
            let fwd = traversed.get(&(e.from, e.to)).copied().unwrap_or(0);
            let rev = if e.oneway == 0 {
                traversed.get(&(e.to, e.from)).copied().unwrap_or(0)
            } else {
                0
            };
            assert!(
                fwd + rev >= 1,
                "edge {} ({}, {}) not covered - CPP completeness violation",
                i,
                e.from,
                e.to
            );
        }
    }
}
