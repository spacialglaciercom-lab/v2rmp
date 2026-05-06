use crate::core::vrp::registry::solve_with;
use crate::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::time::Instant;

use crate::core::vrp::registry::solve_with;
use crate::core::vrp::types::{VrpObjective, VRPSolverInput, VRPSolverStop};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
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

// ── Haversine & turn classification ──────────────────────────────────

/// Haversine distance in meters between two WGS-84 points.
pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    r * c
}

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

/// Run the Chinese Postman Problem route optimization.
fn run_cpp_optimize(req: &OptimizeRequest) -> anyhow::Result<OptimizeResult> {
    let start = Instant::now();

    // 1. Read the .rmp file
    let mut file_data = Vec::new();
    {
        let mut file = std::fs::File::open(&req.cache_file)
            .map_err(|e| anyhow::anyhow!("Failed to open .rmp file '{}': {}", req.cache_file, e))?;
        file.read_to_end(&mut file_data)?;
    }

    let (nodes, edges) = read_rmp_file(&file_data)?;

    if nodes.is_empty() || edges.is_empty() {
        return Ok(OptimizeResult {
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
            elapsed_ms: start.elapsed().as_millis() as u64,
            num_routes: 1,
        });
    }

    // 2. Build adjacency list
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

        match req.oneway_mode {
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

    // 3. Find odd-degree vertices
    let mut degrees = vec![0usize; n];
    for (i, adj_list) in adj.iter().enumerate() {
        degrees[i] = adj_list.len();
    }
    let odd_vertices: Vec<usize> = (0..n).filter(|&i| degrees[i] % 2 != 0).collect();

    // 4. Minimum weight perfect matching (greedy nearest-neighbor)
    let mut duplicate_edges: Vec<(usize, usize, f64, usize)> = Vec::new();
    let mut matched = vec![false; n];
    for i in 0..n {
        if degrees[i] % 2 == 0 {
            matched[i] = true;
        }
    }

    let mut sorted_odd = odd_vertices.clone();
    sorted_odd.sort_by(|&a, &b| nodes[a].lat.partial_cmp(&nodes[b].lat).unwrap());

    let mut pos_in_sorted = vec![0usize; n];
    for (i, &idx) in sorted_odd.iter().enumerate() {
        pos_in_sorted[idx] = i;
    }

    const METERS_PER_LAT_DEGREE: f64 = 111_111.0;

    for &u in &odd_vertices {
        if matched[u] {
            continue;
        }

        let mut best_v = None;
        let mut best_dist = f64::MAX;
        let u_lat = nodes[u].lat;
        let u_lon = nodes[u].lon;
        let u_pos = pos_in_sorted[u];

        let mut forward_idx = u_pos + 1;
        let mut backward_idx = u_pos.wrapping_sub(1);
        let mut forward_done = forward_idx >= sorted_odd.len();
        let mut backward_done = u_pos == 0;

        while !forward_done || !backward_done {
            if !forward_done {
                let v = sorted_odd[forward_idx];
                let v_lat = nodes[v].lat;
                if (v_lat - u_lat) * METERS_PER_LAT_DEGREE >= best_dist {
                    forward_done = true;
                } else {
                    if !matched[v] {
                        let dist = haversine_m(u_lat, u_lon, v_lat, nodes[v].lon);
                        if dist < best_dist {
                            best_dist = dist;
                            best_v = Some(v);
                        }
                    }
                    forward_idx += 1;
                    if forward_idx >= sorted_odd.len() {
                        forward_done = true;
                    }
                }
            }

            if !backward_done {
                let v = sorted_odd[backward_idx];
                let v_lat = nodes[v].lat;
                if (u_lat - v_lat) * METERS_PER_LAT_DEGREE >= best_dist {
                    backward_done = true;
                } else {
                    if !matched[v] {
                        let dist = haversine_m(u_lat, u_lon, v_lat, nodes[v].lon);
                        if dist < best_dist {
                            best_dist = dist;
                            best_v = Some(v);
                        }
                    }
                    if backward_idx == 0 {
                        backward_done = true;
                    } else {
                        backward_idx -= 1;
                    }
                }
            }
        }

        if let Some(v) = best_v {
            matched[u] = true;
            matched[v] = true;
            duplicate_edges.push((u, v, best_dist, usize::MAX));
        }
    }

    // 5. Add duplicate edges
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

    // 6. Find Eulerian circuit using Hierholzer's algorithm
    let start_node = if let Some((dep_lat, dep_lon)) = req.depot {
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
        } else {
            let (v_u32, e) = stack.pop().unwrap();
            circuit_with_edges.push((v_u32, e));
        }
    }

    circuit_with_edges.reverse();
    let circuit: Vec<u32> = circuit_with_edges.iter().map(|(v, _)| *v).collect();

    // 7. Compute total distance, deadhead distance, and turn summary
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

    // 8. Turn classification
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

    // 9. Compute efficiency
    let effective_distance_m = total_distance_m - deadhead_distance_m;
    let efficiency_pct = if total_distance_m > 0.0 {
        (effective_distance_m / total_distance_m) * 100.0
    } else {
        100.0
    };

    // 10. Write route file
    if let Some(ref route_path) = req.route_file {
        let route_json = serde_json::json!({
            "route": circuit,
            "total_distance_km": total_distance_m / 1000.0,
            "deadhead_distance_km": deadhead_distance_m / 1000.0,
            "efficiency_pct": efficiency_pct,
            "nodes": nodes.iter().enumerate().map(|(i, n)| serde_json::json!({ "id": i, "lat": n.lat, "lon": n.lon })).collect::<Vec<_>>(),
        });
        std::fs::write(route_path, serde_json::to_string_pretty(&route_json)?)?;
    }

    Ok(OptimizeResult {
        total_distance_km: total_distance_m / 1000.0,
        total_segments,
        deadhead_distance_km: deadhead_distance_m / 1000.0,
        efficiency_pct,
        turns,
        elapsed_ms: start.elapsed().as_millis() as u64,
        num_routes: 1,
    })
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

fn write_gpx_multi(path: &str, routes: &[Vec<VRPSolverStop>]) -> anyhow::Result<()> {
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
}
