use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeRequest {
    pub cache_file: String,
    pub route_file: Option<String>,
    pub turn_penalties: TurnPenalties,
    pub depot: Option<(f64, f64)>,
    pub oneway_mode: OnewayMode,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

// ── Binary parser ────────────────────────────────────────────────────

/// Magic bytes for the .rmp binary format.
const RMP_MAGIC: &[u8; 4] = b"RMP1";

/// Parse an `.rmp` binary file and return (nodes, edges).
///
/// Binary layout:
///   [4]  magic "RMP1"
///   [4]  node count (u32 LE)
///   [4]  edge count (u32 LE)
///   [N]  node entries: lat(f64 LE) + lon(f64 LE) = 16 bytes each
///   [E]  edge entries: from(u32 LE) + to(u32 LE) + weight_m(f64 LE) + oneway(u8) = 17 bytes each
///   [4]  CRC32 (u32 LE)
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
    let expected_len = edges_end + 4; // +4 for CRC32

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
    let y = lat1_r.cos() * lat2_r.sin() - lat1_r.sin() * lat2_r.cos() * dlon.sin();

    let bearing_rad = y.atan2(x);
    (bearing_rad.to_degrees() + 360.0) % 360.0
}

// ── Route optimization ───────────────────────────────────────────────

/// Run the Chinese Postman Problem route optimization.
pub fn run_optimize(req: &OptimizeRequest) -> anyhow::Result<OptimizeResult> {
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

        // Add reverse edge for non-oneway or if oneway_mode says so
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
                // Reverse oneway direction
                if edge.oneway == 1 {
                    // Original is oneway forward; only allow reverse
                    adj[to].push(AdjEntry {
                        to: edge.from,
                        weight_m: edge.weight_m,
                        edge_idx: idx,
                    });
                    // Remove forward
                    adj[from].retain(|e| e.edge_idx != idx);
                } else {
                    // Bidirectional: keep both
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
    //    For each odd vertex, find its nearest unmatched odd vertex.
    //    Optimization: Use spatial pruning by sorting odd vertices by latitude.
    let mut duplicate_edges: Vec<(usize, usize, f64, usize)> = Vec::new(); // (from, to, weight, edge_idx)
    let mut matched = vec![false; n];

    // Mark vertices that aren't odd as already "matched"
    for i in 0..n {
        if degrees[i] % 2 == 0 {
            matched[i] = true;
        }
    }

    // Sort odd vertices by latitude for spatial pruning
    let mut sorted_odd = odd_vertices.clone();
    sorted_odd.sort_by(|&a, &b| nodes[a].lat.total_cmp(&nodes[b].lat));

    // Map each node index to its position in the sorted_odd list
    let mut pos_in_sorted = vec![0usize; n];
    for (i, &idx) in sorted_odd.iter().enumerate() {
        pos_in_sorted[idx] = i;
    }

    // Rough constant for meters per degree of latitude
    const METERS_PER_LAT_DEGREE: f64 = 111_111.0;

    for &u in &odd_vertices {
        if matched[u] {
            continue;
        }

        // Find nearest unmatched odd vertex using spatial pruning
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
            // Check forward in sorted list
            if !forward_done {
                let v = sorted_odd[forward_idx];
                let v_lat = nodes[v].lat;
                // If latitude difference alone is greater than best_dist, we can prune forward
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

            // Check backward in sorted list
            if !backward_done {
                let v = sorted_odd[backward_idx];
                let v_lat = nodes[v].lat;
                // If latitude difference alone is greater than best_dist, we can prune backward
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
            duplicate_edges.push((u, v, best_dist, usize::MAX)); // usize::MAX = deadhead marker
        }
    }

    // 5. Add duplicate edges to adjacency list to make all vertices even degree
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
    //    Start from depot if specified, otherwise from first odd vertex or node 0
    let start_node = if let Some((dep_lat, dep_lon)) = req.depot {
        // Find nearest node to depot
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

    // Hierholzer's algorithm
    let mut adj_clone = adj.clone();
    let mut stack = vec![(start_node as u32, None)];
    let mut circuit_with_edges: Vec<(u32, Option<AdjEntry>)> = Vec::new();

    while let Some(&(v_u32, _)) = stack.last() {
        let v = v_u32 as usize;
        if let Some(edge) = adj_clone[v].pop() {
            // Remove reverse edge
            let to_idx = edge.to as usize;
            if let Some(pos) = adj_clone[to_idx].iter().position(|e| {
                e.to == v as u32 && e.edge_idx == edge.edge_idx && e.weight_m == edge.weight_m
            }) {
                adj_clone[to_idx].swap_remove(pos);
            }
            stack.push((edge.to, Some(edge)));
        } else if let Some(node_data) = stack.pop() {
            circuit_with_edges.push(node_data);
        }
    }

    // circuit is in reverse order; reverse it
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

    let mut edge_traversal_count: HashMap<usize, u32> = HashMap::new();

    // Walk the circuit and accumulate distances using stored edge metadata
    for (_, edge_opt) in circuit_with_edges.iter().skip(1) {
        if let Some(e) = edge_opt {
            total_distance_m += e.weight_m;
            total_segments += 1;

            if e.edge_idx == deadhead_edge_idx {
                deadhead_distance_m += e.weight_m;
            } else {
                let count = edge_traversal_count.entry(e.edge_idx).or_insert(0);
                *count += 1;
                if *count > 1 {
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

    // 10. Write route file if requested
    if let Some(ref route_path) = req.route_file {
        let route_json = serde_json::json!({
            "route": circuit,
            "total_distance_km": total_distance_m / 1000.0,
            "deadhead_distance_km": deadhead_distance_m / 1000.0,
            "efficiency_pct": efficiency_pct,
            "nodes": nodes.iter().enumerate().map(|(i, n)| serde_json::json!({
                "id": i,
                "lat": n.lat,
                "lon": n.lon,
            })).collect::<Vec<_>>(),
        });
        let json_str = serde_json::to_string_pretty(&route_json)?;
        std::fs::write(route_path, json_str)?;
    }

    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(OptimizeResult {
        total_distance_km: total_distance_m / 1000.0,
        total_segments,
        deadhead_distance_km: deadhead_distance_m / 1000.0,
        efficiency_pct,
        turns,
        elapsed_ms,
    })
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

/// Adjacency list entry for the graph
#[derive(Debug, Clone, Copy)]
struct AdjEntry {
    to: u32,
    weight_m: f64,
    edge_idx: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_turn_straight() {
        assert_eq!(classify_turn(0.0), "straight");
        assert_eq!(classify_turn(10.0), "straight");
        assert_eq!(classify_turn(-10.0), "straight");
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
        // NYC to LA: ~3,935 km → 3,935,000 m
        let dist = haversine_m(40.7128, -74.0060, 34.0522, -118.2437);
        assert!((dist - 3_935_000.0).abs() < 10_000.0);
    }

    #[test]
    fn test_read_rmp_file_invalid() {
        let result = read_rmp_file(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_rmp_file_bad_magic() {
        let data = b"BADM\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = read_rmp_file(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize_simple_network() {
        // Build a simple .rmp file in memory: a triangle with 3 nodes and 3 edges
        let nodes = vec![
            (40.7128_f64, -74.006_f64), // node 0
            (40.748_f64, -73.985_f64),  // node 1
            (40.678_f64, -73.944_f64),  // node 2
        ];
        let edges = vec![
            (0u32, 1u32, 5000.0f64, 0u8), // edge 0→1, 5km, bidirectional
            (1u32, 2u32, 6000.0f64, 0u8), // edge 1→2, 6km, bidirectional
            (2u32, 0u32, 7000.0f64, 0u8), // edge 2→0, 7km, bidirectional
        ];

        let mut buf: Vec<u8> = Vec::new();
        // Magic
        buf.extend_from_slice(b"RMP1");
        // Node count
        buf.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
        // Edge count
        buf.extend_from_slice(&(edges.len() as u32).to_le_bytes());
        // Nodes
        for (lat, lon) in &nodes {
            buf.extend_from_slice(&(*lat).to_le_bytes());
            buf.extend_from_slice(&(*lon).to_le_bytes());
        }
        // Edges
        for (from, to, weight_m, oneway) in &edges {
            buf.extend_from_slice(&from.to_le_bytes());
            buf.extend_from_slice(&to.to_le_bytes());
            buf.extend_from_slice(&weight_m.to_le_bytes());
            buf.push(*oneway);
        }
        // CRC32
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        // Write to temp file
        let temp_path = "/tmp/v2rmp_test_optimize_simple.rmp";
        std::fs::write(temp_path, &buf).unwrap();

        let req = OptimizeRequest {
            cache_file: temp_path.to_string(),
            route_file: None,
            turn_penalties: TurnPenalties::default(),
            depot: None,
            oneway_mode: OnewayMode::Ignore,
        };

        let result = run_optimize(&req).unwrap();

        assert!(result.total_distance_km > 0.0);
        assert!(result.total_segments > 0);

        // Clean up
        let _ = std::fs::remove_file(temp_path);
    }
}
