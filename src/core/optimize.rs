use serde::{Deserialize, Serialize};
use std::io::Read;
use std::time::Instant;

use super::vrp::registry::solve_with;
use super::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};

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
    pub num_vehicles: usize,
    pub solver_id: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSummary {
    pub left: u32,
    pub right: u32,
    pub u_turn: u32,
    pub straight: u32,
}

// ── Binary format structures ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RmpNode {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Debug, Clone)]
pub struct RmpEdge {
    pub from: u32,
    pub to: u32,
    pub weight_m: f64,
    pub oneway: u8,
}

// ── Haversine & turn classification ──────────────────────────────────

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    r * c
}

pub fn classify_turn(bearing_delta: f64) -> &'static str {
    let d = normalize_angle(bearing_delta, -180.0, 180.0);
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

fn normalize_angle(val: f64, lower: f64, upper: f64) -> f64 {
    let width = upper - lower;
    let mut v = val;
    while v < lower {
        v += width;
    }
    while v >= upper {
        v -= width;
    }
    v
}

// ── Binary parser ────────────────────────────────────────────────────

const RMP_MAGIC: &[u8; 4] = b"RMP1";

pub fn read_rmp_file(data: &[u8]) -> anyhow::Result<(Vec<RmpNode>, Vec<RmpEdge>)> {
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
        anyhow::bail!("Invalid .rmp file: too short");
    }
    let mut nodes = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let offset = 12 + i * 16;
        let lat = f64::from_le_bytes(data[offset..offset + 8].try_into()?);
        let lon = f64::from_le_bytes(data[offset + 8..offset + 16].try_into()?);
        nodes.push(RmpNode { lat, lon });
    }
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
    Ok((nodes, edges))
}

fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();
    let x = dlon.cos() * lat2_r.sin();
    let y = lat1_r.cos() * lat2_r.sin() - lat1_r.sin() * lat2_r.cos() * dlon.cos();
    let b_rad = y.atan2(x);
    (b_rad.to_degrees() + 360.0) % 360.0
}

// ── Route optimization ───────────────────────────────────────────────

pub async fn run_optimize(req: &OptimizeRequest) -> anyhow::Result<OptimizeResult> {
    let start = Instant::now();

    // 1. Read .rmp
    let mut file_data = Vec::new();
    std::fs::File::open(&req.cache_file)?.read_to_end(&mut file_data)?;
    let (nodes, _edges) = read_rmp_file(&file_data)?;

    if nodes.is_empty() {
        anyhow::bail!("No nodes found in .rmp file");
    }

    // 2. Build VRP Stops
    // For Route Maintenance, we want to cover EDGES.
    // But VRP solvers in vrp-rust cover STOPS (Nodes).
    // If the user wants to cover all streets, we should use the CPP solver.
    // However, if we are to use the VRP solvers, we'll treat all nodes as stops for now.
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
        deadhead_distance_km: 0.0, // VRP doesn't define deadhead in the same way as CPP
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
        writeln!(file, "    <name>Vehicle {}</name>", i + 1)?;
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
