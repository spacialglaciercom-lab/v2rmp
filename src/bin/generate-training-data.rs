//! Training data generator for ML models.
//!
//! Generates balanced synthetic VRP instances with solver-biased generation.
//! For each target solver, creates instances with characteristics that favor
//! that solver (tight capacity for Clarke-Wright, wide radial for Sweep, etc.),
//! then runs ALL solvers and records actual results.
//!
//! Output features per instance:
//! - 28-dim instance feature vector (same as Rust features.rs)
//! - best solver id + best distance
//! - relative solver gap (log1p metric)
//! - per-solver distances
//!
//! Usage:
//!   cargo run --release --bin generate-training-data -- 5000 > data/training_balanced.jsonl
//!   cargo run --release --bin generate-training-data -- --balanced 5000 > data/training_balanced.jsonl

use tokio::runtime::Runtime;

use v2rmp::core::haversine_m;
use v2rmp::core::ml::features::InstanceFeatures;
use v2rmp::core::vrp::registry::{get_solver_list, solve_with};
use v2rmp::core::vrp::types::{SolverHyperparams, VRPSolverInput, VRPSolverStop, VrpObjective};
use v2rmp::core::vrp::utils::build_haversine_matrix;

/// Generate a single synthetic VRP instance with depot at index 0.
fn generate_instance(seed_offset: usize) -> Vec<VRPSolverStop> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed_offset.hash(&mut hasher);
    let seed = hasher.finish();
    let mut rng = fast_prng(seed);

    let pattern = seed_offset % 4;
    let n_stops = match seed_offset % 10 {
        0 | 1 => rng.range(10, 25),
        2..=4 => rng.range(20, 60),
        5..=7 => rng.range(50, 150),
        _ => rng.range(100, 250),
use v2rmp::core::vrp::registry::solve_with;
use v2rmp::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};
use v2rmp::core::vrp::utils::build_haversine_matrix;

/// Solvers to evaluate (neural/ONNX solvers excluded for CPU compatibility).
const SOLVER_IDS: &[&str] = &["default", "clarke_wright", "sweep", "or_opt", "two_opt"];

/// Instance generation profiles targeting specific solver strengths.
#[derive(Clone, Copy)]
enum GenProfile {
    /// Default/varied instances
    Default,
    /// Tight capacity, medium size → favors Clarke-Wright savings
    TightCapacity,
    /// Many vehicles, wide radial layout → favors Sweep
    WideRadial,
    /// Large instances, high stops → favors Or-Opt
    LargeDense,
    /// Grid-like with crossings → favors Two-Opt
    GridCross,
}

impl GenProfile {
    fn all() -> Vec<GenProfile> {
        vec![
            GenProfile::Default,
            GenProfile::TightCapacity,
            GenProfile::WideRadial,
            GenProfile::LargeDense,
            GenProfile::GridCross,
        ]
    }

    fn label(&self) -> &'static str {
        match self {
            GenProfile::Default => "default",
            GenProfile::TightCapacity => "tight_capacity",
            GenProfile::WideRadial => "wide_radial",
            GenProfile::LargeDense => "large_dense",
            GenProfile::GridCross => "grid_cross",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// PRNG
// ────────────────────────────────────────────────────────────────────────

struct FastPrng(u64);

fn fast_prng(seed: u64) -> FastPrng {
    FastPrng(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
}

impl FastPrng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if lo >= hi {
            return lo;
        }
        lo + (self.next() as usize % (hi - lo))
    }

    fn range_f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }

    fn gauss(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.unit().max(1e-15);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ────────────────────────────────────────────────────────────────────────
// Instance generation
// ────────────────────────────────────────────────────────────────────────

struct InstanceConfig {
    stops: Vec<VRPSolverStop>,
    num_vehicles: usize,
    vehicle_capacity: f64,
    objective: VrpObjective,
    profile: &'static str,
}

fn generate_biased_instance(rng: &mut FastPrng, profile: GenProfile) -> InstanceConfig {
    match profile {
        GenProfile::Default => generate_default(rng),
        GenProfile::TightCapacity => generate_tight_capacity(rng),
        GenProfile::WideRadial => generate_wide_radial(rng),
        GenProfile::LargeDense => generate_large_dense(rng),
        GenProfile::GridCross => generate_grid_cross(rng),
    }
}

fn generate_default(rng: &mut FastPrng) -> InstanceConfig {
    let n_stops = rng.range(10, 200);
    let n_vehicles = rng.range(1, 12);
    let depot_lat = rng.range_f(35.0, 55.0);
    let depot_lon = rng.range_f(-125.0, -65.0);
    let pattern = rng.range(0, 4);
    let area_scale = rng.range_f(0.5, 3.0);
    let objective = objective_from_idx(rng.range(0, 4));

    let stops = make_stops(rng, n_stops, depot_lat, depot_lon, pattern, area_scale, 1.0, 20.0);

    InstanceConfig {
        stops,
        num_vehicles: n_vehicles.max(1),
        vehicle_capacity: 100.0,
        objective,
        profile: "default",
    }
}

fn generate_tight_capacity(rng: &mut FastPrng) -> InstanceConfig {
    // Tight capacity: demand close to vehicle capacity → savings (CW) excels
    let n_stops = rng.range(20, 150);
    let n_vehicles = rng.range(2, 8);
    let depot_lat = rng.range_f(35.0, 55.0);
    let depot_lon = rng.range_f(-125.0, -65.0);
    let pattern = rng.range(0, 2); // uniform or clustered
    let area_scale = rng.range_f(0.5, 2.0);
    let objective = VrpObjective::MinDistance;

    // Tight capacity: total demand ≈ 80-95% of fleet capacity
    let total_capacity = 100.0 * n_vehicles as f64;
    let per_stop_demand = total_capacity * rng.range_f(0.80, 0.95) / n_stops as f64;

    let stops = make_stops_with_demand(rng, n_stops, depot_lat, depot_lon, pattern, area_scale, per_stop_demand);

    InstanceConfig {
        stops,
        num_vehicles: n_vehicles.max(1),
        vehicle_capacity: 100.0,
        objective,
        profile: "tight_capacity",
    }
}

fn generate_wide_radial(rng: &mut FastPrng) -> InstanceConfig {
    // Wide radial spread with many vehicles → Sweep excels
    let n_stops = rng.range(30, 200);
    let n_vehicles = rng.range(5, 15);
    let depot_lat = rng.range_f(35.0, 55.0);
    let depot_lon = rng.range_f(-125.0, -65.0);
    let area_scale = rng.range_f(2.0, 5.0);
    let objective = match rng.range(0, 3) {
        0 => VrpObjective::MinDistance,
        1 => VrpObjective::MinTime,
        _ => VrpObjective::BalanceLoad,
    };

    let stops = make_stops(rng, n_stops, depot_lat, depot_lon, 3, area_scale, 1.0, 8.0);
    // Relaxed capacity so routes aren't overloaded
    InstanceConfig {
        stops,
        num_vehicles: n_vehicles.max(1),
        vehicle_capacity: 200.0,
        objective,
        profile: "wide_radial",
    }
}

fn generate_large_dense(rng: &mut FastPrng) -> InstanceConfig {
    // Large instances → Or-Opt local search excels
    let n_stops = rng.range(100, 300);
    let n_vehicles = rng.range(1, 6);
    let depot_lat = rng.range_f(35.0, 55.0);
    let depot_lon = rng.range_f(-125.0, -65.0);
    let pattern = rng.range(0, 3);
    let area_scale = rng.range_f(1.0, 4.0);
    let objective = match rng.range(0, 4) {
        0 => VrpObjective::MinDistance,
        1 => VrpObjective::MinTime,
        2 => VrpObjective::BalanceLoad,
        _ => VrpObjective::MinVehicles,
    };

    let stops = make_stops(rng, n_stops, depot_lat, depot_lon, pattern, area_scale, 1.0, 15.0);

    InstanceConfig {
        stops,
        num_vehicles: n_vehicles.max(1),
        vehicle_capacity: 150.0,
        objective,
        profile: "large_dense",
    }
}

fn generate_grid_cross(rng: &mut FastPrng) -> InstanceConfig {
    // Grid layout with noise → crossing edges → Two-Opt excels
    let n_stops = rng.range(20, 120);
    let n_vehicles = rng.range(1, 4);
    let depot_lat = rng.range_f(35.0, 55.0);
    let depot_lon = rng.range_f(-125.0, -65.0);
    let area_scale = rng.range_f(0.8, 2.0);
    let objective = VrpObjective::MinDistance;

    let stops = make_stops(rng, n_stops, depot_lat, depot_lon, 2, area_scale, 1.0, 10.0);

    InstanceConfig {
        stops,
        num_vehicles: n_vehicles.max(1),
        vehicle_capacity: 200.0,
        objective,
        profile: "grid_cross",
    }
}

/// Generate stops with a given pattern.
/// Patterns: 0=uniform, 1=clustered, 2=grid, 3=radial
fn make_stops(
    rng: &mut FastPrng,
    n_stops: usize,
    depot_lat: f64,
    depot_lon: f64,
    pattern: usize,
    area_scale: f64,
    demand_lo: f64,
    demand_hi: f64,
) -> Vec<VRPSolverStop> {
    let mut stops = vec![VRPSolverStop {
        lat: depot_lat,
        lon: depot_lon,
        label: "depot".to_string(),
        demand: Some(0.0),
        arrival_time: None,
    }];

    // Pre-generate cluster centres for clustered pattern
    let n_clusters = rng.range(2, 5);
    let cluster_centres: Vec<(f64, f64)> = (0..n_clusters)
        .map(|_| {
            (
                depot_lat + (rng.unit() - 0.5) * area_scale * 1.2,
                depot_lon + (rng.unit() - 0.5) * area_scale * 1.2,
            )
        })
        .collect();

    for i in 1..=n_stops {
        let (lat, lon) = match pattern % 4 {
            0 => {
                // Uniform
                (
                    depot_lat + (rng.unit() - 0.5) * area_scale * 2.0,
                    depot_lon + (rng.unit() - 0.5) * area_scale * 2.0,
                )
            }
            1 => {
                // Clustered: pick a cluster centre, add small noise
                let c = &cluster_centres[i % cluster_centres.len()];
                (
                    c.0 + (rng.unit() - 0.5) * 0.3,
                    c.1 + (rng.unit() - 0.5) * 0.3,
                )
            }
            2 => {
                // Grid with noise
                let grid_n = ((n_stops as f64).sqrt().ceil() as f64).max(2.0);
                let gx = (i as f64 % grid_n) / grid_n;
                let gy = (i as f64 / grid_n) / grid_n;
                (
                    depot_lat + (gx - 0.5) * area_scale * 1.5 + (rng.unit() - 0.5) * 0.05,
                    depot_lon + (gy - 0.5) * area_scale * 1.5 + (rng.unit() - 0.5) * 0.05,
                )
            }
            3 => {
                // Radial
                let angle = rng.unit() * std::f64::consts::PI * 2.0;
                let radius = rng.unit() * area_scale * 0.8;
                (
                    depot_lat + radius * angle.sin(),
                    depot_lon + radius * angle.cos(),
                )
            }
            _ => unreachable!(),
        };

        stops.push(VRPSolverStop {
            lat,
            lon,
            label: format!("stop_{}", i),
            demand: Some(rng.range_f(demand_lo, demand_hi)),
            arrival_time: None,
        });
    }

    stops
}

/// Generate stops with specific per-stop demand.
fn make_stops_with_demand(
    rng: &mut FastPrng,
    n_stops: usize,
    depot_lat: f64,
    depot_lon: f64,
    pattern: usize,
    area_scale: f64,
    per_stop_demand: f64,
) -> Vec<VRPSolverStop> {
    let mut stops = make_stops(rng, n_stops, depot_lat, depot_lon, pattern, area_scale, per_stop_demand * 0.5, per_stop_demand * 1.5);

    // Override demands to be centered around per_stop_demand
    for stop in stops.iter_mut().skip(1) {
        let d = (per_stop_demand + rng.gauss() * per_stop_demand * 0.2).max(1.0);
        stop.demand = Some(d);
    }

    stops
}

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

fn objective_from_idx(idx: usize) -> VrpObjective {
    match idx % 4 {
        0 => VrpObjective::MinDistance,
        1 => VrpObjective::MinTime,
        2 => VrpObjective::BalanceLoad,
        _ => VrpObjective::MinVehicles,
    }
}

/// Better lower bound: MST-based (Held-Karp relaxation approximation).
/// Uses the fact that an optimal TSP tour is bounded below by MST weight.
fn mst_lower_bound(locations: &[VRPSolverStop], n_vehicles: usize) -> f64 {
    let n = locations.len();
    if n <= 1 {
        return 0.0;
    }

    let depot = &locations[0];
    let others = &locations[1..];

    if others.is_empty() {
        return 0.0;
    }

    let n_others = others.len();

    // Build distance matrix for others
    let mut dists = vec![vec![0.0; n_others]; n_others];
    for i in 0..n_others {
        for j in (i + 1)..n_others {
            let d = haversine_m(others[i].lat, others[i].lon, others[j].lat, others[j].lon) / 1000.0;
            dists[i][j] = d;
            dists[j][i] = d;
        }
    }

    // MST on others (Prim's)
    let mut visited = vec![false; n_others];
    let mut min_edge = vec![f64::MAX; n_others];
    min_edge[0] = 0.0;
    let mut mst_weight = 0.0;
    for _ in 0..n_others {
        let mut u = 0;
        let mut best = f64::MAX;
        for i in 0..n_others {
            if !visited[i] && min_edge[i] < best {
                best = min_edge[i];
                u = i;
            }
        }
        visited[u] = true;
        mst_weight += min_edge[u];
        for v in 0..n_others {
            if !visited[v] && dists[u][v] < min_edge[v] {
                min_edge[v] = dists[u][v];
            }
        }
    }

    // Find two smallest depot-to-node edges
    let mut depot_dists: Vec<f64> = others
        .iter()
        .map(|s| haversine_m(depot.lat, depot.lon, s.lat, s.lon) / 1000.0)
        .collect();
    depot_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min1 = depot_dists.first().copied().unwrap_or(0.0);
    let min2 = depot_dists.get(1).copied().unwrap_or(min1);

    // For VRP with v vehicles, lower bound ≈ MST + 2*min_depot_edge (single vehicle)
    // Multi-vehicle: scale down by roughly 1/sqrt(v)
    let v_factor = if n_vehicles > 1 {
        1.0 / (n_vehicles as f64).sqrt()
    } else {
        1.0
    };
    (mst_weight + min1 + min2) * v_factor
}

fn make_input(
    stops: Vec<VRPSolverStop>,
    num_vehicles: usize,
    objective: VrpObjective,
) -> VRPSolverInput {
    let matrix = build_haversine_matrix(&stops, 40.0);
    VRPSolverInput {
        locations: config.stops.clone(),
        num_vehicles: config.num_vehicles,
        vehicle_capacity: config.vehicle_capacity,
        objective: config.objective.clone(),
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────

fn main() {
    let mut args = std::env::args().skip(1).peekable();

    let mut balanced = false;
    let mut n_instances: usize = 1000;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--balanced" => balanced = true,
            "--uniform" => balanced = false,
            n => {
                n_instances = n.parse().unwrap_or_else(|_| {
                    eprintln!("Usage: generate-training-data [--balanced] N_INSTANCES");
                    std::process::exit(1);
                });
            }
        }
    }

    let rt = Runtime::new().expect("Failed to create Tokio runtime");
    let solver_ids: Vec<String> = SOLVER_IDS.iter().map(|s| s.to_string()).collect();

    eprintln!(
        "Generating {} synthetic VRP instances and evaluating {} solvers...",
        n_instances,
        solver_ids.len()
    );

    for i in 0..n_instances {
        let (config, profile_label) = if balanced {
            // Round-robin through profiles for balanced generation
            let profile = profiles[i % n_profiles];
            let mut rng = fast_prng((i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
            let config = generate_biased_instance(&mut rng, profile);
            (config, profile.label().to_string())
        } else {
            // Original uniform generation
            let mut rng = fast_prng(i as u64);
            let config = generate_default(&mut rng);
            (config, "uniform".to_string())
        };

        let input = make_input(&config);

        let features = InstanceFeatures::from_input(&input);
        let feature_vec = features.to_vector();

        let mut best_dist = f64::INFINITY;
        let mut worst_dist = 0.0_f64;
        let mut best_solver = String::new();
        let mut solver_dists: Vec<(String, f64)> = Vec::new();

        for solver_id in solver_ids.iter() {
            let input_clone = VRPSolverInput {
                locations: input.locations.clone(),
                num_vehicles: input.num_vehicles,
                vehicle_capacity: input.vehicle_capacity,
                objective: input.objective.clone(),
                matrix: input.matrix.clone(),
                service_time_secs: input.service_time_secs,
                use_time_windows: input.use_time_windows,
                window_open: input.window_open,
                window_close: input.window_close,
                hyperparams: input.hyperparams.clone(),
            };

            let result = rt.block_on(async { solve_with(solver_id, &input_clone).await });

            match result {
                Ok(output) => {
                    let dist: f64 = output.total_distance_km.parse().unwrap_or(f64::MAX);
                    solver_dists.push((solver_id.clone(), dist));
                    if dist < best_dist {
                        best_dist = dist;
                        best_solver = solver_id.clone();
                    }
                    if dist > worst_dist && dist < f64::MAX / 2.0 {
                        worst_dist = dist;
                    }
                }
                Err(e) => {
                    eprintln!("  Solver {} failed for instance {}: {}", solver_id, i, e);
                }
            }
        }

        // Improved gap metric: relative solver gap
        let gap = if best_dist > 0.0 && best_dist < f64::MAX / 2.0 {
            let relative_gap = (worst_dist - best_dist) / best_dist;
            (1.0 + relative_gap).ln() / 5.0 // log1p normalized
        } else {
            0.0
        };

        let n_stops_actual = input.locations.len().saturating_sub(1);
        let mst_lb = mst_lower_bound(&input.locations, input.num_vehicles);

        let record = serde_json::json!({
            "features": feature_vec,
            "best_solver": best_solver,
            "best_distance_km": best_dist,
            "worst_distance_km": worst_dist,
            "lower_bound_km": mst_lb,
            "gap_pct": gap,
            "n_stops": n_stops_actual,
            "n_vehicles": input.num_vehicles,
            "vehicle_capacity": input.vehicle_capacity,
            "objective": format!("{:?}", input.objective),
            "profile": profile_label,
            "solver_dists": solver_dists.iter().map(|(id, d)| serde_json::json!({"solver": id, "distance": d})).collect::<Vec<_>>(),
        });

        println!("{}", record);

        if i > 0 && i % 100 == 0 {
            eprintln!("  Completed {} instances...", i);
        }
    }

    eprintln!("Done. Output written to stdout as JSON lines.");
}
