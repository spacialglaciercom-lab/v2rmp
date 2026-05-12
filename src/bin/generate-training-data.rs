//! Training data generator for ML models.
//!
//! Generates synthetic VRP instances, runs all 5 solvers, and records:
//! - instance features (28-dim)
//! - best solver id
//! - best achieved distance
//! - gap to a star-tour lower bound (heuristic)
//! - per-solver distances
//!
//! Usage: cargo run --bin generate-training-data --release > training_data.jsonl

use std::sync::Arc;
use tokio::runtime::Runtime;

use v2rmp::core::haversine_m;
use v2rmp::core::ml::features::InstanceFeatures;
use v2rmp::core::vrp::registry::{get_solver_list, solve_with};
use v2rmp::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective, SolverHyperparams};
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
        2 | 3 | 4 => rng.range(20, 60),
        5 | 6 | 7 => rng.range(50, 150),
        _ => rng.range(100, 250),
    };

    // Base depot near Montreal or random
    let depot_lat = 45.5 + rng.unit() * 0.5;
    let depot_lon = -73.7 + rng.unit() * 0.5;

    let mut stops: Vec<VRPSolverStop> = vec![VRPSolverStop {
        lat: depot_lat,
        lon: depot_lon,
        label: "depot".to_string(),
        demand: Some(0.0),
        arrival_time: None,
    }];

    for i in 1..=n_stops {
        let (lat, lon) = match pattern {
            0 => {
                // Random uniform around depot
                let lat = depot_lat + (rng.unit() - 0.5) * 2.0;
                let lon = depot_lon + (rng.unit() - 0.5) * 2.0;
                (lat, lon)
            }
            1 => {
                // Clustered around 2-4 centres
                let num_clusters = rng.range(2, 5) as usize;
                let cx = depot_lat + (rng.unit() - 0.5) * 1.5;
                let cy = depot_lon + (rng.unit() - 0.5) * 1.5;
                let lat = cx + (rng.unit() - 0.5) * 0.3;
                let lon = cy + (rng.unit() - 0.5) * 0.3;
                (lat, lon)
            }
            2 => {
                // Grid-like
                let grid_n = ((n_stops as f64).sqrt().ceil() as f64).max(2.0);
                let gx = (i as f64 % grid_n) / grid_n;
                let gy = (i as f64 / grid_n) / grid_n;
                let lat = depot_lat + (gx - 0.5) * 1.5 + (rng.unit() - 0.5) * 0.05;
                let lon = depot_lon + (gy - 0.5) * 1.5 + (rng.unit() - 0.5) * 0.05;
                (lat, lon)
            }
            _ => {
                // Radial
                let angle = rng.unit() * std::f64::consts::PI * 2.0;
                let radius = rng.unit() * 0.8;
                let lat = depot_lat + radius * angle.sin();
                let lon = depot_lon + radius * angle.cos();
                (lat, lon)
            }
        };

        stops.push(VRPSolverStop {
            lat,
            lon,
            label: format!("stop_{}", i),
            demand: Some(rng.range(1, 20) as f64),
            arrival_time: None,
        });
    }

    stops
}

/// Simple deterministic PRNG.
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
}

/// Lower bound: sum of depot→stop distances × 2 (out-and-back star tour).
fn star_tour_lower_bound(locations: &[VRPSolverStop]) -> f64 {
    let depot = &locations[0];
    let mut total = 0.0;
    for s in &locations[1..] {
        total += haversine_m(depot.lat, depot.lon, s.lat, s.lon) / 1000.0;
    }
    total * 2.0
}

fn make_input(stops: Vec<VRPSolverStop>, num_vehicles: usize, objective: VrpObjective) -> VRPSolverInput {
    let matrix = build_haversine_matrix(&stops, 40.0);
    VRPSolverInput {
        locations: stops,
        num_vehicles,
        vehicle_capacity: 100.0,
        objective,
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    }
}

fn objective_from_idx(idx: usize) -> VrpObjective {
    match idx % 4 {
        0 => VrpObjective::MinDistance,
        1 => VrpObjective::MinTime,
        2 => VrpObjective::BalanceLoad,
        _ => VrpObjective::MinVehicles,
    }
}

fn main() {
    let n_instances: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let rt = Runtime::new().expect("Failed to create Tokio runtime");
    let solver_ids = get_solver_list();

    eprintln!("Generating {} synthetic VRP instances and evaluating {} solvers...", n_instances, solver_ids.len());

    for i in 0..n_instances {
        let stops = generate_instance(i);
        let n_vehicles = fast_prng(i as u64 * 7).range(1, 11).max(1);
        let objective = objective_from_idx(i);
        let input = make_input(stops.clone(), n_vehicles, objective.clone());

        let features = InstanceFeatures::from_input(&input);
        let feature_vec = features.to_vector();

        let mut best_dist = f64::INFINITY;
        let mut best_solver = String::new();
        let mut solver_dists: Vec<(String, f64)> = Vec::new();

        for solver_id in &solver_ids {
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

            let result = rt.block_on(async {
                solve_with(solver_id, &input_clone).await
            });

            match result {
                Ok(output) => {
                    let dist: f64 = output.total_distance_km.parse().unwrap_or(f64::MAX);
                    solver_dists.push((solver_id.clone(), dist));
                    if dist < best_dist {
                        best_dist = dist;
                        best_solver = solver_id.clone();
                    }
                }
                Err(e) => {
                    eprintln!("  Solver {} failed for instance {}: {}", solver_id, i, e);
                }
            }
        }

        let lower = star_tour_lower_bound(&input.locations);
        let gap = if lower > 0.0 {
            ((best_dist - lower) / lower * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };

        let record = serde_json::json!({
            "features": feature_vec,
            "best_solver": best_solver,
            "best_distance_km": best_dist,
            "lower_bound_km": lower,
            "gap_pct": gap,
            "n_stops": input.locations.len().saturating_sub(1),
            "n_vehicles": input.num_vehicles,
            "objective": format!("{:?}", objective),
            "solver_dists": solver_dists.into_iter().map(|(id, d)| serde_json::json!({"solver": id, "distance": d})).collect::<Vec<_>>(),
        });

        println!("{}", record.to_string());

        if i > 0 && i % 100 == 0 {
            eprintln!("  Completed {} instances...", i);
        }
    }

    eprintln!("Done. Output written to stdout as JSON lines.");
}
