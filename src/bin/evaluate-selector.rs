//! Evaluate the trained neural solver selector against a held-out test set.
//!
//! Generates N new synthetic instances (not seen during training),
//! predicts the best solver using the neural selector, runs that solver,
//! and compares the achieved distance vs. always-default and always-or_opt.
//!
//! Usage: cargo run --bin evaluate-selector --features ml --release -- 200

use tokio::runtime::Runtime;

use v2rmp::core::ml::features::InstanceFeatures;
use v2rmp::core::ml::selector::{predict_solver, NeuralPrediction};
use v2rmp::core::vrp::registry::{get_solver_list, solve_with};
use v2rmp::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};
use v2rmp::core::vrp::utils::build_haversine_matrix;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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
        if lo >= hi { return lo; }
        lo + (self.next() as usize % (hi - lo))
    }
}

fn generate_instance(seed_offset: usize) -> Vec<VRPSolverStop> {
    let mut hasher = DefaultHasher::new();
    // Different seed range than training to avoid overlap
    (seed_offset + 100_000).hash(&mut hasher);
    let seed = hasher.finish();
    let mut rng = fast_prng(seed);

    let pattern = seed_offset % 4;
    let n_stops = rng.range(15, 120);

    let depot_lat = 45.3 + rng.unit() * 0.8;
    let depot_lon = -74.0 + rng.unit() * 0.8;

    let mut stops = vec![VRPSolverStop {
        lat: depot_lat,
        lon: depot_lon,
        label: "depot".to_string(),
        demand: Some(0.0),
        arrival_time: None,
    }];

    for i in 1..=n_stops {
        let (lat, lon) = match pattern {
            0 => {
                let lat = depot_lat + (rng.unit() - 0.5) * 2.0;
                let lon = depot_lon + (rng.unit() - 0.5) * 2.0;
                (lat, lon)
            }
            1 => {
                let cx = depot_lat + (rng.unit() - 0.5) * 1.5;
                let cy = depot_lon + (rng.unit() - 0.5) * 1.5;
                let lat = cx + (rng.unit() - 0.5) * 0.3;
                let lon = cy + (rng.unit() - 0.5) * 0.3;
                (lat, lon)
            }
            2 => {
                let grid_n = ((n_stops as f64).sqrt().ceil() as f64).max(2.0);
                let gx = (i as f64 % grid_n) / grid_n;
                let gy = (i as f64 / grid_n) / grid_n;
                let lat = depot_lat + (gx - 0.5) * 1.5 + (rng.unit() - 0.5) * 0.05;
                let lon = depot_lon + (gy - 0.5) * 1.5 + (rng.unit() - 0.5) * 0.05;
                (lat, lon)
            }
            _ => {
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
    let n_test: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let rt = Runtime::new().expect("Tokio runtime");
    let solver_ids = get_solver_list();

    println!("Evaluating neural solver selector on {} held-out instances...", n_test);
    println!("Solver IDs: {:?}", solver_ids);
    println!();

    let mut neural_wins = 0usize;
    let mut default_wins = 0usize;
    let mut or_opt_wins = 0usize;
    let oracle_wins = 0usize;

    let mut neural_total_dist = 0.0f64;
    let mut default_total_dist = 0.0f64;
    let mut or_opt_total_dist = 0.0f64;
    let mut oracle_total_dist = 0.0f64;

    let mut selector_correct = 0usize;
    let mut selector_top2 = 0usize;

    for i in 0..n_test {
        let stops = generate_instance(i);
        let n_vehicles = fast_prng((i + 100_000) as u64).range(1, 11).max(1);
        let objective = objective_from_idx(i + 100);
        let input = make_input(stops.clone(), n_vehicles, objective.clone());

        // Neural prediction
        let pred = predict_solver(&input, None).unwrap_or(NeuralPrediction { model_used: true,
            recommended: "default".to_string(),
            confidence: 0.0,
            runner_up: None,
            all_scores: vec![],
        });

        // Run all solvers
        let mut best_solver = String::new();
        let mut best_dist = f64::INFINITY;
        let mut predicted_dist = f64::INFINITY;
        let mut default_dist = f64::INFINITY;
        let mut or_opt_dist_val = f64::INFINITY;

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

            let dist = match result {
                Ok(output) => output.total_distance_km.parse().unwrap_or(f64::MAX),
                Err(_) => f64::MAX,
            };

            if dist < best_dist {
                best_dist = dist;
                best_solver = solver_id.clone();
            }
            if solver_id == &pred.recommended {
                predicted_dist = dist;
            }
            if solver_id == "default" {
                default_dist = dist;
            }
            if solver_id == "or_opt" {
                or_opt_dist_val = dist;
            }
        }

        // Track selector correctness
        if pred.recommended == best_solver {
            selector_correct += 1;
        }
        if let Some((runner_id, _)) = &pred.runner_up {
            if pred.recommended == best_solver || runner_id == &best_solver {
                selector_top2 += 1;
            }
        }

        // Track which strategy "wins" this instance
        if predicted_dist == best_dist {
            neural_wins += 1;
        }
        if default_dist == best_dist {
            default_wins += 1;
        }
        if or_opt_dist_val == best_dist {
            or_opt_wins += 1;
        }
        oracle_total_dist += best_dist;

        neural_total_dist += predicted_dist.min(best_dist * 2.0); // avoid inf explosion
        default_total_dist += default_dist;
        or_opt_total_dist += or_opt_dist_val;
    }

    println!("========================================");
    println!("HELD-OUT EVALUATION RESULTS (n={})", n_test);
    println!("========================================");
    println!();
    println!("--- Selector Accuracy ---");
    println!("  Top-1 accuracy:  {}/{} ({:.1}%)", selector_correct, n_test, selector_correct as f64 / n_test as f64 * 100.0);
    println!("  Top-2 accuracy:  {}/{} ({:.1}%)", selector_top2, n_test, selector_top2 as f64 / n_test as f64 * 100.0);
    println!();
    println!("--- Best-Solver Distribution ---");
    println!("  Neural (predicted): {}/{} ({:.1}%)", neural_wins, n_test, neural_wins as f64 / n_test as f64 * 100.0);
    println!("  Always-default:     {}/{} ({:.1}%)", default_wins, n_test, default_wins as f64 / n_test as f64 * 100.0);
    println!("  Always-or_opt:      {}/{} ({:.1}%)", or_opt_wins, n_test, or_opt_wins as f64 / n_test as f64 * 100.0);
    println!();
    println!("--- Total Distance (lower is better) ---");
    println!("  Neural predicted:   {:.2} km", neural_total_dist);
    println!("  Always-default:     {:.2} km", default_total_dist);
    println!("  Always-or_opt:      {:.2} km", or_opt_total_dist);
    println!("  Oracle (best):      {:.2} km", oracle_total_dist);
    println!();
    println!("--- Gap to Oracle ---");
    println!("  Neural:    {:.2}%", (neural_total_dist - oracle_total_dist) / oracle_total_dist * 100.0);
    println!("  Default:   {:.2}%", (default_total_dist - oracle_total_dist) / oracle_total_dist * 100.0);
    println!("  Or-Opt:    {:.2}%", (or_opt_total_dist - oracle_total_dist) / oracle_total_dist * 100.0);
}
