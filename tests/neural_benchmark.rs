#![cfg(feature = "ml")]
use v2rmp::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective, VRPSolver};
use v2rmp::core::vrp::utils::build_haversine_matrix;
use v2rmp::core::vrp::solvers::neural_guided::NeuralGuidedSolver;
use v2rmp::core::vrp::solvers::two_opt::TwoOptSolver;
use std::time::Instant;

fn make_stop(lat: f64, lon: f64, label: &str) -> VRPSolverStop {
    VRPSolverStop {
        lat,
        lon,
        label: label.into(),
        demand: None,
        arrival_time: None,
    }
}

#[tokio::test]
async fn benchmark_neural_vs_2opt() {
    // Generate a non-trivial problem: 50 stops in a grid
    let mut stops = vec![make_stop(0.0, 0.0, "depot")];
    for x in 0..7 {
        for y in 0..7 {
            if x == 0 && y == 0 { continue; }
            stops.push(make_stop(x as f64 * 0.1, y as f64 * 0.1, &format!("stop_{}_{}", x, y)));
        }
    }

    let matrix = build_haversine_matrix(&stops, 40.0);
    let input = VRPSolverInput {
        locations: stops.clone(),
        num_vehicles: 3,
        vehicle_capacity: 100.0,
        objective: VrpObjective::MinDistance,
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };

    println!("Benchmarking NeuralGuidedSolver vs TwoOptSolver (50 stops, 3 vehicles)");

    let solver_2opt = TwoOptSolver;
    let start = Instant::now();
    let out_2opt = solver_2opt.solve(&input).await.unwrap();
    let duration_2opt = start.elapsed();
    println!("TwoOptSolver: dist = {}, time = {:?}", out_2opt.total_distance_km, duration_2opt);

    let solver_neural = NeuralGuidedSolver;
    let start = Instant::now();
    let out_neural = solver_neural.solve(&input).await.unwrap();
    let duration_neural = start.elapsed();
    println!("NeuralGuidedSolver: dist = {}, time = {:?}", out_neural.total_distance_km, duration_neural);
    
    let dist_2opt: f64 = out_2opt.total_distance_km.parse().unwrap();
    let dist_neural: f64 = out_neural.total_distance_km.parse().unwrap();
    
    if dist_neural < dist_2opt {
        println!("NeuralGuidedSolver improved result by {:.2}%", (dist_2opt - dist_neural) / dist_2opt * 100.0);
    } else if dist_neural > dist_2opt {
        println!("NeuralGuidedSolver was {:.2}% worse than 2-Opt", (dist_neural - dist_2opt) / dist_2opt * 100.0);
    } else {
        println!("Both solvers produced the same distance.");
    }
}
