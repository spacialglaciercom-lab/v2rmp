#![cfg(feature = "ml")]
use v2rmp::core::ml::selector::predict_solver;
use v2rmp::core::ml::quality_predictor::QualityPredictor;
use v2rmp::core::ml::features::InstanceFeatures;
use v2rmp::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};
use v2rmp::core::vrp::utils::build_haversine_matrix;
use v2rmp::core::vrp::solvers::neural_guided::MoveScorer;
use std::path::Path;

fn make_stop(lat: f64, lon: f64, label: &str, demand: Option<f64>) -> VRPSolverStop {
    VRPSolverStop {
        lat,
        lon,
        label: label.into(),
        demand,
        arrival_time: None,
    }
}

fn make_input(locations: Vec<VRPSolverStop>, num_vehicles: usize, capacity: f64) -> VRPSolverInput {
    let matrix = build_haversine_matrix(&locations, 40.0);
    VRPSolverInput {
        locations,
        num_vehicles,
        vehicle_capacity: capacity,
        objective: VrpObjective::MinDistance,
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    }
}

#[test]
fn test_solver_selector_behavior() {
    let model_path = Path::new("models/solver_selector.safetensors");
    if !model_path.exists() { return; }

    // Case 1: Large instance -> Should favor neural_guided
    let mut large_stops = vec![make_stop(0.0, 0.0, "depot", None)];
    for i in 0..400 {
        large_stops.push(make_stop(i as f64 * 0.001, i as f64 * 0.001, "s", None));
    }
    let large_input = make_input(large_stops, 10, 1000.0);
    let large_feats = InstanceFeatures::from_input(&large_input);
    println!("Large instance features: {:?}", large_feats.to_vector());
    let large_pred = predict_solver(&large_input, Some(model_path)).unwrap();
    println!("Large instance scores: {:?}", large_pred.all_scores);
    // According to training logic: if n_stops > 300 (0.6 * 500), favor neural_guided
    assert_eq!(large_pred.recommended, "neural_guided");

    // Case 2: Tight capacity -> Should favor clarke_wright
    let tight_stops = vec![
        make_stop(0.0, 0.0, "depot", None),
        make_stop(0.1, 0.1, "a", Some(90.0)),
        make_stop(0.2, 0.2, "b", Some(90.0)),
    ];
    let tight_input = make_input(tight_stops, 2, 100.0); // 180 demand / 200 capacity = 0.9 ratio
    let tight_pred = predict_solver(&tight_input, Some(model_path)).unwrap();
    println!("Tight capacity scores: {:?}", tight_pred.all_scores);
    assert_eq!(tight_pred.recommended, "clarke_wright");
}

#[test]
fn test_move_scorer_behavior() {
    let model_path = Path::new("models/move_scorer.safetensors");
    if !model_path.exists() { return; }

    let scorer = MoveScorer::from_file(model_path).unwrap();

    // Move A: High improvement (negative delta)
    let mut move_a = vec![0.0f32; 16];
    move_a[4] = -5.0; // delta
    move_a[15] = 1.0; // is_improvement flag

    // Move B: Bad move (positive delta)
    let mut move_b = vec![0.0f32; 16];
    move_b[4] = 5.0; // delta
    move_b[15] = 0.0;

    let mut batch = Vec::new();
    batch.extend_from_slice(&move_a);
    batch.extend_from_slice(&move_b);

    let scores = scorer.score_moves(&batch).unwrap();
    assert_eq!(scores.len(), 2);
    // Move A should have a significantly higher score than Move B
    assert!(scores[0] > scores[1], "Good move score {} should be > bad move score {}", scores[0], scores[1]);
}

#[test]
fn test_quality_predictor_behavior() {
    let model_path = Path::new("models/quality_predictor.safetensors");
    if !model_path.exists() { return; }

    let predictor = QualityPredictor::from_file(model_path).unwrap();

    // Small, simple instance
    let small_stops = vec![make_stop(0.0, 0.0, "depot", None), make_stop(0.01, 0.01, "a", None)];
    let small_input = make_input(small_stops, 1, 100.0);
    let small_feats = InstanceFeatures::from_input(&small_input);
    let small_pred = predictor.predict(&small_feats).unwrap();

    // Large, complex instance
    let mut large_stops = vec![make_stop(0.0, 0.0, "depot", None)];
    for i in 0..100 {
        large_stops.push(make_stop(i as f64 * 0.01, i as f64 * 0.01, "s", None));
    }
    let large_input = make_input(large_stops, 5, 100.0);
    let large_feats = InstanceFeatures::from_input(&large_input);
    let large_pred = predictor.predict(&large_feats).unwrap();

    // Predicted length for large instance should be much higher
    assert!(large_pred.predicted_tour_length_km > small_pred.predicted_tour_length_km);
}
