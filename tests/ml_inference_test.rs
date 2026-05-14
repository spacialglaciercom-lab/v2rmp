#![cfg(feature = "ml")]
use v2rmp::core::ml::selector::predict_solver;
use v2rmp::core::ml::quality_predictor::QualityPredictor;
use v2rmp::core::ml::automl::HyperparamPredictor;
use v2rmp::core::ml::features::InstanceFeatures;
use v2rmp::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};
use v2rmp::core::vrp::utils::build_haversine_matrix;
use std::path::Path;

fn make_stop(lat: f64, lon: f64, label: &str) -> VRPSolverStop {
    VRPSolverStop {
        lat,
        lon,
        label: label.into(),
        demand: None,
        arrival_time: None,
    }
}

fn make_input(locations: Vec<VRPSolverStop>, num_vehicles: usize) -> VRPSolverInput {
    let matrix = build_haversine_matrix(&locations, 40.0);
    VRPSolverInput {
        locations,
        num_vehicles,
        vehicle_capacity: 100.0,
        objective: VrpObjective::MinDistance,
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None, hyperparams: None,
    }
}

#[test]
fn test_solver_selector_loading() {
    let stops = vec![
        make_stop(0.0, 0.0, "depot"),
        make_stop(1.0, 0.0, "a"),
        make_stop(0.0, 1.0, "b"),
    ];
    let input = make_input(stops, 1);
    
    let model_path = Path::new("models/solver_selector.safetensors");
    if model_path.exists() {
        let pred = predict_solver(&input, Some(model_path)).unwrap();
        assert!(!pred.recommended.is_empty());
        assert!(pred.confidence > 0.0);
    } else {
        eprintln!("Skipping: Model file missing"); return;
    }
}

#[test]
fn test_quality_predictor_loading() {
    let model_path = Path::new("models/quality_predictor.safetensors");
    if model_path.exists() {
        let predictor = QualityPredictor::from_file(model_path).unwrap();
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
        ];
        let input = make_input(stops, 1);
        let features = InstanceFeatures::from_input(&input);
        let pred = predictor.predict(&features).unwrap();
        assert!(pred.model_used);
        assert!(pred.predicted_gap_pct >= 0.0);
    }
}

#[test]
fn test_automl_predictor_loading() {
    let model_path = Path::new("models/automl.safetensors");
    if model_path.exists() {
        let predictor = HyperparamPredictor::from_file(model_path).unwrap();
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
        ];
        let input = make_input(stops, 1);
        let features = InstanceFeatures::from_input(&input);
        let params = predictor.predict(&features).unwrap();
        assert!(params.model_used);
        assert!(params.max_iterations >= 100);
    }
}

#[test]
fn test_graph_sage_loading() {
    use v2rmp::core::ml::graph_embed::embed_network;
    use v2rmp::core::optimize::{RmpNode, RmpEdge};

    let model_path = Path::new("models/graph_embed.safetensors");
    if model_path.exists() {
        let nodes = vec![
            RmpNode { lat: 0.0, lon: 0.0 },
            RmpNode { lat: 0.01, lon: 0.01 },
        ];
        let edges = vec![
            RmpEdge { from: 0, to: 1, weight_m: 1000.0, oneway: 0 },
        ];
        let embeddings = embed_network(&nodes, &edges, Some(model_path));
        // If it loaded and ran, we should have 1 embedding (for the 1 edge)
        assert!(!embeddings.is_empty());
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].vector.len(), 64);
    }
}

#[test]
fn test_move_scorer_loading() {
    use v2rmp::core::vrp::solvers::neural_guided::MoveScorer;

    let model_path = Path::new("models/move_scorer.safetensors");
    if model_path.exists() {
        let scorer = MoveScorer::from_file(model_path).unwrap();
        let features = vec![0.0f32; 16];
        let scores = scorer.score_moves(&features).unwrap();
        assert_eq!(scores.len(), 1);
    }
}
