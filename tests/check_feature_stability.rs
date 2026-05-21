// tests/check_feature_stability.rs
// Golden-vector regression test for InstanceFeatures::to_vector().
//
// If this test fails, the feature extraction logic has changed.
// Changes to `features.rs` MUST be intentional — update the golden values
// here only after verifying the change is correct.
//
// Run: cargo test --test check_feature_stability

use v2rmp::core::ml::features::InstanceFeatures;
use v2rmp::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};
use v2rmp::core::vrp::utils::build_haversine_matrix;

// Pre-computed golden vector from InstanceFeatures::to_vector()
// for a Montreal 8-stop + depot VRP instance with 3 vehicles.
// Computed via temporary compute_golden_temp.rs integration test.
const GOLDEN: [f32; 28] = [
    0.01600000, // n_stops_norm
    0.15000001, // n_vehicles_norm
    0.03417591, // avg_pairwise_km_norm
    0.00371534, // lat_spread_norm
    0.00419106, // lon_spread_norm
    0.04759511, // density_norm
    0.00336169, // area_km2_norm
    0.02165792, // depot_centroid_dist_norm
    14.0000000, // knn_avg_degree
    14.0000000, // knn_max_degree
    0.92307693, // knn_clustering
    0.06859717, // knn_diameter_norm
    0.01269179, // knn_mst_weight_norm
    0.03417591, // knn_avg_shortest_path
    0.00000000, // knn_spectral_gap
    0.00000000, // knn_assortativity
    0.03600000, // total_demand_norm
    0.02449490, // demand_std_norm
    0.00000000, // tight_capacity_flag
    0.12000000, // capacity_ratio
    0.03417591, // dist_mean_norm
    0.01504877, // dist_std_norm
    0.25445142, // dist_skewness
    0.02814293, // depot_dist_mean_norm
    1.00000000, // objective_min_distance
    0.00000000, // objective_min_time
    0.00000000, // objective_balance_load
    0.00000000, // objective_min_vehicles
];

/// Maximum allowed delta between computed and golden values.
/// Relaxed to 5e-5 to account for floating-point accumulation differences
/// across platforms and compiler optimizations.
const TOLERANCE: f32 = 5e-5;

fn build_instance() -> VRPSolverInput {
    let stops = vec![
        VRPSolverStop {
            lat: 45.5017,
            lon: -73.5673,
            label: "depot".into(),
            demand: None,
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.5088,
            lon: -73.5540,
            label: "a".into(),
            demand: Some(5.0),
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.5260,
            lon: -73.5900,
            label: "b".into(),
            demand: Some(3.0),
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.5150,
            lon: -73.5770,
            label: "c".into(),
            demand: Some(7.0),
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.4950,
            lon: -73.5800,
            label: "d".into(),
            demand: Some(2.0),
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.5340,
            lon: -73.6100,
            label: "e".into(),
            demand: Some(4.0),
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.5530,
            lon: -73.5500,
            label: "f".into(),
            demand: Some(6.0),
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.5050,
            lon: -73.5530,
            label: "g".into(),
            demand: Some(1.0),
            arrival_time: None,
        },
        VRPSolverStop {
            lat: 45.5300,
            lon: -73.5650,
            label: "h".into(),
            demand: Some(8.0),
            arrival_time: None,
        },
    ];
    let matrix = build_haversine_matrix(&stops, 40.0);
    VRPSolverInput {
        locations: stops,
        num_vehicles: 3,
        vehicle_capacity: 100.0,
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
fn golden_vector_length_is_28() {
    let input = build_instance();
    let f = InstanceFeatures::from_input(&input);
    let vec = f.to_vector();
    assert_eq!(vec.len(), 28, "Feature vector must have 28 dimensions");
}

#[test]
fn golden_vector_elementwise() {
    let input = build_instance();
    let f = InstanceFeatures::from_input(&input);
    let vec = f.to_vector();

    let mut failures = Vec::new();
    for i in 0..28 {
        let delta = (vec[i] - GOLDEN[i]).abs();
        if delta > TOLERANCE {
            failures.push(format!(
                "  [{i:2}] computed={:>14.8}  golden={:>14.8}  delta={:e}",
                vec[i], GOLDEN[i], delta
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "Feature vector changed! {} of 28 elements differ > {:.0e}:\n{}\n\
             If this change is intentional, update GOLDEN in tests/check_feature_stability.rs.",
            failures.len(),
            TOLERANCE,
            failures.join("\n")
        );
    }
}

#[test]
fn golden_vector_finite() {
    let input = build_instance();
    let f = InstanceFeatures::from_input(&input);
    let vec = f.to_vector();
    for (i, v) in vec.iter().enumerate() {
        assert!(v.is_finite(), "Feature [{i}] is not finite: {v}");
    }
}

#[test]
fn objective_onehot_is_correct() {
    let input = build_instance();
    let f = InstanceFeatures::from_input(&input);
    // MinDistance is set → objective_min_distance should be 1.0
    assert_eq!(f.objective_min_distance, 1.0);
    assert_eq!(f.objective_min_time, 0.0);
    assert_eq!(f.objective_balance_load, 0.0);
    assert_eq!(f.objective_min_vehicles, 0.0);
}
