#![allow(dead_code)]
//! Machine-learning utilities for route optimization.
//!
//! Provides lightweight, deterministic predictors and scorers that can be
//! swapped for learned models later.  All functions are pure (no I/O) so
//! they work in MCP handlers, TUI callbacks, and WASM targets.

use super::haversine_m;
use super::vrp::types::{VRPSolverInput, VRPSolverOutput, VRPSolverStop, VrpObjective};

// ── Route feature extraction ───────────────────────────────────────────

/// Geometric and topological features of a VRP instance.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct RouteFeatures {
    /// Number of stops (excluding depot).
    pub num_stops: usize,
    /// Number of vehicles requested.
    pub num_vehicles: usize,
    /// Average pairwise haversine distance in km.
    pub avg_pairwise_km: f64,
    /// Standard deviation of stop latitudes (proxy for north-south spread).
    pub lat_spread: f64,
    /// Standard deviation of stop longitudes (proxy for east-west spread).
    pub lon_spread: f64,
    /// Ratio of spread in km to stop count (density proxy).
    pub density: f64,
    /// Whether capacity constraints are tight.
    pub tight_capacity: bool,
    /// Optimization objective.
    pub objective: VrpObjective,
}

impl RouteFeatures {
    /// Extract features from a VRP input (stops + depot at index 0).
    pub fn from_input(input: &VRPSolverInput) -> Self {
        let stops = &input.locations;
        let n = stops.len();
        if n <= 1 {
            return Self {
                num_stops: 0,
                num_vehicles: input.num_vehicles.max(1),
                ..Default::default()
            };
        }

        let depot = &stops[0];
        let others: Vec<&VRPSolverStop> = stops.iter().skip(1).collect();

        // Pairwise distances
        let mut dists = Vec::new();
        for (i, a) in others.iter().enumerate() {
            for b in others.iter().skip(i + 1) {
                dists.push(haversine_m(a.lat, a.lon, b.lat, b.lon) / 1000.0);
            }
        }
        let avg_pairwise_km = if !dists.is_empty() {
            dists.iter().sum::<f64>() / dists.len() as f64
        } else {
            0.0
        };

        // Spread
        let lats: Vec<f64> = others.iter().map(|s| s.lat).collect();
        let lons: Vec<f64> = others.iter().map(|s| s.lon).collect();
        let lat_spread = std_dev(&lats);
        let lon_spread = std_dev(&lons);

        // Approx density: km² spread / stop count
        let lat_km = lat_spread * 111.0; // 1° lat ≈ 111 km
        let lon_km = lon_spread * 111.0 * depot.lat.to_radians().cos();
        let area_km2 = (lat_km * lon_km).max(0.001);
        let density = others.len() as f64 / area_km2;

        // Capacity tightness
        let total_demand: f64 = others.iter().filter_map(|s| s.demand).sum();
        let tight_capacity = total_demand > input.vehicle_capacity * 0.8;

        Self {
            num_stops: others.len(),
            num_vehicles: input.num_vehicles.max(1),
            avg_pairwise_km,
            lat_spread,
            lon_spread,
            density,
            tight_capacity,
            objective: input.objective.clone(),
        }
    }
}

fn std_dev(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

// ── Solver recommendation ─────────────────────────────────────────────

/// Recommend a solver id and a confidence score [0,1] for the given instance.
///
/// Rules derived from heuristics informed by EARLI (2025) and RouteFinder (2024)
/// literature: small/dense instances prefer fast constructive heuristics,
/// large/sparse or multi-objective instances benefit from local-search metaheuristics.
pub fn predict_solver(features: &RouteFeatures) -> SolverPrediction {
    let n = features.num_stops;
    let v = features.num_vehicles;

    // Score each solver 0–1; higher = better fit
    let mut scores: Vec<(&str, f64)> = vec![
        ("default", 0.5),
        ("clarke_wright", 0.5),
        ("sweep", 0.5),
        ("two_opt", 0.5),
        ("or_opt", 0.5),
    ];

    // Adjust scores based on instance characteristics
    for (id, score) in scores.iter_mut() {
        match *id {
            "default" => {
                // Zone-cluster + NN + 2-opt: good default, especially for dense areas
                *score += 0.1;
                if features.density > 5.0 {
                    *score += 0.15; // dense clusters
                }
                if n <= 20 {
                    *score += 0.1; // small instances
                }
            }
            "clarke_wright" => {
                // Savings: excellent for capacity-constrained, medium size
                if features.tight_capacity {
                    *score += 0.2;
                }
                if (20..=200).contains(&n) {
                    *score += 0.1;
                }
                if (2..=10).contains(&v) {
                    *score += 0.1;
                }
                if features.objective == VrpObjective::MinDistance {
                    *score += 0.05;
                }
            }
            "sweep" => {
                // Sweep: geometric, good for radial layouts with many vehicles
                if v >= 5 {
                    *score += 0.2;
                }
                if features.lat_spread > 0.5 || features.lon_spread > 0.5 {
                    *score += 0.1; // wide spread
                }
                if features.objective == VrpObjective::BalanceLoad {
                    *score += 0.1;
                }
            }
            "two_opt" if (30..=300).contains(&n) => {
                // 2-Opt untangling: good for post-processing or medium routes
                if (30..=300).contains(&n) {
                    *score += 0.1;
                }
            }
            "or_opt" => {
                // Or-Opt metaheuristic: best for large instances, balance, time objectives
                if n > 100 {
                    *score += 0.15;
                }
                if features.objective == VrpObjective::BalanceLoad {
                    *score += 0.15;
                }
                if features.objective == VrpObjective::MinTime {
                    *score += 0.1;
                }
                if features.tight_capacity {
                    *score += 0.05;
                }
            }
            _ => {}
        }
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (best_id, best_score) = scores[0];
    let runner_up = scores.get(1).map(|(id, s)| (id.to_string(), *s));

    SolverPrediction {
        recommended: best_id.to_string(),
        confidence: best_score.clamp(0.0, 1.0),
        runner_up,
        all_scores: scores
            .into_iter()
            .map(|(id, s)| (id.to_string(), s.clamp(0.0, 1.0)))
            .collect(),
        features: features.clone(),
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SolverPrediction {
    pub recommended: String,
    pub confidence: f64,
    pub runner_up: Option<(String, f64)>,
    pub all_scores: Vec<(String, f64)>,
    pub features: RouteFeatures,
}

// ── Route quality scoring ─────────────────────────────────────────────

/// Score a solved route on multiple dimensions.  Each sub-score is 0–100.
/// Higher is better (more efficient, more balanced, fewer bad turns).
#[derive(Debug, Clone, Default, serde::Serialize)]
#[allow(dead_code)]
pub struct RouteQualityScore {
    pub distance_efficiency: f64,
    pub load_balance: f64,
    pub turn_quality: f64,
    pub coverage: f64,
    pub overall: f64,
}

/// Score a route output given the original input.
///
/// * `distance_efficiency` — ratio of straight-line lower bound to actual distance.
/// * `load_balance` — 100 when all routes have equal stop counts, 0 when max imbalance.
/// * `turn_quality` — penalizes U-turns and left turns relative to straights.
/// * `coverage` — what fraction of requested stops were assigned.
#[allow(dead_code)]
pub fn score_route(input: &VRPSolverInput, output: &VRPSolverOutput) -> RouteQualityScore {
    let n_stops = input.locations.len().saturating_sub(1).max(1);

    // --- Distance efficiency ---
    // Lower bound: sum of depot→stop distances (star tour)
    let depot = &input.locations[0];
    let star_lower_km: f64 = input
        .locations
        .iter()
        .skip(1)
        .map(|s| haversine_m(depot.lat, depot.lon, s.lat, s.lon) / 1000.0)
        .sum::<f64>()
        * 2.0; // out and back
    let actual_km: f64 = output.total_distance_km.parse().unwrap_or(f64::MAX);
    let distance_efficiency = if actual_km > 0.0 {
        ((star_lower_km / actual_km) * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    // --- Load balance ---
    let load_balance = if let Some(ref routes) = output.routes {
        let counts: Vec<usize> = routes.iter().map(|r| r.len().saturating_sub(2)).collect();
        if counts.is_empty() || counts.iter().sum::<usize>() == 0 {
            100.0
        } else {
            let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
            let max_dev = counts
                .iter()
                .map(|&c| (c as f64 - mean).abs())
                .fold(0.0f64, f64::max);
            let max_possible = mean.max(n_stops as f64 - mean);
            (100.0 * (1.0 - max_dev / max_possible.max(1.0))).clamp(0.0, 100.0)
        }
    } else {
        // Single route: perfectly balanced by definition
        100.0
    };

    // --- Turn quality ---
    // Count turns from stop-level routes if available, else assume straight
    let (left, right, u_turn, straight) = if let Some(ref routes) = output.routes {
        routes.iter().fold((0, 0, 0, 0), |(l, r, u, s), route| {
            if route.len() < 3 {
                return (l, r, u, s + route.len().saturating_sub(2) as u32);
            }
            let mut counts = (l, r, u, s);
            for i in 1..route.len() - 1 {
                let prev = &route[i - 1];
                let curr = &route[i];
                let next = &route[i + 1];
                let delta = bearing_delta(prev, curr, next);
                let d = delta.abs();
                if d > 150.0 {
                    counts.2 += 1; // u-turn
                } else if d > 45.0 && d <= 135.0 {
                    if delta < 0.0 {
                        counts.0 += 1; // left
                    } else {
                        counts.1 += 1; // right
                    }
                } else {
                    counts.3 += 1; // straight
                }
            }
            counts
        })
    } else {
        (0, 0, 0, n_stops.saturating_sub(1) as u32)
    };
    let total_turns = left + right + u_turn + straight;
    let turn_quality = if total_turns > 0 {
        let penalty = u_turn as f64 * 2.0 + left as f64 * 0.3 + right as f64 * 0.1;
        (100.0 - penalty * 100.0 / total_turns as f64).clamp(0.0, 100.0)
    } else {
        100.0
    };

    // --- Coverage ---
    let assigned = output.stops.len().saturating_sub(1).max(1);
    let coverage = (100.0 * assigned as f64 / n_stops as f64).clamp(0.0, 100.0);

    // --- Overall ---
    let overall =
        (distance_efficiency * 0.35 + load_balance * 0.25 + turn_quality * 0.20 + coverage * 0.20)
            .clamp(0.0, 100.0);

    RouteQualityScore {
        distance_efficiency: round2(distance_efficiency),
        load_balance: round2(load_balance),
        turn_quality: round2(turn_quality),
        coverage: round2(coverage),
        overall: round2(overall),
    }
}

#[allow(dead_code)]
fn bearing_delta(a: &VRPSolverStop, b: &VRPSolverStop, c: &VRPSolverStop) -> f64 {
    let b1 = bearing(a.lat, a.lon, b.lat, b.lon);
    let b2 = bearing(b.lat, b.lon, c.lat, c.lon);
    let d = b2 - b1;

    ((d + 180.0) % 360.0) - 180.0
}

#[allow(dead_code)]
fn bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();
    let x = dlon.cos() * lat2_r.sin();
    let y = lat1_r.cos() * lat2_r.sin() - lat1_r.sin() * lat2_r.cos() * dlon.cos();
    let br = y.atan2(x);
    (br.to_degrees() + 360.0) % 360.0
}

#[allow(dead_code)]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

// ── Route embedding / similarity (feature-vector representation) ──────

/// Produce a dense feature vector that can be used for similarity search
/// or as input to a learned model.  12 dimensions, normalized to ~[0,1].
#[allow(dead_code)]
pub fn route_feature_vector(features: &RouteFeatures) -> Vec<f32> {
    let n = features.num_stops as f64;
    vec![
        (n / 500.0).min(1.0) as f32,
        (features.num_vehicles as f64 / 20.0).min(1.0) as f32,
        (features.avg_pairwise_km / 100.0).min(1.0) as f32,
        (features.lat_spread / 5.0).min(1.0) as f32,
        (features.lon_spread / 5.0).min(1.0) as f32,
        (features.density / 50.0).min(1.0) as f32,
        if features.tight_capacity { 1.0 } else { 0.0 },
        match features.objective {
            VrpObjective::MinDistance => 0.0,
            VrpObjective::MinTime => 0.33,
            VrpObjective::BalanceLoad => 0.66,
            VrpObjective::MinVehicles => 1.0,
        } as f32,
        // Higher-order interactions
        (n / features.num_vehicles.max(1) as f64 / 50.0).min(1.0) as f32,
        (features.avg_pairwise_km * features.num_vehicles as f64 / 500.0).min(1.0) as f32,
        (features.lat_spread * features.lon_spread).min(1.0) as f32,
        (features.density * n / 1000.0).min(1.0) as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vrp::test_utils::{make_input, make_stop};

    #[test]
    fn test_features_small_instance() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
        ];
        let input = make_input(stops, 1);
        let f = RouteFeatures::from_input(&input);
        assert_eq!(f.num_stops, 2);
        assert!(f.avg_pairwise_km > 0.0);
        assert!(!f.tight_capacity);
    }

    #[test]
    fn test_predict_solver_returns_valid_id() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(2.0, 0.0, "b"),
            make_stop(3.0, 0.0, "c"),
        ];
        let input = make_input(stops, 1);
        let features = RouteFeatures::from_input(&input);
        let pred = predict_solver(&features);
        assert!(!pred.recommended.is_empty());
        assert!(pred.confidence > 0.0 && pred.confidence <= 1.0);
        assert_eq!(pred.all_scores.len(), 5);
    }

    #[test]
    fn test_route_feature_vector_dim() {
        let stops = vec![make_stop(0.0, 0.0, "depot"), make_stop(1.0, 0.0, "a")];
        let input = make_input(stops, 1);
        let features = RouteFeatures::from_input(&input);
        let vec = route_feature_vector(&features);
        assert_eq!(vec.len(), 12);
        for v in &vec {
            assert!(*v >= 0.0 && *v <= 1.0, "feature out of range: {}", v);
        }
    }

    #[test]
    fn test_score_route_basic() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
        ];
        let input = make_input(stops.clone(), 1);
        let output = VRPSolverOutput {
            stops: stops.clone(),
            routes: Some(vec![stops]),
            geometry: None,
            total_distance_km: format!("{:.2}", 0.0),
            total_time_min: 0,
            route_stats: None,
            route_metrics: None,
            unassigned: None,
        };
        let score = score_route(&input, &output);
        assert!(score.overall >= 0.0 && score.overall <= 100.0);
        assert!(score.coverage >= 99.0); // all stops assigned
    }
}
