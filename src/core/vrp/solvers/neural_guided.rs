//! Neural-Guided Local Search Solver.
//!
//! Constructs routes with sweep + nearest-neighbor, then applies neural-guided
//! 2-Opt and Or-Opt improvement. Instead of evaluating all O(n²) swaps or relocations
//! exhaustively, a tiny MLP (MoveScorer) scores candidate moves and only the
//! top-k are considered at each iteration.
//!
//! Research basis: RLOR (2303.13117)

#![allow(clippy::ptr_arg, clippy::needless_range_loop)]

use super::super::types::*;
use super::super::utils::{build_sweep_routes, matrix_get_dist};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::path::{Path, PathBuf};

const MOVE_FEATURE_DIM: usize = 16;

/// A tiny MLP to score local-search moves: 16 → 32 → 16 → 1
pub struct MoveScorer {
    lin1: Linear,
    lin2: Linear,
    lin3: Linear,
    device: Device,
}

impl MoveScorer {
    /// Load from a safetensors file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let device = crate::core::ml::best_device()?;
        let tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("Failed to load safetensors from {}", path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let lin1 = linear(MOVE_FEATURE_DIM, 32, vb.pp("lin1"))?;
        let lin2 = linear(32, 16, vb.pp("lin2"))?;
        let lin3 = linear(16, 1, vb.pp("lin3"))?;

        Ok(Self {
            lin1,
            lin2,
            lin3,
            device,
        })
    }

    /// Predict scores for a batch of moves.
    pub fn score_moves(&self, features: &[f32]) -> Result<Vec<f32>> {
        let num_moves = features.len() / MOVE_FEATURE_DIM;
        if num_moves == 0 {
            return Ok(Vec::new());
        }

        let x = Tensor::from_vec(
            features.to_vec(),
            (num_moves, MOVE_FEATURE_DIM),
            &self.device,
        )?;
        let h1 = self.lin1.forward(&x)?.relu()?;
        let h2 = self.lin2.forward(&h1)?.relu()?;
        let out = self.lin3.forward(&h2)?;

        let scores = out.to_vec2::<f32>()?;
        Ok(scores.into_iter().map(|row| row[0]).collect())
    }
}

/// Compute 16-dim feature vector for a candidate 2-opt move.
fn move_features_2opt(
    route: &[usize],
    i: usize,
    j: usize,
    matrix: &DistMatrix,
    route_total_dist: f64,
) -> [f32; MOVE_FEATURE_DIM] {
    let n = route.len();
    let a = route[i - 1];
    let b = route[i];
    let c = route[j];
    let d = route[(j + 1) % n];

    let d_ab = matrix_get_dist(matrix, a, b);
    let d_cd = matrix_get_dist(matrix, c, d);
    let d_ac = matrix_get_dist(matrix, a, c);
    let d_bd = matrix_get_dist(matrix, b, d);

    let delta = (d_ac + d_bd) - (d_ab + d_cd);

    let avg_edge = if n > 1 {
        route_total_dist / (n - 1) as f64
    } else {
        1.0
    };
    let avg_edge = avg_edge.max(1e-6);

    let seg_len = (i..=j)
        .map(|k| matrix_get_dist(matrix, route[k], route[(k + 1) % n]))
        .sum::<f64>();

    let seg_stops = (j - i + 1) as f64;
    let route_len = n as f64;

    [
        d_ab as f32,
        d_cd as f32,
        d_ac as f32,
        d_bd as f32,
        delta as f32,
        (d_ab / avg_edge) as f32,
        (d_cd / avg_edge) as f32,
        (d_ac / avg_edge) as f32,
        (d_bd / avg_edge) as f32,
        seg_len as f32,
        (seg_len / route_total_dist.max(1e-6)) as f32,
        seg_stops as f32,
        (seg_stops / route_len) as f32,
        (i as f64 / route_len) as f32,
        (j as f64 / route_len) as f32,
        if delta < -1e-6 { 1.0 } else { 0.0 },
    ]
}

/// Compute 16-dim feature vector for a candidate Or-Opt move (relocation).
fn move_features_oropt(
    _matrix: &DistMatrix,
    remove_gain: f64,
    insert_cost: f64,
    chain_len: usize,
    is_rev: bool,
    route_len: usize,
) -> [f32; MOVE_FEATURE_DIM] {
    let delta = insert_cost - remove_gain;
    [
        remove_gain as f32,
        insert_cost as f32,
        delta as f32,
        chain_len as f32,
        if is_rev { 1.0 } else { 0.0 },
        (delta / (remove_gain.abs() + 1e-6)) as f32,
        (chain_len as f32 / route_len as f32),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0, // padding to 16
        if delta < -1e-6 { 1.0 } else { 0.0 },
    ]
}

/// Neural-guided 2-Opt improvement for a single route.
fn neural_guided_two_opt(
    route: &mut Vec<usize>,
    matrix: &DistMatrix,
    scorer: Option<&MoveScorer>,
    max_iter: u32,
) {
    let n = route.len();
    if n <= 3 {
        return;
    }

    let route_dist = |r: &[usize]| -> f64 {
        (0..r.len() - 1)
            .map(|k| matrix_get_dist(matrix, r[k], r[k + 1]))
            .sum()
    };

    let mut improved = true;
    let mut iter = 0;

    while improved && iter < max_iter {
        improved = false;
        iter += 1;

        let total_dist = route_dist(route);
        let mut features = Vec::new();
        let mut meta = Vec::new();

        for i in 1..n - 2 {
            for j in (i + 1)..n - 1 {
                let a = route[i - 1];
                let b = route[i];
                let c = route[j];
                let d = route[(j + 1) % n];

                let before = matrix_get_dist(matrix, a, b) + matrix_get_dist(matrix, c, d);
                let after = matrix_get_dist(matrix, a, c) + matrix_get_dist(matrix, b, d);
                let delta = after - before;

                if scorer.is_some() || delta < -1e-9 {
                    features
                        .extend_from_slice(&move_features_2opt(route, i, j, matrix, total_dist));
                    meta.push((i, j, delta));
                }
            }
        }

        if meta.is_empty() {
            break;
        }

        let mut scored: Vec<(usize, f64)> = if let Some(s) = scorer {
            match s.score_moves(&features) {
                Ok(scores) => scores
                    .into_iter()
                    .enumerate()
                    .map(|(idx, score)| (idx, score as f64))
                    .collect(),
                Err(_) => meta
                    .iter()
                    .enumerate()
                    .map(|(idx, (_, _, delta))| (idx, -delta))
                    .collect(),
            }
        } else {
            meta.iter()
                .enumerate()
                .map(|(idx, (_, _, delta))| (idx, -delta))
                .collect()
        };

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = if scorer.is_some() {
            scored.len().min(50)
        } else {
            scored.len().min(1)
        };

        for &(idx, _) in &scored[..top_k] {
            let (i, j, delta) = meta[idx];
            if delta < -1e-9 {
                route[i..=j].reverse();
                improved = true;
                break;
            }
        }
    }
}

/// Neural-guided Or-Opt improvement across routes.
fn neural_guided_or_opt(
    routes: &mut Vec<Vec<usize>>,
    matrix: &DistMatrix,
    scorer: Option<&MoveScorer>,
    max_passes: u32,
) {
    let mut improved = true;
    let mut passes = 0;

    while improved && passes < max_passes {
        improved = false;
        passes += 1;

        let mut features = Vec::new();
        let mut meta = Vec::new();

        for ri in 0..routes.len() {
            let route_a = &routes[ri];
            if route_a.len() < 4 {
                continue;
            }

            for k in 1..=3 {
                for pos in 1..(route_a.len().saturating_sub(k)) {
                    let chain_first = route_a[pos];
                    let chain_last = route_a[pos + k - 1];
                    let prev = route_a[pos - 1];
                    let next = route_a[pos + k];

                    let remove_gain = matrix_get_dist(matrix, prev, chain_first)
                        + matrix_get_dist(matrix, chain_last, next)
                        - matrix_get_dist(matrix, prev, next);

                    for rj in 0..routes.len() {
                        let route_b = &routes[rj];
                        for ins in 0..route_b.len().saturating_sub(1) {
                            if ri == rj && ins >= pos.saturating_sub(1) && ins < pos + k {
                                continue;
                            }

                            let a_node = route_b[ins];
                            let b_node = route_b[ins + 1];

                            let cost_fwd = matrix_get_dist(matrix, a_node, chain_first)
                                + matrix_get_dist(matrix, chain_last, b_node)
                                - matrix_get_dist(matrix, a_node, b_node);

                            let delta_fwd = cost_fwd - remove_gain;
                            if scorer.is_some() || delta_fwd < -1e-9 {
                                features.extend_from_slice(&move_features_oropt(
                                    matrix,
                                    remove_gain,
                                    cost_fwd,
                                    k,
                                    false,
                                    route_a.len(),
                                ));
                                meta.push((ri, rj, pos, k, ins, false, delta_fwd));
                            }
                        }
                    }
                }
            }
        }

        if meta.is_empty() {
            break;
        }

        let mut scored: Vec<(usize, f64)> = if let Some(s) = scorer {
            match s.score_moves(&features) {
                Ok(scores) => scores
                    .into_iter()
                    .enumerate()
                    .map(|(idx, score)| (idx, score as f64))
                    .collect(),
                Err(_) => meta
                    .iter()
                    .enumerate()
                    .map(|(idx, m)| (idx, -m.6))
                    .collect(),
            }
        } else {
            meta.iter()
                .enumerate()
                .map(|(idx, m)| (idx, -m.6))
                .collect()
        };

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = if scorer.is_some() {
            scored.len().min(20)
        } else {
            scored.len().min(1)
        };

        for &(idx, _) in &scored[..top_k] {
            let (ri, rj, pos, k, ins, _rev, delta) = meta[idx];
            if delta < -1e-9 {
                let chain: Vec<usize> = routes[ri][pos..pos + k].to_vec();
                let insert_chain = chain; // Simplified, not supporting rev for now

                if ri == rj {
                    let mut r = routes[ri].clone();
                    r.splice(pos..pos + k, std::iter::empty());
                    let adj_ins = if ins >= pos { ins - k } else { ins };
                    r.splice(adj_ins + 1..adj_ins + 1, insert_chain);
                    routes[ri] = r;
                } else {
                    routes[ri].splice(pos..pos + k, std::iter::empty());
                    routes[rj].splice(ins + 1..ins + 1, insert_chain);
                }
                improved = true;
                break;
            }
        }
    }
}
fn solve(
    matrix: &DistMatrix,
    locations: &[VRPSolverStop],
    num_vehicles: usize,
    hyperparams: Option<&SolverHyperparams>,
) -> SolveResult {
    let n = matrix.len();
    if n <= 1 {
        return SolveResult {
            routes: vec![vec![0]],
            total_distance: 0.0,
            total_time: 0.0,
        };
    }

    let max_iterations = hyperparams.map(|p| p.max_iterations).unwrap_or(200);

    let path = default_model_path();
    let scorer = if path.exists() {
        MoveScorer::from_file(&path).ok()
    } else {
        None
    };

    if scorer.is_some() {
        tracing::info!(
            "Neural-guided solver: MoveScorer loaded. Max iterations: {}",
            max_iterations
        );
    } else {
        tracing::info!(
            "Neural-guided solver: MoveScorer not available, falling back to exhaustive 2-opt. Max iterations: {}",
            max_iterations
        );
    }

    let mut routes = build_sweep_routes(matrix, locations, num_vehicles);

    for route in &mut routes {
        neural_guided_two_opt(route, matrix, scorer.as_ref(), max_iterations);
    }

    neural_guided_or_opt(&mut routes, matrix, scorer.as_ref(), max_iterations / 4);

    let final_routes: Vec<Vec<usize>> = routes.into_iter().filter(|r| r.len() > 2).collect();
    if final_routes.is_empty() {
        return SolveResult {
            routes: vec![vec![0, 0]],
            total_distance: 0.0,
            total_time: 0.0,
        };
    }

    let mut total_distance = 0.0;
    let mut total_time = 0.0;
    for r in &final_routes {
        for i in 0..r.len() - 1 {
            total_distance += matrix_get_dist(matrix, r[i], r[i + 1]);
            total_time += matrix
                .get(r[i])
                .and_then(|row| row.get(r[i + 1]))
                .map(|c| c.time)
                .unwrap_or(0.0);
        }
    }

    SolveResult {
        routes: final_routes,
        total_distance,
        total_time,
    }
}

pub fn default_model_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let path = parent.join("models").join("move_scorer.safetensors");
            if path.exists() {
                return path;
            }
        }
    }
    PathBuf::from("models/move_scorer.safetensors")
}

pub struct NeuralGuidedSolver;

#[async_trait::async_trait]
impl VRPSolver for NeuralGuidedSolver {
    fn id(&self) -> &str {
        "neural_guided"
    }

    fn label(&self) -> &str {
        "Neural-Guided (2-Opt + Or-Opt)"
    }

    fn requires_matrix(&self) -> bool {
        true
    }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        let matrix = input
            .matrix
            .as_ref()
            .ok_or("Neural-guided solver requires a distance matrix")?;
        let result = solve(
            matrix,
            &input.locations,
            input.num_vehicles,
            input.hyperparams.as_ref(),
        );
        Ok(result.into_output(input))
    }

    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(NeuralGuidedSolver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vrp::test_utils::{make_input, make_stop};

    #[tokio::test]
    async fn test_neural_guided_single_depot() {
        let stops = vec![make_stop(0.0, 0.0, "depot")];
        let input = make_input(stops, 1);
        let solver = NeuralGuidedSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_some());
    }

    #[tokio::test]
    async fn test_neural_guided_metadata() {
        let solver = NeuralGuidedSolver;
        assert_eq!(solver.id(), "neural_guided");
        assert!(solver.requires_matrix());
    }

    #[tokio::test]
    async fn test_neural_guided_no_matrix_error() {
        let stops = vec![make_stop(0.0, 0.0, "depot"), make_stop(1.0, 0.0, "a")];
        let input = VRPSolverInput {
            locations: stops,
            num_vehicles: 1,
            vehicle_capacity: 100.0,
            objective: VrpObjective::MinDistance,
            matrix: None,
            service_time_secs: None,
            use_time_windows: false,
            window_open: None,
            window_close: None,
            hyperparams: None,
        };
        let solver = NeuralGuidedSolver;
        let err = solver.solve(&input).await.unwrap_err();
        assert!(err.contains("requires a distance matrix"));
    }

    #[tokio::test]
    async fn test_neural_guided_crossing() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 1.0, "a"),
            make_stop(1.0, 0.0, "b"),
            make_stop(0.0, 1.0, "c"),
        ];
        let input = make_input(stops.clone(), 1);
        let solver = NeuralGuidedSolver;
        let output = solver.solve(&input).await.unwrap();
        let dist: f64 = output.total_distance_km.parse().unwrap();
        assert!(dist > 0.0);
    }
}
