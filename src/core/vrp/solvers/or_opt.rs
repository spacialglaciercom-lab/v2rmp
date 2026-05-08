//! Or-Opt: sweep construction + iterative relocation of chains of 1-3
//! consecutive stops to the best position across all routes.

use super::super::types::*;
use super::super::utils::{matrix_get_dist, matrix_get_time};

fn solve(matrix: &DistMatrix, locations: &[VRPSolverStop], num_vehicles: usize, balance_load: bool) -> SolveResult {
    let n = matrix.len();
    if n <= 1 {
        return SolveResult {
            routes: vec![vec![0]],
            total_distance: 0.0,
            total_time: 0.0,
        };
    }

    let mut routes = crate::core::vrp::utils::build_sweep_routes(matrix, locations, num_vehicles);

    let intermediate_count = |r: &[usize]| -> usize {
        if r.len() > 2 {
            r.len() - 2
        } else {
            0
        }
    };

    let mut improved = true;
    let max_passes = 100;
    let mut passes = 0;

    while improved && passes < max_passes {
        improved = false;
        passes += 1;

        'outer: for ri in 0..routes.len() {
            let route_a_len = routes[ri].len();
            if route_a_len < 3 {
                continue;
            }

            for k in 1..=3usize {
                for pos in 1..(route_a_len.saturating_sub(k - 1)) {
                    if pos + k > route_a_len - 1 {
                        break;
                    }
                    let chain_first = routes[ri][pos];
                    let chain_last = routes[ri][pos + k - 1];
                    let prev = routes[ri][pos - 1];
                    let next_idx = pos + k;
                    let next = if next_idx < routes[ri].len() {
                        routes[ri][next_idx]
                    } else {
                        routes[ri][0]
                    };

                    let remove_gain = matrix_get_dist(matrix, prev, chain_first)
                        + matrix_get_dist(matrix, chain_last, next)
                        - matrix_get_dist(matrix, prev, next);

                    let mut best_gain = 1e-9;
                    let mut best_rj: Option<usize> = None;
                    let mut best_ins: Option<usize> = None;
                    let mut best_rev = false;

                    for (rj, _) in routes.iter().enumerate() {
                        let route_b_len = routes[rj].len();
                        for ins in 0..route_b_len.saturating_sub(1) {
                            if rj == ri && ins >= pos.saturating_sub(1) && ins < pos + k {
                                continue;
                            }

                            let a_node = routes[rj][ins];
                            let b_node = if ins + 1 < route_b_len {
                                routes[rj][ins + 1]
                            } else {
                                routes[rj][0]
                            };

                            let cost_fwd = matrix_get_dist(matrix, a_node, chain_first)
                                + matrix_get_dist(matrix, chain_last, b_node)
                                - matrix_get_dist(matrix, a_node, b_node);
                            if remove_gain - cost_fwd > best_gain {
                                best_gain = remove_gain - cost_fwd;
                                best_rj = Some(rj);
                                best_ins = Some(ins);
                                best_rev = false;
                            }

                            if k > 1 {
                                let cost_rev = matrix_get_dist(matrix, a_node, chain_last)
                                    + matrix_get_dist(matrix, chain_first, b_node)
                                    - matrix_get_dist(matrix, a_node, b_node);
                                if remove_gain - cost_rev > best_gain {
                                    best_gain = remove_gain - cost_rev;
                                    best_rj = Some(rj);
                                    best_ins = Some(ins);
                                    best_rev = true;
                                }
                            }
                        }
                    }

                    let Some(rj_val) = best_rj else { continue };
                    let Some(ins_val) = best_ins else { continue };

                    if balance_load && rj_val != ri {
                        let new_count_a = intermediate_count(&routes[ri]).saturating_sub(k);
                        let new_count_b = intermediate_count(&routes[rj_val]).saturating_add(k);
                        let counts: Vec<usize> = routes
                            .iter()
                            .enumerate()
                            .map(|(idx, r)| {
                                if idx == ri {
                                    new_count_a
                                } else if idx == rj_val {
                                    new_count_b
                                } else {
                                    intermediate_count(r)
                                }
                            })
                            .collect();
                        let max_c = *counts.iter().max().unwrap_or(&0);
                        let min_c = *counts.iter().min().unwrap_or(&0);
                        if max_c - min_c > 1 {
                            continue;
                        }
                    }

                    improved = true;
                    let chain: Vec<usize> = routes[ri][pos..pos + k].to_vec();
                    let insert_chain: Vec<usize> = if best_rev {
                        chain.iter().copied().rev().collect()
                    } else {
                        chain.clone()
                    };

                    if rj_val == ri {
                        let mut r = routes[ri].clone();
                        r.splice(pos..pos + k, std::iter::empty());
                        let adj_ins = if ins_val >= pos { ins_val - k } else { ins_val };
                        let insert_at = adj_ins + 1;
                        let mut result = r[..insert_at].to_vec();
                        result.extend_from_slice(&insert_chain);
                        result.extend_from_slice(&r[insert_at..]);
                        routes[ri] = result;
                    } else {
                        let mut r_a = routes[ri].clone();
                        r_a.splice(pos..pos + k, std::iter::empty());
                        routes[ri] = r_a;
                        let r_b = routes[rj_val].clone();
                        let insert_at = ins_val + 1;
                        let mut result = r_b[..insert_at].to_vec();
                        result.extend_from_slice(&insert_chain);
                        result.extend_from_slice(&r_b[insert_at..]);
                        routes[rj_val] = result;
                    }

                    break 'outer;
                }
            }
        }
    }

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
            total_time += matrix_get_time(matrix, r[i], r[i + 1]);
        }
    }

    SolveResult {
        routes: final_routes,
        total_distance,
        total_time,
    }
}

pub struct OrOptSolver;

#[async_trait::async_trait]
impl VRPSolver for OrOptSolver {
    fn id(&self) -> &str {
        "or_opt"
    }
    fn label(&self) -> &str {
        "Or-Opt (local search)"
    }
    fn requires_matrix(&self) -> bool {
        true
    }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        let matrix = input
            .matrix
            .as_ref()
            .ok_or("Or-Opt solver requires a distance matrix")?;
        let balance_load = input.objective == VrpObjective::BalanceLoad;
        let result = solve(matrix, &input.locations, input.num_vehicles, balance_load);
        Ok(result.into_output(input))
    }
    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(OrOptSolver)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::vrp::test_utils::{make_input, make_stop, build_haversine_matrix};
    use super::*;

    #[tokio::test]
    async fn test_or_opt_single_depot() {
        let stops = vec![make_stop(0.0, 0.0, "depot")];
        let input = make_input(stops, 1);
        let solver = OrOptSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_none());
    }

    #[tokio::test]
    async fn test_or_opt_metadata() {
        let solver = OrOptSolver;
        assert_eq!(solver.id(), "or_opt");
        assert!(solver.requires_matrix());
    }

    #[tokio::test]
    async fn test_or_opt_no_matrix_error() {
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
        };
        let solver = OrOptSolver;
        let err = solver.solve(&input).await.unwrap_err();
        assert!(err.contains("requires a distance matrix"));
    }

    #[tokio::test]
    async fn test_or_opt_all_stops_assigned() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(2.0, 0.0, "b"),
            make_stop(3.0, 0.0, "c"),
        ];
        let input = make_input(stops.clone(), 2);
        let solver = OrOptSolver;
        let output = solver.solve(&input).await.unwrap();
        for s in &stops[1..] {
            assert!(output.stops.iter().any(|o| o.label == s.label));
        }
    }

    #[tokio::test]
    async fn test_or_opt_balance_load() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
            make_stop(-1.0, 0.0, "c"),
            make_stop(0.0, -1.0, "d"),
            make_stop(1.0, 1.0, "e"),
        ];
        let matrix = build_haversine_matrix(&stops, 40.0);
        let input = VRPSolverInput {
            locations: stops,
            num_vehicles: 2,
            vehicle_capacity: 100.0,
            objective: VrpObjective::BalanceLoad,
            matrix: Some(matrix),
            service_time_secs: None,
            use_time_windows: false,
            window_open: None,
            window_close: None,
        };
        let solver = OrOptSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_some());
    }
}
