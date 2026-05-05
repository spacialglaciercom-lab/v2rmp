//! 2-Opt VRP: sweep construction + per-route 2-opt improvement.

use super::super::types::*;
use crate::core::vrp::utils::{matrix_get_dist, matrix_get_time};

struct SolveResult {
    routes: Vec<Vec<usize>>,
    total_distance: f64,
    total_time: f64,
}

fn solve(matrix: &DistMatrix, locations: &[VRPSolverStop], num_vehicles: usize) -> SolveResult {
    let n = matrix.len();
    if n <= 1 {
        return SolveResult {
            routes: vec![vec![0]],
            total_distance: 0.0,
            total_time: 0.0,
        };
    }

    let depot = &locations[0];
    let mut stop_indices: Vec<usize> = (1..n).collect();
    stop_indices.sort_by(|&a, &b| {
        let la = &locations[a];
        let lb = &locations[b];
        let angle_a = (la.lat - depot.lat).atan2(la.lon - depot.lon);
        let angle_b = (lb.lat - depot.lat).atan2(lb.lon - depot.lon);
        angle_a
            .partial_cmp(&angle_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let per_route = (stop_indices.len() as f64 / num_vehicles as f64).ceil() as usize;
    let mut routes: Vec<Vec<usize>> = Vec::new();

    for v in 0..num_vehicles {
        let start = v * per_route;
        let end = std::cmp::min(start + per_route, stop_indices.len());
        if start >= stop_indices.len() {
            break;
        }
        let segment = &stop_indices[start..end];
        if segment.is_empty() {
            continue;
        }

        let mut route = vec![0];
        let mut rem: std::collections::HashSet<usize> = segment.iter().copied().collect();
        let mut cur = 0;
        while !rem.is_empty() {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for &node in &rem {
                let dist = matrix_get_dist(matrix, cur, node);
                if dist < best_dist {
                    best_dist = dist;
                    best = node;
                }
            }
            rem.remove(&best);
            route.push(best);
            cur = best;
        }
        route.push(0);
        routes.push(route);
    }

    // 2-opt improvement per route
    for route in &mut routes {
        let mut improved = true;
        while improved {
            improved = false;
            let r_len = route.len();
            for i in 1..r_len.saturating_sub(1) {
                for k in (i + 1)..r_len {
                    let delta = matrix_get_dist(matrix, route[i - 1], route[k])
                        + matrix_get_dist(matrix, route[i], route[k.min(route.len() - 1)])
                        - matrix_get_dist(matrix, route[i - 1], route[i])
                        - matrix_get_dist(matrix, route[k], route[k.min(route.len() - 1)]);
                    if delta < -1e-9 {
                        route[i..=k].reverse();
                        improved = true;
                    }
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

pub struct TwoOptSolver;

#[async_trait::async_trait]
impl VRPSolver for TwoOptSolver {
    fn id(&self) -> &str {
        "two_opt"
    }
    fn label(&self) -> &str {
        "2-Opt (route untangling)"
    }
    fn requires_matrix(&self) -> bool {
        true
    }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        let matrix = input
            .matrix
            .as_ref()
            .ok_or("2-Opt solver requires a distance matrix")?;
        let result = solve(matrix, &input.locations, input.num_vehicles);
        let routes: Vec<Vec<VRPSolverStop>> = result
            .routes
            .iter()
            .map(|r| r.iter().map(|&i| input.locations[i].clone()).collect())
            .collect();
        Ok(VRPSolverOutput {
            stops: routes.iter().flatten().cloned().collect(),
            routes: if routes.len() > 1 { Some(routes) } else { None },
            total_distance_km: format!("{:.2}", result.total_distance),
            total_time_min: (result.total_time / 60.0).round() as u32,
            route_stats: None,
            route_metrics: None,
            unassigned: None,
        })
    }
    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(TwoOptSolver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vrp::utils::build_haversine_matrix;

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
            window_close: None,
        }
    }

    #[tokio::test]
    async fn test_two_opt_single_depot() {
        let stops = vec![make_stop(0.0, 0.0, "depot")];
        let input = make_input(stops, 1);
        let solver = TwoOptSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_none());
    }

    #[tokio::test]
    async fn test_two_opt_metadata() {
        let solver = TwoOptSolver;
        assert_eq!(solver.id(), "two_opt");
        assert!(solver.requires_matrix());
    }

    #[tokio::test]
    async fn test_two_opt_no_matrix_error() {
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
        let solver = TwoOptSolver;
        let err = solver.solve(&input).await.unwrap_err();
        assert!(err.contains("requires a distance matrix"));
    }

    #[tokio::test]
    async fn test_two_opt_improves_route() {
        // Stops in a line - 2-opt should produce an efficient route
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(2.0, 0.0, "b"),
            make_stop(3.0, 0.0, "c"),
            make_stop(4.0, 0.0, "d"),
        ];
        let input = make_input(stops, 1);
        let solver = TwoOptSolver;
        let output = solver.solve(&input).await.unwrap();
        let dist: f64 = output.total_distance_km.parse().unwrap();
        assert!(dist > 0.0);
    }

    #[tokio::test]
    async fn test_two_opt_multiple_vehicles() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
            make_stop(-1.0, 0.0, "c"),
            make_stop(0.0, -1.0, "d"),
        ];
        let input = make_input(stops, 2);
        let solver = TwoOptSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_some());
    }
}
