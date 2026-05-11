//! 2-Opt VRP: sweep construction + per-route 2-opt improvement.

use super::super::types::*;
use super::super::utils::{matrix_get_dist, matrix_get_time};

fn solve(matrix: &DistMatrix, locations: &[VRPSolverStop], num_vehicles: usize) -> SolveResult {
    let n = matrix.len();
    if n <= 1 {
        return SolveResult {
            routes: vec![vec![0]],
            total_distance: 0.0,
            total_time: 0.0,
        };
    }

    let mut routes = crate::core::vrp::utils::build_sweep_routes(matrix, locations, num_vehicles);

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
        Ok(result.into_output(input))
    }
    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(TwoOptSolver)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::vrp::test_utils::{make_input, make_stop};
    use super::*;

    #[tokio::test]
    async fn test_two_opt_single_depot() {
        let stops = vec![make_stop(0.0, 0.0, "depot")];
        let input = make_input(stops, 1);
        let solver = TwoOptSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_some());
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
