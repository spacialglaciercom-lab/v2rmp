//! Sweep algorithm: partition stops by angle from depot into even sectors,
//! then order each sector with nearest-neighbor.

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

    let route_indices = crate::core::vrp::utils::build_sweep_routes(matrix, locations, num_vehicles);

    let mut total_distance = 0.0;
    let mut total_time = 0.0;
    for r in &route_indices {
        for k in 0..r.len() - 1 {
            total_distance += matrix_get_dist(matrix, r[k], r[k + 1]);
            total_time += matrix_get_time(matrix, r[k], r[k + 1]);
        }
    }

    SolveResult {
        routes: route_indices,
        total_distance,
        total_time,
    }
}

pub struct SweepSolver;

#[async_trait::async_trait]
impl VRPSolver for SweepSolver {
    fn id(&self) -> &str {
        "sweep"
    }
    fn label(&self) -> &str {
        "Sweep (balanced sectors)"
    }
    fn requires_matrix(&self) -> bool {
        true
    }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        let matrix = input
            .matrix
            .as_ref()
            .ok_or("Sweep solver requires a distance matrix")?;
        let result = solve(matrix, &input.locations, input.num_vehicles);
        Ok(result.into_output(input))
    }
    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(SweepSolver)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::vrp::test_utils::{make_input, make_stop};
    use super::*;

    #[tokio::test]
    async fn test_sweep_single_depot() {
        let stops = vec![make_stop(0.0, 0.0, "depot")];
        let input = make_input(stops, 1);
        let solver = SweepSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_some());
        assert_eq!(output.stops.len(), 1);
    }

    #[tokio::test]
    async fn test_sweep_metadata() {
        let solver = SweepSolver;
        assert_eq!(solver.id(), "sweep");
        assert!(solver.requires_matrix());
    }

    #[tokio::test]
    async fn test_sweep_no_matrix_error() {
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
        let solver = SweepSolver;
        let err = solver.solve(&input).await.unwrap_err();
        assert!(err.contains("requires a distance matrix"));
    }

    #[tokio::test]
    async fn test_sweep_balanced_routes() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
            make_stop(-1.0, 0.0, "c"),
            make_stop(0.0, -1.0, "d"),
        ];
        let input = make_input(stops, 2);
        let solver = SweepSolver;
        let output = solver.solve(&input).await.unwrap();
        // Should produce multiple routes
        assert!(output.routes.is_some());
        let dist: f64 = output.total_distance_km.parse().unwrap();
        assert!(dist > 0.0);
    }

    #[tokio::test]
    async fn test_sweep_all_stops_assigned() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 1.0, "a"),
            make_stop(2.0, 2.0, "b"),
            make_stop(3.0, 3.0, "c"),
        ];
        let input = make_input(stops.clone(), 2);
        let solver = SweepSolver;
        let output = solver.solve(&input).await.unwrap();
        // Every non-depot stop should appear in output
        for s in &stops[1..] {
            assert!(
                output.stops.iter().any(|o| o.label == s.label),
                "Stop {} missing from output",
                s.label
            );
        }
    }
}
