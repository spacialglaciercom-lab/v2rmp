//! Default fallback solver: zone-clustering + nearest-neighbor + 2-opt improvement.

use super::super::types::*;
use super::super::utils::{
    cluster_by_zones, matrix_get_dist, matrix_get_time, nearest_neighbor_route,
    order_clusters_by_start, two_opt_improve,
};

pub struct DefaultSolver;

#[async_trait::async_trait]
impl VRPSolver for DefaultSolver {
    fn id(&self) -> &str {
        "default"
    }
    fn label(&self) -> &str {
        "Default (Zone-cluster + 2-Opt)"
    }
    fn requires_matrix(&self) -> bool {
        true
    }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        let matrix = input
            .matrix
            .as_ref()
            .ok_or("Default solver requires a distance matrix")?;
        let n = input.locations.len();

        let num_zones = ((n as f64).sqrt().ceil() as usize).clamp(2, 6);
        let clusters = cluster_by_zones(&input.locations, num_zones);
        let ordered_clusters = order_clusters_by_start(&clusters, &input.locations, 0);

        let mut route_indices: Vec<usize> = Vec::new();
        for cluster in &ordered_clusters {
            let start_in_cluster = cluster.iter().position(|&i| i == 0);
            let start_idx = start_in_cluster.unwrap_or(0);
            let segment = nearest_neighbor_route(matrix, cluster, start_idx);
            route_indices.extend(segment);
        }

        let mut with_return = route_indices;
        with_return.push(0);
        let improved = two_opt_improve(matrix, &with_return);

        let mut total_dist = 0.0;
        let mut total_time = 0.0;
        for i in 0..improved.len() - 1 {
            total_dist += matrix_get_dist(matrix, improved[i], improved[i + 1]);
            total_time += matrix_get_time(matrix, improved[i], improved[i + 1]);
        }

        let result = crate::core::vrp::types::SolveResult {
            routes: vec![improved],
            total_distance: total_dist,
            total_time,
        };
        
        Ok(result.into_output(input))
    }
    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(DefaultSolver)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::vrp::test_utils::{make_input, make_stop};
    use super::*;

    #[tokio::test]
    async fn test_default_single_depot() {
        let stops = vec![make_stop(0.0, 0.0, "depot")];
        let input = make_input(stops, 1);
        let solver = DefaultSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_none()); // default solver produces single route
    }

    #[tokio::test]
    async fn test_default_metadata() {
        let solver = DefaultSolver;
        assert_eq!(solver.id(), "default");
        assert!(solver.requires_matrix());
    }

    #[tokio::test]
    async fn test_default_no_matrix_error() {
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
        let solver = DefaultSolver;
        let err = solver.solve(&input).await.unwrap_err();
        assert!(err.contains("requires a distance matrix"));
    }

    #[tokio::test]
    async fn test_default_produces_tour() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(2.0, 0.0, "b"),
            make_stop(3.0, 0.0, "c"),
        ];
        let input = make_input(stops, 1);
        let solver = DefaultSolver;
        let output = solver.solve(&input).await.unwrap();
        // Default produces a single flat route, starts and ends at depot
        assert_eq!(output.stops.first().unwrap().label, "depot");
        assert_eq!(output.stops.last().unwrap().label, "depot");
        let dist: f64 = output.total_distance_km.parse().unwrap();
        assert!(dist > 0.0);
    }

    #[tokio::test]
    async fn test_default_visits_all_stops() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
            make_stop(-1.0, 0.0, "c"),
        ];
        let input = make_input(stops.clone(), 1);
        let solver = DefaultSolver;
        let output = solver.solve(&input).await.unwrap();
        for s in &stops {
            let count = output.stops.iter().filter(|o| o.label == s.label).count();
            assert!(count >= 1, "Stop {} missing", s.label);
        }
    }
}
