//! Clarke-Wright Savings algorithm for VRP with balanced distribution.
//!
//! Depot is index 0. Merges on highest savings first but caps route size so
//! stops are spread evenly across drivers; then force-merges smallest routes
//! if needed.

use super::super::types::*;
use super::super::utils::{matrix_get_dist, matrix_get_time};

/// Internal solve result.
struct SolveResult {
    routes: Vec<Vec<usize>>,
    total_distance: f64,
    total_time: f64,
}

fn solve(matrix: &DistMatrix, num_vehicles: usize) -> SolveResult {
    let n = matrix.len();
    if n <= 1 {
        return SolveResult { routes: vec![vec![0]], total_distance: 0.0, total_time: 0.0 };
    }

    let max_intermediate = ((n - 1) as f64 / num_vehicles as f64).ceil() as usize;
    let cap = max_intermediate + 1;

    let mut savings: Vec<(usize, usize, f64)> = Vec::new();
    for i in 1..n {
        for j in (i + 1)..n {
            let s = matrix_get_dist(matrix, 0, i) + matrix_get_dist(matrix, 0, j) - matrix_get_dist(matrix, i, j);
            savings.push((i, j, s));
        }
    }
    savings.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut routes: Vec<Vec<usize>> = (1..n).map(|i| vec![0, i, 0]).collect();

    let find_route = |node: usize, routes: &[Vec<usize>]| -> Option<usize> {
        routes.iter().position(|r| r.contains(&node))
    };

    let intermediate_count = |r: &[usize]| -> usize { if r.len() > 2 { r.len() - 2 } else { 0 } };

    for &(i, j, _) in &savings {
        if routes.len() <= num_vehicles {
            break;
        }
        let ri = match find_route(i, &routes) {
            Some(idx) => idx,
            None => continue,
        };
        let rj = match find_route(j, &routes) {
            Some(idx) => idx,
            None => continue,
        };
        if ri == rj {
            continue;
        }

        let ra = &routes[ri];
        let rb = &routes[rj];
        let end_of_a = ra[ra.len() - 2];
        let start_of_b = rb[1];

        let merged = if end_of_a == i && start_of_b == j {
            let mut m = ra[..ra.len() - 1].to_vec();
            m.extend_from_slice(&rb[1..]);
            Some(m)
        } else if end_of_a == j && start_of_b == i {
            let mut m = ra[..ra.len() - 1].to_vec();
            m.extend_from_slice(&rb[1..]);
            Some(m)
        } else {
            let ra_rev: Vec<usize> = ra.iter().copied().rev().collect();
            let rb_rev: Vec<usize> = rb.iter().copied().rev().collect();
            let end_rev_a = ra_rev[ra_rev.len() - 2];
            let start_rev_b = rb_rev[1];
            if end_rev_a == i && start_rev_b == j {
                let mut m = ra_rev[..ra_rev.len() - 1].to_vec();
                m.extend_from_slice(&rb_rev[1..]);
                Some(m)
            } else if end_rev_a == j && start_rev_b == i {
                let mut m = ra_rev[..ra_rev.len() - 1].to_vec();
                m.extend_from_slice(&rb_rev[1..]);
                Some(m)
            } else {
                None
            }
        };

        let merged = match merged {
            Some(m) => m,
            None => continue,
        };

        if intermediate_count(&merged) > cap {
            continue;
        }

        let mut new_routes: Vec<Vec<usize>> = routes
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != ri && *idx != rj)
            .map(|(_, r)| r.clone())
            .collect();
        new_routes.push(merged);
        routes = new_routes;
    }

    // Force-merge remaining excess routes
    while routes.len() > num_vehicles {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_merged: Option<Vec<usize>> = None;
        let mut best_cost = f64::INFINITY;

        for i in 0..routes.len() {
            for j in (i + 1)..routes.len() {
                let ra = &routes[i];
                let rb = &routes[j];

                let candidates: Vec<Vec<usize>> = vec![
                    { let mut m = ra[..ra.len() - 1].to_vec(); m.extend_from_slice(&rb[1..]); m },
                    { let mut m = ra[..ra.len() - 1].to_vec(); m.extend(rb[1..rb.len() - 1].iter().rev()); m.push(0); m },
                    { let mut m = vec![0]; m.extend(rb[1..rb.len() - 1].iter().rev()); m.extend_from_slice(&ra[1..]); m },
                    { let mut m = vec![0]; m.extend_from_slice(&rb[1..]); m.extend_from_slice(&ra[1..]); m },
                ];

                for merged in &candidates {
                    if merged.len() < 3 {
                        continue;
                    }
                    let cost: f64 = (0..merged.len() - 1)
                        .map(|k| matrix_get_dist(matrix, merged[k], merged[k + 1]))
                        .sum();
                    if cost < best_cost {
                        best_cost = cost;
                        best_i = i;
                        best_j = j;
                        best_merged = Some(merged.clone());
                    }
                }
            }
        }

        match best_merged {
            Some(merged) => {
                let mut new_routes: Vec<Vec<usize>> = routes
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != best_i && *idx != best_j)
                    .map(|(_, r)| r.clone())
                    .collect();
                new_routes.push(merged);
                routes = new_routes;
            }
            None => break,
        }
    }

    let mut total_distance = 0.0;
    let mut total_time = 0.0;
    for r in &routes {
        for k in 0..r.len() - 1 {
            total_distance += matrix_get_dist(matrix, r[k], r[k + 1]);
            total_time += matrix_get_time(matrix, r[k], r[k + 1]);
        }
    }

    SolveResult { routes, total_distance, total_time }
}

pub struct ClarkeWrightSolver;

#[async_trait::async_trait]
impl VRPSolver for ClarkeWrightSolver {
    fn id(&self) -> &str { "clarke_wright" }
    fn label(&self) -> &str { "Clarke-Wright Savings" }
    fn requires_matrix(&self) -> bool { true }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        let matrix = input.matrix.as_ref().ok_or("Clarke-Wright solver requires a distance matrix")?;
        let result = solve(matrix, input.num_vehicles);
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
    fn clone_box(&self) -> Box<dyn VRPSolver> { Box::new(ClarkeWrightSolver) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::utils::build_haversine_matrix;

    fn make_stop(lat: f64, lon: f64, label: &str) -> VRPSolverStop {
        VRPSolverStop { lat, lon, label: label.into(), demand: None, arrival_time: None }
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
    async fn test_clarke_wright_single_depot() {
        let stops = vec![make_stop(0.0, 0.0, "depot")];
        let input = make_input(stops, 1);
        let solver = ClarkeWrightSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.routes.is_none());
        assert_eq!(output.stops.len(), 1);
    }

    #[tokio::test]
    async fn test_clarke_wright_requires_matrix() {
        let solver = ClarkeWrightSolver;
        assert!(solver.requires_matrix());
    }

    #[tokio::test]
    async fn test_clarke_wright_id_and_label() {
        let solver = ClarkeWrightSolver;
        assert_eq!(solver.id(), "clarke_wright");
        assert!(!solver.label().is_empty());
    }

    #[tokio::test]
    async fn test_clarke_wright_no_matrix_error() {
        let stops = vec![make_stop(0.0, 0.0, "depot"), make_stop(1.0, 1.0, "a")];
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
        let solver = ClarkeWrightSolver;
        let err = solver.solve(&input).await.unwrap_err();
        assert!(err.contains("requires a distance matrix"));
    }

    #[tokio::test]
    async fn test_clarke_wright_two_stops_one_vehicle() {
        let stops = vec![make_stop(0.0, 0.0, "depot"), make_stop(1.0, 0.0, "a")];
        let input = make_input(stops, 1);
        let solver = ClarkeWrightSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.stops.iter().any(|s| s.label == "a"));
    }

    #[tokio::test]
    async fn test_clarke_wright_multiple_stops() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(2.0, 0.0, "b"),
            make_stop(3.0, 0.0, "c"),
            make_stop(4.0, 0.0, "d"),
        ];
        let input = make_input(stops, 2);
        let solver = ClarkeWrightSolver;
        let output = solver.solve(&input).await.unwrap();
        assert!(output.unassigned.is_none());
        assert!(output.routes.is_some());
    }

    #[tokio::test]
    async fn test_clarke_wright_positive_total_distance() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
        ];
        let input = make_input(stops, 1);
        let solver = ClarkeWrightSolver;
        let output = solver.solve(&input).await.unwrap();
        let dist: f64 = output.total_distance_km.parse().unwrap();
        assert!(dist > 0.0);
    }
}