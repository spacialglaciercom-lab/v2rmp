#![allow(dead_code)]
#![allow(dead_code)]
#![allow(dead_code)]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_macros,
    clippy::all
)]
#![allow(dead_code)]
#![allow(dead_code)]
#![allow(dead_code)]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_macros,
    clippy::all
)]
//! Dynamic solver registry: built-ins first, then dynamically registered solvers.

use super::solvers::clarke_wright::ClarkeWrightSolver;
use super::solvers::default::DefaultSolver;
use super::solvers::or_opt::OrOptSolver;
use super::solvers::sweep::SweepSolver;
use super::solvers::two_opt::TwoOptSolver;
use super::types::{VRPSolver, VRPSolverInput, VRPSolverOutput};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

static REGISTRY: OnceLock<RwLock<SolverRegistryInner>> = OnceLock::new();

fn registry() -> &'static RwLock<SolverRegistryInner> {
    REGISTRY.get_or_init(|| RwLock::new(SolverRegistryInner::new()))
}

struct SolverRegistryInner {
    solvers: HashMap<String, Arc<dyn VRPSolver>>,
    builtin_ids: Vec<String>,
    dynamic_order: Vec<String>,
}

impl SolverRegistryInner {
    fn new() -> Self {
        let builtins: Vec<Arc<dyn VRPSolver>> = vec![
            Arc::new(ClarkeWrightSolver),
            Arc::new(SweepSolver),
            Arc::new(OrOptSolver),
            Arc::new(TwoOptSolver),
            Arc::new(DefaultSolver),
        ];
        let builtin_ids: Vec<String> = builtins.iter().map(|s| s.id().to_string()).collect();
        let mut solvers = HashMap::new();
        for s in builtins {
            let id = s.id().to_string();
            solvers.insert(id, s);
        }
        Self {
            solvers,
            builtin_ids,
            dynamic_order: Vec::new(),
        }
    }
}

/// Register a custom solver. If the id already exists, it is replaced.
pub fn register_solver(solver: Arc<dyn VRPSolver>) {
    let id = solver.id().to_string();
    let mut reg = registry().write().unwrap();
    let had = reg.solvers.contains_key(&id);
    reg.solvers.insert(id.clone(), solver);
    if !had && !reg.builtin_ids.contains(&id) {
        reg.dynamic_order.push(id);
    }
}

/// Unregister a solver by id. Built-in solver ids are ignored.
/// Returns true if a dynamic solver was removed.
pub fn unregister_solver(id: &str) -> bool {
    let mut reg = registry().write().unwrap();
    if reg.builtin_ids.contains(&id.to_string()) {
        return false;
    }
    let removed = reg.solvers.remove(id).is_some();
    if removed {
        reg.dynamic_order.retain(|x| x != id);
    }
    removed
}

/// Return the ordered list of solvers: built-ins first, then dynamically registered.
pub fn get_solver_list() -> Vec<String> {
    let reg = registry().read().unwrap();
    let mut ids: Vec<String> = reg.builtin_ids.clone();
    for id in &reg.dynamic_order {
        if reg.solvers.contains_key(id) {
            ids.push(id.clone());
        }
    }
    ids
}

/// Solve using a solver by id. Falls back to default.
pub async fn solve_with(id: &str, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
    // Clone the Arc out before .await so we don't hold RwLockReadGuard across await.
    let solver: Arc<dyn VRPSolver> = {
        let reg = registry().read().unwrap();
        match reg.solvers.get(id).or_else(|| reg.solvers.get("default")) {
            Some(s) => Arc::clone(s),
            None => return Err(format!("Solver '{id}' not found and no default available")),
        }
    };
    solver.solve(input).await
}

/// Algorithm options for UI dropdowns.
pub fn get_algorithm_options() -> Vec<(String, String)> {
    let ids = get_solver_list();
    let reg = registry().read().unwrap();
    ids.iter()
        .filter_map(|id| {
            reg.solvers
                .get(id)
                .map(|s| (s.id().to_string(), s.label().to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::super::utils::build_haversine_matrix;
    use super::*;

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

    #[test]
    fn test_builtin_solvers_registered() {
        let ids = get_solver_list();
        assert!(ids.contains(&"clarke_wright".to_string()));
        assert!(ids.contains(&"sweep".to_string()));
        assert!(ids.contains(&"two_opt".to_string()));
        assert!(ids.contains(&"or_opt".to_string()));
        assert!(ids.contains(&"default".to_string()));
    }

    #[test]
    fn test_algorithm_options() {
        let opts = get_algorithm_options();
        assert!(!opts.is_empty());
        for (id, label) in &opts {
            assert!(!id.is_empty());
            assert!(!label.is_empty());
        }
    }

    #[tokio::test]
    async fn test_solve_with_known_solver() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(2.0, 0.0, "b"),
        ];
        let input = make_input(stops, 1);
        let output = solve_with("clarke_wright", &input).await.unwrap();
        assert!(!output.stops.is_empty());
    }

    #[tokio::test]
    async fn test_solve_with_default_fallback() {
        let stops = vec![make_stop(0.0, 0.0, "depot"), make_stop(1.0, 0.0, "a")];
        let input = make_input(stops, 1);
        let output = solve_with("nonexistent_solver", &input).await.unwrap();
        assert!(!output.stops.is_empty());
    }

    #[test]
    fn test_unregister_builtin_fails() {
        assert!(!unregister_solver("clarke_wright"));
        assert!(!unregister_solver("default"));
    }

    #[test]
    fn test_register_and_unregister_dynamic() {
        struct TestSolver;
        #[async_trait::async_trait]
        impl VRPSolver for TestSolver {
            fn id(&self) -> &str {
                "test_custom"
            }
            fn label(&self) -> &str {
                "Test Custom Solver"
            }
            fn requires_matrix(&self) -> bool {
                false
            }
            async fn solve(&self, _input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
                Ok(VRPSolverOutput {
                    stops: vec![],
                    routes: None,
                    total_distance_km: "0.00".into(),
                    total_time_min: 0,
                    route_stats: None,
                    route_metrics: None,
                    unassigned: None,
                })
            }
            fn clone_box(&self) -> Box<dyn VRPSolver> {
                Box::new(TestSolver)
            }
        }

        register_solver(Arc::new(TestSolver));
        let ids = get_solver_list();
        assert!(ids.contains(&"test_custom".to_string()));

        let opts = get_algorithm_options();
        assert!(opts.iter().any(|(id, _)| id == "test_custom"));

        assert!(unregister_solver("test_custom"));
        let ids_after = get_solver_list();
        assert!(!ids_after.contains(&"test_custom".to_string()));

        assert!(!unregister_solver("test_custom"));
    }
}
