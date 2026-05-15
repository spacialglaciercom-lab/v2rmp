pub mod registry;
pub mod solvers;
pub mod types;
pub mod utils;

#[cfg(test)]
pub(crate) mod test_utils {
    use super::types::*;
    pub use super::utils::build_haversine_matrix;

    pub fn make_stop(lat: f64, lon: f64, label: &str) -> VRPSolverStop {
        VRPSolverStop {
            lat,
            lon,
            label: label.into(),
            demand: None,
            arrival_time: None,
        }
    }

    pub fn make_input(locations: Vec<VRPSolverStop>, num_vehicles: usize) -> VRPSolverInput {
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
            window_close: None, hyperparams: None,
        }
    }
}
