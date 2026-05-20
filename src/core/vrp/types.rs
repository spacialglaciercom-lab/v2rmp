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
//! Core types for the VRP solver system.

use serde::{Deserialize, Serialize};

/// Internal solve result shared across all VRP solvers.
pub(crate) struct SolveResult {
    pub routes: Vec<Vec<usize>>,
    pub total_distance: f64,
    pub total_time: f64,
}

impl SolveResult {
    pub fn into_output(self, input: &VRPSolverInput) -> VRPSolverOutput {
        let routes: Vec<Vec<VRPSolverStop>> = self
            .routes
            .iter()
            .map(|r| r.iter().map(|&i| input.locations[i].clone()).collect())
            .collect();
        VRPSolverOutput {
            stops: routes.iter().flatten().cloned().collect(),
            routes: Some(routes),
            geometry: None,
            total_distance_km: format!("{:.2}", self.total_distance),
            total_time_min: (self.total_time / 60.0).round() as u32,
            route_stats: None,
            route_metrics: None,
            unassigned: None,
        }
    }
}

/// A single stop in a VRP problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRPSolverStop {
    pub lat: f64,
    pub lon: f64,
    pub label: String,
    /// Per-stop demand (units/kg).
    pub demand: Option<f64>,
    /// Estimated arrival time at this stop (Unix epoch seconds).
    pub arrival_time: Option<i64>,
}

/// Per-route statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRPSolverRouteStats {
    /// Total distance for this route in metres.
    pub distance: f64,
    /// Total driving + service duration for this route in seconds.
    pub duration: f64,
}

/// A single cell in a distance/time matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistCell {
    pub distance: f64,
    pub time: f64,
    /// Optional sequence of node indices forming the path.
    pub path: Option<Vec<u32>>,
}

/// Distance/time matrix: `matrix[from][to]`.
pub type DistMatrix = Vec<Vec<DistCell>>;

/// Hyperparameters for a VRP solver.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverHyperparams {
    /// Maximum iterations for metaheuristics (e.g. simulated annealing).
    pub max_iterations: u32,
    /// Temperature for simulated annealing.
    pub temperature: f64,
    /// Tabu tenure for tabu search.
    pub tabu_tenure: usize,
    /// Cooling rate for SA.
    pub cooling_rate: f64,
    /// Neighbourhood radius for local search.
    pub neighbourhood_radius: usize,
    /// Whether the learned model was used (true) or fallback defaults (false).
    pub model_used: bool,
    /// Flexible solver-specific parameters.
    #[serde(default)]
    pub other: std::collections::HashMap<String, serde_json::Value>,
}

/// Input to a VRP solver.
#[derive(Debug, Clone)]
pub struct VRPSolverInput {
    /// All locations. Depot is index 0.
    pub locations: Vec<VRPSolverStop>,
    pub num_vehicles: usize,
    pub vehicle_capacity: f64,
    pub objective: VrpObjective,
    /// Pre-built distance/time matrix. Required by local solvers (requires_matrix == true).
    pub matrix: Option<DistMatrix>,
    /// Per-stop service time in seconds.
    pub service_time_secs: Option<f64>,
    /// Whether shift time windows are enabled.
    pub use_time_windows: bool,
    /// Shift open epoch (Unix seconds).
    pub window_open: Option<i64>,
    /// Shift close epoch (Unix seconds).
    pub window_close: Option<i64>,
    /// Learned or manually tuned hyperparameters.
    pub hyperparams: Option<SolverHyperparams>,
}

/// VRP objective type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VrpObjective {
    MinTime,
    MinDistance,
    BalanceLoad,
    MinVehicles,
}

impl Default for VrpObjective {
    fn default() -> Self {
        VrpObjective::MinDistance
    }
}

/// Per-route metrics (physical distance, adjusted cost, elevation, turns).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRPRouteMetrics {
    pub physical_distance_m: f64,
    pub adjusted_cost_m: f64,
    pub elevation_gain_m: f64,
    pub turns: TurnCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCounts {
    pub left: u32,
    pub right: u32,
    pub u_turn: u32,
    pub straight: u32,
}

/// Output from a VRP solver.
#[derive(Debug, Clone)]
pub struct VRPSolverOutput {
    pub stops: Vec<VRPSolverStop>,
    pub routes: Option<Vec<Vec<VRPSolverStop>>>,
    /// High-fidelity road geometry for each route.
    /// Vec of routes, where each route is a Vec of [lat, lon] coordinates.
    pub geometry: Option<Vec<Vec<[f64; 2]>>>,
    pub total_distance_km: String,
    pub total_time_min: u32,
    pub route_stats: Option<Vec<VRPSolverRouteStats>>,
    pub route_metrics: Option<Vec<VRPRouteMetrics>>,
    pub unassigned: Option<Vec<String>>,
}

/// A VRP solver algorithm.
#[async_trait::async_trait]
pub trait VRPSolver: Send + Sync {
    /// Unique identifier.
    fn id(&self) -> &str;
    /// Human-readable label.
    fn label(&self) -> &str;
    /// Whether the solver requires a pre-built distance matrix.
    fn requires_matrix(&self) -> bool;
    /// Solve the VRP instance.
    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String>;
    /// Clone self into a boxed trait object (needed for registry).
    fn clone_box(&self) -> Box<dyn VRPSolver>;
}

// ─── Backend request/response models (mirrors Python vrp.py) ────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpLocation {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpStop {
    pub id: i64,
    pub location: VrpLocation,
    #[serde(default = "default_demand")]
    pub demand: Vec<i64>,
    pub time_window: Option<(i64, i64)>,
    #[serde(default)]
    pub service_duration: i64,
    pub label: Option<String>,
}

fn default_demand() -> Vec<i64> {
    vec![0]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpVehicle {
    pub id: i64,
    pub start_location: VrpLocation,
    pub end_location: Option<VrpLocation>,
    #[serde(default = "default_capacity")]
    pub capacity: Vec<i64>,
    pub time_window: Option<(i64, i64)>,
}

fn default_capacity() -> Vec<i64> {
    vec![100]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpRequest {
    pub stops: Vec<VrpStop>,
    pub vehicles: Vec<VrpVehicle>,
    #[serde(default)]
    pub use_time_windows: bool,
    #[serde(default = "default_objective")]
    pub objective: String,
    pub solver: Option<String>,
}

fn default_objective() -> String {
    "min_distance".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpRouteStep {
    pub step_type: String,
    pub id: Option<i64>,
    pub location: VrpLocation,
    pub arrival: i64,
    pub duration: i64,
    #[serde(default)]
    pub wait: i64,
    #[serde(default)]
    pub distance: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpRoute {
    pub vehicle_id: i64,
    pub steps: Vec<VrpRouteStep>,
    pub total_distance: i64,
    pub total_duration: i64,
    pub total_load: i64,
    #[serde(default)]
    pub metrics: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpResponse {
    pub routes: Vec<VrpRoute>,
    pub total_distance: i64,
    pub total_duration: i64,
    #[serde(default)]
    pub unassigned: Vec<i64>,
}

// ─── OSRM request/response models ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpOsrmLocation {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    #[serde(default)]
    pub demand: f64,
    #[serde(default = "default_service_time")]
    pub service_time_minutes: f64,
    pub time_window_start: Option<f64>,
    pub time_window_end: Option<f64>,
}

fn default_service_time() -> f64 {
    5.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpOsrmVehicle {
    pub id: String,
    pub capacity: f64,
    pub start_lat: f64,
    pub start_lon: f64,
    pub end_lat: Option<f64>,
    pub end_lon: Option<f64>,
    pub max_route_time_minutes: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpOsrmRequest {
    pub locations: Vec<VrpOsrmLocation>,
    pub vehicles: Vec<VrpOsrmVehicle>,
    #[serde(default = "default_solver")]
    pub solver: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
    #[serde(default = "default_true")]
    pub use_osrm: bool,
    #[serde(default = "default_osrm_annotations")]
    pub osrm_annotations: String,
    #[serde(default = "default_true")]
    pub cache_distance_matrix: bool,
}

fn default_solver() -> String {
    "clarke-wright".to_string()
}
fn default_max_iterations() -> u32 {
    100
}
fn default_true() -> bool {
    true
}
fn default_osrm_annotations() -> String {
    "duration,distance".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpOsrmRoute {
    pub vehicle_id: String,
    pub stops: Vec<String>,
    pub total_distance_km: f64,
    pub total_time_minutes: f64,
    pub total_demand: f64,
    pub geometry: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrpOsrmResponse {
    pub routes: Vec<VrpOsrmRoute>,
    pub total_distance_km: f64,
    pub total_time_minutes: f64,
    pub unassigned: Vec<String>,
    pub timing_ms: serde_json::Value,
    pub distance_matrix_source: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vrp_objective_default() {
        assert_eq!(VrpObjective::default(), VrpObjective::MinDistance);
    }

    #[test]
    fn test_vrp_objective_serde_roundtrip() {
        let obj = VrpObjective::MinTime;
        let json = serde_json::to_string(&obj).unwrap();
        assert_eq!(json, "\"min_time\"");
        let back: VrpObjective = serde_json::from_str(&json).unwrap();
        assert_eq!(back, obj);
    }

    #[test]
    fn test_vrp_objective_all_variants() {
        for obj in [
            VrpObjective::MinTime,
            VrpObjective::MinDistance,
            VrpObjective::BalanceLoad,
            VrpObjective::MinVehicles,
        ] {
            let json = serde_json::to_string(&obj).unwrap();
            let back: VrpObjective = serde_json::from_str(&json).unwrap();
            assert_eq!(back, obj);
        }
    }

    #[test]
    fn test_vrp_stop_serde() {
        let stop = VRPSolverStop {
            lat: 40.7128,
            lon: -74.006,
            label: "NYC".into(),
            demand: Some(10.0),
            arrival_time: Some(1700000000),
        };
        let json = serde_json::to_string(&stop).unwrap();
        let back: VRPSolverStop = serde_json::from_str(&json).unwrap();
        assert_eq!(back.lat, stop.lat);
        assert_eq!(back.label, stop.label);
        assert_eq!(back.demand, stop.demand);
    }

    #[test]
    fn test_vrp_request_default_demand() {
        let json = r#"{"id": 1, "location": {"lat": 0.0, "lon": 0.0}}"#;
        let stop: VrpStop = serde_json::from_str(json).unwrap();
        assert_eq!(stop.demand, vec![0]);
    }

    #[test]
    fn test_vrp_request_default_capacity() {
        let json = r#"{"id": 1, "start_location": {"lat": 0.0, "lon": 0.0}}"#;
        let vehicle: VrpVehicle = serde_json::from_str(json).unwrap();
        assert_eq!(vehicle.capacity, vec![100]);
    }

    #[test]
    fn test_vrp_request_default_objective() {
        let json = r#"{"stops": [], "vehicles": []}"#;
        let req: VrpRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.objective, "min_distance");
        assert!(!req.use_time_windows);
    }

    #[test]
    fn test_vrp_osrm_location_defaults() {
        let json = r#"{"id": "loc1", "lat": 40.0, "lon": -74.0}"#;
        let loc: VrpOsrmLocation = serde_json::from_str(json).unwrap();
        assert_eq!(loc.demand, 0.0);
        assert_eq!(loc.service_time_minutes, 5.0);
    }

    #[test]
    fn test_vrp_osrm_request_defaults() {
        let json = r#"{"locations": [], "vehicles": []}"#;
        let req: VrpOsrmRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.solver, "clarke-wright");
        assert_eq!(req.max_iterations, 100);
        assert!(req.use_osrm);
        assert!(req.cache_distance_matrix);
    }

    #[test]
    fn test_dist_cell() {
        let cell = DistCell {
            distance: 10.0,
            time: 600.0,
            path: None,
        };
        let json = serde_json::to_string(&cell).unwrap();
        let back: DistCell = serde_json::from_str(&json).unwrap();
        assert_eq!(back.distance, cell.distance);
        assert_eq!(back.time, cell.time);
    }

    #[test]
    fn test_vrp_response_serde() {
        let resp = VrpResponse {
            routes: vec![],
            total_distance: 100,
            total_duration: 200,
            unassigned: vec![1, 2],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: VrpResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_distance, 100);
        assert_eq!(back.unassigned, vec![1, 2]);
    }

    #[test]
    fn test_vrp_route_step_serde() {
        let step = VrpRouteStep {
            step_type: "job".into(),
            id: Some(5),
            location: VrpLocation {
                lat: 40.0,
                lon: -74.0,
            },
            arrival: 1000,
            duration: 50,
            wait: 0,
            distance: 200,
        };
        let json = serde_json::to_string(&step).unwrap();
        let back: VrpRouteStep = serde_json::from_str(&json).unwrap();
        assert_eq!(back.step_type, "job");
        assert_eq!(back.id, Some(5));
    }

    #[test]
    fn test_route_metrics() {
        let metrics = VRPRouteMetrics {
            physical_distance_m: 1500.0,
            adjusted_cost_m: 2000.0,
            elevation_gain_m: 50.0,
            turns: TurnCounts {
                left: 3,
                right: 2,
                u_turn: 0,
                straight: 10,
            },
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let back: VRPRouteMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back.turns.left, 3);
        assert_eq!(back.turns.straight, 10);
    }
}
