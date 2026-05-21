use serde::{Deserialize, Serialize};

/// Type of turn: straight, left, right, or u-turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnType {
    Straight,
    Left,
    Right,
    UTurn,
}

/// Bounding box in WGS-84.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

/// A stop in a VRP route.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRPSolverStop {
    pub lat: f64,
    pub lon: f64,
    pub label: String,
    /// Demand at this stop. Depot is usually 0.
    pub demand: Option<f64>,
    /// Optional arrival time deadline (Unix seconds).
    pub arrival_time: Option<u64>,
}

/// Result cell in a distance matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistCell {
    pub distance: f64,
    pub time: f64,
    /// Ordered indices of nodes forming the path.
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

impl SolverHyperparams {
    /// Return sensible default hyperparameters when no model is available.
    pub fn default_fallback() -> Self {
        Self {
            max_iterations: 1000,
            temperature: 100.0,
            tabu_tenure: 10,
            cooling_rate: 0.99,
            neighbourhood_radius: 5,
            model_used: false,
            other: std::collections::HashMap::new(),
        }
    }
}

/// Input to a VRP solver.
#[derive(Debug, Clone, Default)]
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
    pub window_open: Option<u64>,
    /// Shift close epoch (Unix seconds).
    pub window_close: Option<u64>,
    /// Instance-aware hyperparameters.
    pub hyperparams: Option<SolverHyperparams>,
}

/// Result of a raw VRP solve (indices only).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolveResult {
    pub routes: Vec<Vec<usize>>,
    pub total_distance: f64,
    pub total_time: f64,
}

impl SolveResult {
    pub fn into_output(self, input: &VRPSolverInput) -> VRPSolverOutput {
        let mut stops = Vec::new();
        let mut routes = Vec::new();

        for route_indices in &self.routes {
            let mut route_stops = Vec::new();
            for &idx in route_indices {
                if idx < input.locations.len() {
                    route_stops.push(input.locations[idx].clone());
                }
            }
            routes.push(route_stops);
        }
        
        // Populate combined stops (excluding repeated depot visits if desired, but default is all)
        if !self.routes.is_empty() {
            for &idx in &self.routes[0] {
                if idx < input.locations.len() {
                    stops.push(input.locations[idx].clone());
                }
            }
        }

        VRPSolverOutput {
            stops,
            routes: Some(routes),
            geometry: None,
            total_distance_km: format!("{:.2}", self.total_distance),
            total_time_min: (self.total_time / 60.0) as u32,
            route_stats: None,
            route_metrics: None,
            unassigned: None,
        }
    }
}

/// Optimization objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum, Default)]
#[serde(rename_all = "kebab-case")]
pub enum VrpObjective {
    /// Minimize total distance across all vehicles.
    #[default]
    MinDistance,
    /// Minimize total travel time.
    MinTime,
    /// Balance load/stops between vehicles.
    BalanceLoad,
    /// Minimize number of vehicles used.
    MinVehicles,
}

/// Result of a VRP optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRPSolverOutput {
    /// Ordered sequence of stops visited.
    pub stops: Vec<VRPSolverStop>,
    /// Per-vehicle routes (optional).
    pub routes: Option<Vec<Vec<VRPSolverStop>>>,
    /// High-fidelity geometry (e.g. road-following paths).
    /// Represented as a list of routes, where each route is a list of [lon, lat] points.
    pub geometry: Option<Vec<Vec<[f64; 2]>>>,
    pub total_distance_km: String,
    pub total_time_min: u32,
    /// Solver-specific performance metrics.
    pub route_stats: Option<std::collections::HashMap<String, String>>,
    /// Multi-dimensional quality scores (0-100).
    pub route_metrics: Option<RouteMetrics>,
    /// Stops that could not be assigned due to capacity/constraints.
    pub unassigned: Option<Vec<VRPSolverStop>>,
}

/// Quality metrics for a solved route.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteMetrics {
    pub total_score: f32,
    pub distance_efficiency: f32,
    pub load_balance: f32,
    pub turn_quality: f32,
    pub coverage: f32,
}

/// Marker trait for VRP solvers.
#[async_trait::async_trait]
pub trait VRPSolver: Send + Sync {
    /// Unique solver ID (e.g. "clarke_wright").
    fn id(&self) -> &str;
    /// Human-readable label.
    fn label(&self) -> &str;
    /// Whether this solver requires a distance matrix.
    fn requires_matrix(&self) -> bool;
    /// Solve the VRP instance.
    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String>;
    /// Clone into a Box.
    fn clone_box(&self) -> Box<dyn VRPSolver>;
}
