use crate::core::neural_routing::{NeuralInferenceEngine, NeuralRouteRequest};
use crate::core::vrp::types::{VRPSolver, VRPSolverInput, VRPSolverOutput};
use async_trait::async_trait;

pub struct NeuralSolver;

#[async_trait]
impl VRPSolver for NeuralSolver {
    fn id(&self) -> &str {
        "neural"
    }

    fn label(&self) -> &str {
        "Neural ONNX (GNN/Attention)"
    }

    fn requires_matrix(&self) -> bool {
        false
    }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        // Find model path from hyperparams or fallback
        let model_path = input
            .hyperparams
            .as_ref()
            .and_then(|h| h.other.get("model_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("cvrp50_model.onnx");

        if !std::path::Path::new(model_path).exists() {
            return Err(format!(
                "ONNX model not found: {}. Please ensure it is in the project root.",
                model_path
            ));
        }

        let mut engine = NeuralInferenceEngine::new(model_path)
            .map_err(|e| format!("Failed to initialize neural engine: {}", e))?;

        // Prepare request
        let locations = input
            .locations
            .iter()
            .map(|s| [s.lat, s.lon, 0.0])
            .collect();
        let demands = input
            .locations
            .iter()
            .map(|s| s.demand.unwrap_or(1.0))
            .collect();

        let req = NeuralRouteRequest {
            locations,
            demands,
            capacity: input.vehicle_capacity,
            model_path: model_path.to_string(),
        };

        let resp = engine
            .solve(&req)
            .map_err(|e| format!("Neural solver error: {}", e))?;

        // Convert sequence to routes
        // For now, assume it's a single route or handled by the model
        // The model output usually is one giant sequence of visits.
        let mut stops = Vec::new();
        // Index 0 in sequence refers to index 0 in input.locations
        for &idx in &resp.visit_sequence {
            if idx < input.locations.len() {
                stops.push(input.locations[idx].clone());
            }
        }

        Ok(VRPSolverOutput {
            stops: stops.clone(),
            routes: Some(vec![stops]),
            total_distance_km: format!("{:.2}", resp.total_cost),
            total_time_min: resp.solve_time_ms as u32 / 60, // Rough estimate
            route_stats: None,
            route_metrics: None,
            unassigned: None,
        })
    }

    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(NeuralSolver)
    }
}
