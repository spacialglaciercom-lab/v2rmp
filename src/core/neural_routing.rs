use anyhow::{Context, Result};
#[cfg(feature = "ort")]
use ort::session::Session;
#[cfg(feature = "ort")]
use ort::value::Value;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Request for the neural CVRP solver.
///
/// `locations` contains `[x, y, z]` per customer — the third element is
/// altitude or terrain elevation. The ONNX model was trained with the
/// projection `nn.Linear(4, dim)` over `[x, y, z, demand]`.
///
/// For flat-terrain use, set `z = 0.0` for every location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralRouteRequest {
    /// Customer locations as `[x, y, z]` triples (lon, lat, elevation).
    pub locations: Vec<[f64; 3]>,
    /// Demand per customer (same length as `locations`).
    pub demands: Vec<f64>,
    /// Vehicle capacity.
    pub capacity: f64,
    /// Path to the `.onnx` model file.
    pub model_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralRouteResponse {
    pub visit_sequence: Vec<usize>,
    pub total_cost: f64,
    pub solve_time_ms: u64,
}

#[cfg(feature = "ort")]
pub struct NeuralInferenceEngine {
    session: Session,
}

#[cfg(feature = "ort")]
impl NeuralInferenceEngine {
    /// Initialize the engine by loading an ONNX model.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        Ok(Self { session })
    }

    /// Solve a VRP instance using the loaded neural model.
    pub fn solve(&mut self, req: &NeuralRouteRequest) -> Result<NeuralRouteResponse> {
        let start_time = std::time::Instant::now();

        let n = req.locations.len();

        // 1. Prepare raw vectors for inputs
        // locs: [1, n, 3] — [x, y, z] per customer
        let mut locs_data = Vec::with_capacity(n * 3);
        for loc in &req.locations {
            locs_data.push(loc[0] as f32);
            locs_data.push(loc[1] as f32);
            locs_data.push(loc[2] as f32);
        }

        // demands: [1, n, 1]
        let mut demands_data = Vec::with_capacity(n);
        for &demand in &req.demands {
            demands_data.push(demand as f32);
        }

        // capacity: [1, 1]
        let capacity_data = vec![req.capacity as f32];

        // 2. Create Values using (shape, vec) which is more version-resilient
        let locs_value = Value::from_array(([1, n, 3], locs_data))?;
        let demands_value = Value::from_array(([1, n, 1], demands_data))?;
        let capacity_value = Value::from_array(([1, 1], capacity_data))?;

        // 3. Run Inference
        let inputs = ort::inputs![
            "locs" => locs_value,
            "demand" => demands_value,
            "capacity" => capacity_value,
        ];

        let outputs = self.session.run(inputs)?;

        // 4. Post-process: Extract the route from the output tensor
        let output_tensor_value = outputs
            .get("actions")
            .context("Model output 'actions' not found")?;

        let (_shape, data) = output_tensor_value.try_extract_tensor::<i64>()?;

        let visit_sequence: Vec<usize> = data
            .iter()
            .map(|&id| id as usize)
            .filter(|&id| id != 0) // Usually skip depot (0) in actions
            .collect();

        Ok(NeuralRouteResponse {
            visit_sequence,
            total_cost: 0.0,
            solve_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }
}

/// Convenience function for one-off neural routing
#[cfg(feature = "ort")]
pub fn solve_neural(req: &NeuralRouteRequest) -> Result<NeuralRouteResponse> {
    let mut engine = NeuralInferenceEngine::new(&req.model_path)?;
    engine.solve(req)
}

#[cfg(not(feature = "ort"))]
pub struct NeuralInferenceEngine;

#[cfg(not(feature = "ort"))]
impl NeuralInferenceEngine {
    pub fn new<P: AsRef<Path>>(_model_path: P) -> Result<Self> {
        anyhow::bail!("Neural inference requires the 'ort' feature, which is not available on this platform.")
    }
    pub fn solve(&mut self, _req: &NeuralRouteRequest) -> Result<NeuralRouteResponse> {
        anyhow::bail!("Neural inference requires the 'ort' feature.")
    }
}

#[cfg(not(feature = "ort"))]
pub fn solve_neural(_req: &NeuralRouteRequest) -> Result<NeuralRouteResponse> {
    anyhow::bail!("Neural inference requires the 'ort' feature.")
}
