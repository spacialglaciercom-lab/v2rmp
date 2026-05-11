//! Instance-aware AutoML hyperparameter tuner.
//!
//! Predicts solver hyperparameters from instance features using
//! Ridge regression or a small multi-output MLP.
//!
//! Research basis: Instance-Aware Parameter Configuration (2605.00572, 2026)

use crate::core::ml::features::InstanceFeatures;
use serde::{Deserialize, Serialize};

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
}

/// Predict hyperparameters for a given instance.
///
/// Currently returns default values.  A learned model can be loaded
/// from a safetensors file in the same pattern as `selector.rs`.
pub fn predict_hyperparams(_features: &InstanceFeatures) -> SolverHyperparams {
    // TODO: load Ridge/MLP weights and do inference.
    SolverHyperparams::default()
}

impl SolverHyperparams {
    /// Sensible defaults tuned on a broad synthetic instance set.
    pub fn default() -> Self {
        Self {
            max_iterations: 1000,
            temperature: 100.0,
            tabu_tenure: 7,
            cooling_rate: 0.995,
            neighbourhood_radius: 3,
        }
    }
}
