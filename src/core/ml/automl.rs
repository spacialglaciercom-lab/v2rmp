//! Instance-aware AutoML hyperparameter tuner.
//!
//! Predicts solver hyperparameters from instance features using
//! Ridge regression or a small multi-output MLP.
//!
//! Research basis: Instance-Aware Parameter Configuration (2605.00572, 2026)

use crate::core::ml::features::InstanceFeatures;
pub use crate::core::vrp::types::SolverHyperparams;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::path::Path;

const NUM_FEATURES: usize = 28;
const HIDDEN1: usize = 64;
const NUM_OUTPUTS: usize = 5;

/// Learned multi-output MLP hyperparameter predictor.
pub struct HyperparamPredictor {
    lin1: Linear,
    lin2: Linear,
    device: Device,
}

impl HyperparamPredictor {
    /// Load from a safetensors file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let device = crate::core::ml::best_device()?;
        let tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("Failed to load safetensors from {}", path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let lin1 = linear(NUM_FEATURES, HIDDEN1, vb.pp("lin1"))?;
        let lin2 = linear(HIDDEN1, NUM_OUTPUTS, vb.pp("lin2"))?;
        Ok(Self { lin1, lin2, device })
    }

    /// Predict hyperparameters from instance features.
    pub fn predict(&self, features: &InstanceFeatures) -> Result<SolverHyperparams> {
        let x = features.to_vector();
        let input = Tensor::from_vec(x, (1, NUM_FEATURES), &self.device)?;
        let h1 = self.lin1.forward(&input)?.relu()?;
        let out = self.lin2.forward(&h1)?;
        let vals: Vec<f32> = out.squeeze(0)?.to_vec1()?;
        let v = vals.as_slice();

        // Clamp each output to [0, 1] and scale to parameter range
        let max_iterations = (100.0 + v[0].clamp(0.0, 1.0) as f64 * 9900.0) as u32;
        let temperature = 1.0 + v[1].clamp(0.0, 1.0) as f64 * 999.0;
        let tabu_tenure = (1.0 + v[2].clamp(0.0, 1.0) as f64 * 49.0).round() as usize;
        let cooling_rate = 0.8 + v[3].clamp(0.0, 1.0) as f64 * 0.199;
        let neighbourhood_radius = (1.0 + v[4].clamp(0.0, 1.0) as f64 * 19.0).round() as usize;

        Ok(SolverHyperparams {
            max_iterations,
            temperature,
            tabu_tenure,
            cooling_rate,
            neighbourhood_radius,
            model_used: true,
            ..Default::default()
        })
    }
}

/// Default model path relative to the executable.
pub fn default_model_path() -> std::path::PathBuf {
    // 1. Try relative to current executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let p = exe_dir.join("models").join("automl.safetensors");
            if p.exists() {
                return p;
            }
        }
    }

    // 2. Try relative to current working directory
    let p = std::path::PathBuf::from("models/automl.safetensors");
    if p.exists() {
        return p;
    }

    // Fallback to exe-relative path anyway (for creation/error reporting)
    std::env::current_exe()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("models")
        .join("automl.safetensors")
}

/// Predict hyperparameters for a given instance.
///
/// Tries to load the neural model. If missing or failed, falls back to
/// sensible defaults.
pub fn predict_hyperparams(features: &InstanceFeatures) -> SolverHyperparams {
    let path = default_model_path();
    if path.exists() {
        match HyperparamPredictor::from_file(&path) {
            Ok(model) => match model.predict(features) {
                Ok(params) => return params,
                Err(e) => {
                    tracing::warn!(
                        "Hyperparam predictor inference failed: {}. Falling back to defaults.",
                        e
                    );
                }
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to load hyperparam predictor: {}. Falling back to defaults.",
                    e
                );
            }
        }
    }
    SolverHyperparams::default_fallback()
}

impl SolverHyperparams {
    /// Sensible defaults tuned on a broad synthetic instance set.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            temperature: 100.0,
            tabu_tenure: 7,
            cooling_rate: 0.995,
            neighbourhood_radius: 3,
            model_used: false,
            other: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vrp::test_utils::{make_input, make_stop};

    #[test]
    fn test_predict_hyperparams_fallback() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
        ];
        let input = make_input(stops, 1);
        let features = InstanceFeatures::from_input(&input);
        let params = predict_hyperparams(&features);
        assert!(params.max_iterations >= 100);
        assert!(params.temperature >= 1.0);
        assert!(params.tabu_tenure >= 1);
        assert!(params.cooling_rate >= 0.8 && params.cooling_rate <= 1.0);
        assert!(params.neighbourhood_radius >= 1);
    }
}
