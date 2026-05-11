//! Learned solver selector using a small MLP in Candle.
//!
//! Replaces the rule-based `predict_solver` in `ml_legacy.rs`.
//!
//! Architecture: 28 → 128 → 64 → 5  (input → hidden → hidden → num_solvers)
//! Activation: ReLU hidden, Softmax output.
//!
//! Weights are loaded from a safetensors file at runtime.
//! If the model file is missing, falls back to the rule-based selector.

use crate::core::ml::features::InstanceFeatures;
use crate::core::ml_legacy::{predict_solver as rule_predict_solver, SolverPrediction};
use crate::core::vrp::types::{VRPSolverInput, VrpObjective};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor, DType};
use candle_nn::{linear, Module, VarBuilder, Linear};
use serde::{Deserialize, Serialize};
use std::path::Path;

const NUM_FEATURES: usize = 28;
const HIDDEN1: usize = 128;
const HIDDEN2: usize = 64;
const NUM_SOLVERS: usize = 5;

const SOLVER_IDS: [&str; NUM_SOLVERS] = [
    "default",
    "clarke_wright",
    "sweep",
    "two_opt",
    "or_opt",
];

/// Learned MLP solver selector.
pub struct NeuralSelector {
    lin1: Linear,
    lin2: Linear,
    lin3: Linear,
    device: Device,
}

impl NeuralSelector {
    /// Load from a safetensors file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("Failed to load safetensors from {}", path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let lin1 = linear(NUM_FEATURES, HIDDEN1, vb.pp("lin1"))?;
        let lin2 = linear(HIDDEN1, HIDDEN2, vb.pp("lin2"))?;
        let lin3 = linear(HIDDEN2, NUM_SOLVERS, vb.pp("lin3"))?;
        Ok(Self { lin1, lin2, lin3, device })
    }

    /// Predict solver probabilities from instance features.
    pub fn predict(&self,
        features: &InstanceFeatures,
    ) -> Result<NeuralPrediction> {
        let x = features.to_vector();
        let input = Tensor::from_vec(x, (1, NUM_FEATURES), &self.device)?;
        let h1 = self.lin1.forward(&input)?.relu()?;
        let h2 = self.lin2.forward(&h1)?.relu()?;
        let logits = self.lin3.forward(&h2)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_idx = indexed[0].0;
        let best_prob = indexed[0].1;
        let runner_up = indexed.get(1).map(|(i, p)| (SOLVER_IDS[*i].to_string(), *p));

        let all_scores: Vec<(String, f64)> = indexed
            .iter()
            .map(|(i, p)| (SOLVER_IDS[*i].to_string(), *p as f64))
            .collect();

        Ok(NeuralPrediction {
            recommended: SOLVER_IDS[best_idx].to_string(),
            confidence: best_prob as f64,
            runner_up,
            all_scores,
        })
    }
}

/// Prediction result from the neural selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPrediction {
    pub recommended: String,
    pub confidence: f64,
    pub runner_up: Option<(String, f32)>,
    pub all_scores: Vec<(String, f64)>,
}

/// High-level API: predict the best solver for a VRP instance.
///
/// Tries to load the neural model from `model_path`. If missing or failed,
/// falls back to the legacy rule-based selector.
pub fn predict_solver(
    input: &VRPSolverInput,
    model_path: Option<&Path>,
) -> Result<NeuralPrediction> {
    let features = InstanceFeatures::from_input(input);

    if let Some(path) = model_path {
        if path.exists() {
            match NeuralSelector::from_file(path) {
                Ok(selector) => {
                    match selector.predict(&features) {
                        Ok(pred) => return Ok(pred),
                        Err(e) => {
                            tracing::warn!("Neural selector inference failed: {}. Falling back to rule-based.", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load neural selector: {}. Falling back to rule-based.", e);
                }
            }
        }
    }

    // Fallback: rule-based
    let legacy = rule_predict_solver(&crate::core::ml_legacy::RouteFeatures::from_input(input)
    );
    let all_scores: Vec<(String, f64)> = legacy.all_scores;
    let runner_up = legacy.runner_up.map(|(id, score)| (id, score as f32));
    Ok(NeuralPrediction {
        recommended: legacy.recommended,
        confidence: legacy.confidence,
        runner_up,
        all_scores,
    })
}

/// Default model path relative to the executable.
pub fn default_model_path() -> std::path::PathBuf {
    std::env::current_exe()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("models")
        .join("solver_selector.safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vrp::test_utils::{make_input, make_stop};

    #[test]
    fn test_neural_selector_shape() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
        ];
        let input = make_input(stops, 1);
        let features = InstanceFeatures::from_input(&input);
        let vec = features.to_vector();
        assert_eq!(vec.len(), NUM_FEATURES);
    }

    #[test]
    fn test_predict_solver_fallback() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(2.0, 0.0, "b"),
        ];
        let input = make_input(stops, 1);
        let pred = predict_solver(&input, Some(Path::new("/nonexistent/model.safetensors"))
        ).unwrap();
        assert!(!pred.recommended.is_empty());
        assert!(pred.confidence > 0.0);
    }
}
