//! Route quality predictor.
//!
//! Predicts expected gap to optimal and tour length *before* solving,
//! allowing early stopping or solver spawning decisions.
//!
//! Architecture: 28 → 64 → 32 → 2  (gap %, tour length km)
//!
//! Research basis: RouteFinder encoder + gap prediction (2406.15007)

use crate::core::ml::features::InstanceFeatures;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::path::Path;

const NUM_FEATURES: usize = 28;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 32;
const NUM_OUTPUTS: usize = 2;

/// Predicted route quality metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityPrediction {
    /// Predicted gap to optimal tour length (%).
    pub predicted_gap_pct: f64,
    /// Predicted absolute tour length (km).
    pub predicted_tour_length_km: f64,
    /// Confidence score [0,1] (if calibrated).
    pub confidence: f64,
    /// Whether the learned model was used (true) or heuristic fallback (false).
    pub model_used: bool,
}

/// Learned MLP quality predictor.
pub struct QualityPredictor {
    lin1: Linear,
    lin2: Linear,
    lin3: Linear,
    device: Device,
}

impl QualityPredictor {
    /// Load from a safetensors file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let device = crate::core::ml::best_device()?;
        let tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("Failed to load safetensors from {}", path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let lin1 = linear(NUM_FEATURES, HIDDEN1, vb.pp("lin1"))?;
        let lin2 = linear(HIDDEN1, HIDDEN2, vb.pp("lin2"))?;
        let lin3 = linear(HIDDEN2, NUM_OUTPUTS, vb.pp("lin3"))?;
        Ok(Self {
            lin1,
            lin2,
            lin3,
            device,
        })
    }

    /// Predict gap (%) and tour length (km) from instance features.
    pub fn predict(&self, features: &InstanceFeatures) -> Result<QualityPrediction> {
        let x = features.to_vector();
        let input = Tensor::from_vec(x, (1, NUM_FEATURES), &self.device)?;
        let h1 = self.lin1.forward(&input)?.relu()?;
        let h2 = self.lin2.forward(&h1)?.relu()?;
        let out = self.lin3.forward(&h2)?;
        let vals: Vec<f32> = out.squeeze(0)?.to_vec1()?;

        // gap is output 0, scaled to [0, 50]%
        let gap = (vals[0].clamp(0.0, 1.0) * 50.0) as f64;
        // tour length is output 1, scaled to [0, 5000] km
        let tour = (vals[1].clamp(0.0, 1.0) * 5000.0) as f64;

        Ok(QualityPrediction {
            predicted_gap_pct: gap,
            predicted_tour_length_km: tour,
            confidence: 0.8,
            model_used: true,
        })
    }
}

/// Default model path relative to the executable.
pub fn default_model_path() -> std::path::PathBuf {
    // 1. Try relative to current executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let p = exe_dir.join("models").join("quality_predictor.safetensors");
            if p.exists() {
                return p;
            }
        }
    }

    // 2. Try relative to current working directory
    let p = std::path::PathBuf::from("models/quality_predictor.safetensors");
    if p.exists() {
        return p;
    }

    // Fallback
    std::env::current_exe()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("models")
        .join("quality_predictor.safetensors")
}

/// Predict quality from instance features.
///
/// Tries to load the neural model. If missing or failed, falls back to
/// heuristic estimates.
pub fn predict_quality(features: &InstanceFeatures) -> QualityPrediction {
    let path = default_model_path();
    if path.exists() {
        match QualityPredictor::from_file(&path) {
            Ok(model) => match model.predict(features) {
                Ok(pred) => return pred,
                Err(e) => {
                    tracing::warn!(
                        "Quality predictor inference failed: {}. Falling back to heuristic.",
                        e
                    );
                }
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to load quality predictor: {}. Falling back to heuristic.",
                    e
                );
            }
        }
    }

    // Heuristic fallback: more stops + more spread → higher gap.
    let gap = (features.n_stops_norm * 20.0
        + features.density_norm * 10.0
        + features.knn_diameter_norm * 5.0)
        .min(50.0);

    // Rough tour length lower bound (star tour heuristic)
    let tour_est = features.dist_mean_norm * 100.0 * features.n_stops_norm * 500.0;

    QualityPrediction {
        predicted_gap_pct: gap,
        predicted_tour_length_km: tour_est,
        confidence: 0.5,
        model_used: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vrp::test_utils::{make_input, make_stop};

    #[test]
    fn test_predict_quality_fallback() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
        ];
        let input = make_input(stops, 1);
        let features = InstanceFeatures::from_input(&input);
        let pred = predict_quality(&features);
        assert!(pred.predicted_gap_pct >= 0.0 && pred.predicted_gap_pct <= 50.0);
        assert!(pred.predicted_tour_length_km >= 0.0);
    }
}
