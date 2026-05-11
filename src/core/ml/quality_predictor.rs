//! Route quality predictor.
//!
//! Predicts expected gap to optimal and tour length *before* solving,
//! allowing early stopping or solver spawning decisions.
//!
//! Architecture: 28 → 64 → 32 → 2  (gap %, tour length km)
//!
//! Research basis: RouteFinder encoder + gap prediction (2406.15007)

use crate::core::ml::features::InstanceFeatures;
use serde::{Deserialize, Serialize};

/// Predicted route quality metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityPrediction {
    /// Predicted gap to optimal tour length (%).
    pub predicted_gap_pct: f64,
    /// Predicted absolute tour length (km).
    pub predicted_tour_length_km: f64,
    /// Confidence score [0,1] (if calibrated).
    pub confidence: f64,
}

/// Predict quality from instance features.
///
/// Currently returns heuristic estimates.  A learned MLP can be loaded
/// from safetensors in the same pattern as `selector.rs`.
pub fn predict_quality(features: &InstanceFeatures) -> QualityPrediction {
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
    }
}
