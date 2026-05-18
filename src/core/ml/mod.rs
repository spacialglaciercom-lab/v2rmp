//! Machine Learning module for v2rmp.
//!
//! Provides learned models for:
//! - Solver selection (neural ensemble)
//! - Route quality prediction
//! - Instance-aware hyperparameter tuning
//! - Graph embeddings for road networks
//!
//! All inference is pure Rust via Candle. Training happens offline in Python.

#[cfg(feature = "ml")]
pub mod automl;
#[cfg(feature = "ml")]
pub mod features;
#[cfg(feature = "ml")]
pub mod feedback;
#[cfg(feature = "ml")]
pub mod graph_embed;
#[cfg(feature = "ml")]
pub mod node_embed;
#[cfg(feature = "ml")]
pub mod quality_predictor;
#[cfg(feature = "ml")]
pub mod selector;

#[cfg(feature = "ml")]
use candle_core::Device;

// Re-export the legacy rule-based module for backwards compatibility.
// New code should prefer `selector::predict_solver`.
// #[cfg(feature = "ml")]
// pub use crate::core::ml_legacy as legacy;

/// Returns the best available device (CUDA > Metal > CPU).
#[cfg(feature = "ml")]
pub fn best_device() -> candle_core::Result<Device> {
    let device = Device::cuda_if_available(0)?;
    if matches!(device, Device::Cpu) {
        Device::metal_if_available(0)
    } else {
        Ok(device)
    }
}
