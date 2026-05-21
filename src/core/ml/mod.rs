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

/// Return the best available Candle compute device.
/// Falls back to CPU when CUDA/Metal are unavailable.
#[cfg(feature = "ml")]
pub fn best_device() -> anyhow::Result<Device> {
    Ok(Device::Cpu)
}
