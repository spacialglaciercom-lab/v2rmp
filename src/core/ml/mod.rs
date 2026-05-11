//! Machine Learning module for v2rmp.
//!
//! Provides learned models for:
//! - Solver selection (neural ensemble)
//! - Route quality prediction
//! - Instance-aware hyperparameter tuning
//! - Graph embeddings for road networks
//!
//! All inference is pure Rust via Candle. Training happens offline in Python.

pub mod automl;
pub mod features;
pub mod graph_embed;
pub mod quality_predictor;
pub mod selector;

// Re-export the legacy rule-based module for backwards compatibility.
// New code should prefer `selector::predict_solver`.
pub use crate::core::ml_legacy as legacy;
