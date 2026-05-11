//! Graph embeddings for road networks.
//!
//! Converts OSM road networks to line-graphs and produces
//! learned edge embeddings that can be fused into VRP distance matrices.
//!
//! Research basis: GAIN (2107.07791) + RRNCO (2503.16159)

use crate::core::optimize::{RmpEdge, RmpNode};
use serde::{Deserialize, Serialize};

/// Embedding dimension for road segments.
pub const EMBED_DIM: usize = 64;

/// Learned embedding for a road segment (edge in the original graph).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadEmbedding {
    pub edge_idx: usize,
    pub vector: Vec<f32>,
}

/// Embed a road network graph.
///
/// Currently returns random-normal embeddings as a placeholder.
/// A real implementation would load a pre-trained GraphSAGE model
/// (PyTorch → ONNX → Candle) or train directly in Rust.
pub fn embed_network(_nodes: &[RmpNode], _edges: &[RmpEdge]) -> Vec<RoadEmbedding> {
    // TODO: load GraphSAGE/Graph Attention model and run inference.
    Vec::new()
}
