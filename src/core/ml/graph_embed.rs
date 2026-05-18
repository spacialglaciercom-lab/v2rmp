//! Graph embeddings for road networks.
//!
//! Converts OSM road networks to line-graphs and produces
//! learned edge embeddings that can be fused into VRP distance matrices.
//!
//! Research basis: GAIN (2107.07791) + RRNCO (2503.16159)

use crate::core::optimize::{RmpEdge, RmpNode};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Embedding dimension for road segments.
pub const EMBED_DIM: usize = 64;

/// Learned embedding for a road segment (edge in the original graph).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadEmbedding {
    pub edge_idx: usize,
    pub vector: Vec<f32>,
}

struct SAGEConv {
    lin_l: Linear,
    lin_r: Linear,
}

impl SAGEConv {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin_l = candle_nn::linear(in_dim, out_dim, vb.pp("lin_l"))?;
        let lin_r = candle_nn::linear_no_bias(in_dim, out_dim, vb.pp("lin_r"))?;
        Ok(Self { lin_l, lin_r })
    }

    fn forward(&self, x: &Tensor, x_neigh: &Tensor) -> Result<Tensor> {
        let out_l = self.lin_l.forward(x)?;
        let out_r = self.lin_r.forward(x_neigh)?;
        let out = (out_l.add(&out_r))?;
        Ok(out)
    }
}

fn aggregate_neighbors(x: &Tensor, adj: &[Vec<usize>]) -> Result<Tensor> {
    let (num_nodes, in_dim) = x.dims2()?;
    let x_vec = x.to_vec2::<f32>()?;
    let mut x_neigh = vec![vec![0.0f32; in_dim]; num_nodes];

    for i in 0..num_nodes {
        let neighbors = &adj[i];
        if neighbors.is_empty() {
            continue;
        }
        let mut sum = vec![0.0f32; in_dim];
        for &n in neighbors {
            for d in 0..in_dim {
                sum[d] += x_vec[n][d];
            }
        }
        let count = neighbors.len() as f32;
        for d in 0..in_dim {
            x_neigh[i][d] = sum[d] / count;
        }
    }

    Ok(Tensor::from_vec(
        x_neigh.into_iter().flatten().collect(),
        (num_nodes, in_dim),
        x.device(),
    )?)
}

struct GraphSAGE {
    conv1: SAGEConv,
    conv2: SAGEConv,
    device: Device,
}

impl GraphSAGE {
    fn from_file(path: &Path) -> Result<Self> {
        let device = crate::core::ml::best_device()?;
        let tensors = candle_core::safetensors::load(path, &device)
            .with_context(|| format!("Failed to load safetensors from {}", path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let conv1 = SAGEConv::new(10, 64, vb.pp("conv1"))?;
        let conv2 = SAGEConv::new(64, 64, vb.pp("conv2"))?;

        Ok(Self {
            conv1,
            conv2,
            device,
        })
    }

    fn forward(&self, x: &Tensor, adj: &[Vec<usize>]) -> Result<Tensor> {
        let x_neigh1 = aggregate_neighbors(x, adj)?;
        let h1 = self.conv1.forward(x, &x_neigh1)?.relu()?;

        let x_neigh2 = aggregate_neighbors(&h1, adj)?;
        let h2 = self.conv2.forward(&h1, &x_neigh2)?.relu()?;

        Ok(h2)
    }
}

pub fn default_model_path() -> std::path::PathBuf {
    std::env::current_exe()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("models")
        .join("graph_embed.safetensors")
}

fn try_embed_network(
    nodes: &[RmpNode],
    edges: &[RmpEdge],
    path: &Path,
) -> Result<Vec<RoadEmbedding>> {
    if edges.is_empty() {
        return Ok(Vec::new());
    }

    let model = GraphSAGE::from_file(path)?;

    // 1. Build adjacency
    let num_edges = edges.len();
    let mut node_to_edges = vec![Vec::new(); nodes.len()];
    for (i, edge) in edges.iter().enumerate() {
        node_to_edges[edge.from as usize].push(i);
        node_to_edges[edge.to as usize].push(i);
    }

    let mut adj = vec![Vec::new(); num_edges];
    for (i, edge) in edges.iter().enumerate() {
        for &e in &node_to_edges[edge.from as usize] {
            if e != i {
                adj[i].push(e);
            }
        }
        for &e in &node_to_edges[edge.to as usize] {
            if e != i {
                adj[i].push(e);
            }
        }
        adj[i].sort_unstable();
        adj[i].dedup();
    }

    // 2. Build input features
    let mut x_features = Vec::with_capacity(num_edges * 10);
    for edge in edges {
        x_features.push((edge.weight_m / 1000.0) as f32); // length
        x_features.push(edge.oneway as f32); // oneway flag
                                             // padding to 10 dims
        x_features.resize(x_features.len() + 8, 0.0);
    }
    let x = Tensor::from_vec(x_features, (num_edges, 10), &model.device)?;

    // 3. Forward pass
    let embeddings = model.forward(&x, &adj)?;
    let embeddings_vec = embeddings.to_vec2::<f32>()?;

    let mut result = Vec::with_capacity(num_edges);
    for (i, vector) in embeddings_vec.into_iter().enumerate() {
        result.push(RoadEmbedding {
            edge_idx: i,
            vector,
        });
    }

    Ok(result)
}

/// Embed a road network graph.
///
/// Converts the nodes and edges to a line graph, uses a loaded GraphSAGE model
/// to produce 64-dim learned representations for each road segment.
pub fn embed_network(
    nodes: &[RmpNode],
    edges: &[RmpEdge],
    model_path: Option<&Path>,
) -> Vec<RoadEmbedding> {
    let path = model_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_model_path);

    if !path.exists() {
        tracing::debug!("GraphSAGE model not found at {:?}, returning empty", path);
        return Vec::new();
    }

    match try_embed_network(nodes, edges, &path) {
        Ok(embs) => embs,
        Err(e) => {
            tracing::warn!(
                "GraphSAGE embedding failed: {}. Returning empty embeddings.",
                e
            );
            Vec::new()
        }
    }
}
