//! Road network node embeddings.
//!
//! Provides unsupervised and supervised methods for generating dense vector
//! representations of road network nodes (intersections) and edges (segments).
//!
//! # Methods
//!
//! - **node2vec**: Biased random walks + Skip-gram (Grover & Leskovec 2016)
//! - **LINE**: 1st + 2nd order proximity embedding (Tang et al. 2015)
//! - **Random Projection (FastRP)**: Sparse random projection of adjacency (Chen et al. 2019)
//! - **Spatial+Structural**: Handcrafted features (degree, betweenness, coordinates)
//!
//! # Research basis
//!
//! - node2vec: arxiv 1607.00653
//! - LINE: arxiv 1503.03578
//! - FastRP: arxiv 1908.11512
//! - GAIN: arxiv 2107.07791 (line-graph transform for road networks)
//! - SP-GEM: MDPI IJGI 14(7), 275, 2025 (spatial pattern awareness)

use crate::core::optimize::{RmpEdge, RmpNode};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

// ── Public types ──────────────────────────────────────────────────────

/// Embedding method selection.
#[derive(Debug, Clone, Copy, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbedMethod {
    /// node2vec: biased random walks + skip-gram
    Node2Vec,
    /// LINE: 1st + 2nd order proximity
    Line,
    /// FastRP: sparse random projection
    FastRp,
    /// Spatial + structural handcrafted features
    Spatial,
}

/// A single node embedding result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEmbedding {
    /// Node index in the road network.
    pub node_idx: usize,
    /// Latitude of the node.
    pub lat: f64,
    /// Longitude of the node.
    pub lon: f64,
    /// Embedding vector.
    pub vector: Vec<f32>,
}

/// A single edge embedding result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeEmbedding {
    /// Edge index in the road network.
    pub edge_idx: usize,
    /// Source node index.
    pub from: u32,
    /// Target node index.
    pub to: u32,
    /// Edge embedding vector (hadamard of source + target node embeddings).
    pub vector: Vec<f32>,
}

/// Complete embedding result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// Method used to generate embeddings.
    pub method: String,
    /// Embedding dimensionality.
    pub dimensions: usize,
    /// Number of nodes.
    pub num_nodes: usize,
    /// Number of edges.
    pub num_edges: usize,
    /// Node embeddings.
    pub nodes: Vec<NodeEmbedding>,
    /// Edge embeddings (optional, derived from node embeddings via hadamard product).
    pub edges: Option<Vec<EdgeEmbedding>>,
}

/// Configuration for embedding generation.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// Embedding method.
    pub method: EmbedMethod,
    /// Embedding dimension (default: 64).
    pub dimensions: usize,
    /// Walk length for node2vec (default: 30).
    pub walk_length: usize,
    /// Number of walks per node for node2vec (default: 10).
    pub num_walks: usize,
    /// Return parameter p for node2vec (default: 1.0).
    pub p: f64,
    /// In-out parameter q for node2vec (default: 1.0).
    pub q: f64,
    /// Window size for skip-gram (default: 5).
    pub window: usize,
    /// Number of negative samples for skip-gram (default: 5).
    pub negative_samples: usize,
    /// Learning rate (default: 0.025).
    pub lr: f64,
    /// Number of LINE training epochs (default: 5).
    pub epochs: usize,
    /// Number of threads (default: 1).
    pub threads: usize,
    /// Include edge embeddings (default: false).
    pub include_edges: bool,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            method: EmbedMethod::Node2Vec,
            dimensions: 64,
            walk_length: 30,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            window: 5,
            negative_samples: 5,
            lr: 0.025,
            epochs: 5,
            threads: 1,
            include_edges: false,
        }
    }
}

// ── Adjacency structure ───────────────────────────────────────────────

struct Adjacency {
    /// neighbors[node_idx] = [(neighbor_idx, edge_idx, weight)]
    neighbors: Vec<Vec<(usize, usize, f64)>>,
    /// degree[node_idx] = total degree
    degree: Vec<usize>,
    /// num_nodes
    num_nodes: usize,
}

impl Adjacency {
    fn build(nodes: &[RmpNode], edges: &[RmpEdge]) -> Self {
        let n = nodes.len();
        let mut neighbors = vec![Vec::new(); n];
        let mut degree = vec![0usize; n];

        for (ei, e) in edges.iter().enumerate() {
            let from = e.from as usize;
            let to = e.to as usize;
            if from < n && to < n {
                let w = e.weight_m;
                neighbors[from].push((to, ei, w));
                neighbors[to].push((from, ei, w));
                degree[from] += 1;
                degree[to] += 1;
            }
        }

        Adjacency {
            neighbors,
            degree,
            num_nodes: n,
        }
    }

    fn degree(&self, node: usize) -> usize {
        self.degree.get(node).copied().unwrap_or(0)
    }
}

// ── node2vec ──────────────────────────────────────────────────────────

/// Generate biased random walks for node2vec.
fn generate_walks(adj: &Adjacency, config: &EmbedConfig) -> Vec<Vec<usize>> {
    let n = adj.num_nodes;
    let mut rng = thread_rng();
    let mut walks = Vec::with_capacity(n * config.num_walks);

    for _ in 0..config.num_walks {
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        for &start in &order {
            if adj.degree(start) == 0 {
                continue;
            }
            let walk = single_walk(adj, start, config.walk_length, config.p, config.q, &mut rng);
            walks.push(walk);
        }
    }

    walks
}

fn single_walk(
    adj: &Adjacency,
    start: usize,
    length: usize,
    p: f64,
    q: f64,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let mut walk = Vec::with_capacity(length);
    walk.push(start);

    if adj.degree(start) == 0 {
        return walk;
    }

    // First step: uniform random neighbor
    let neighbors = &adj.neighbors[start];
    let next = neighbors.choose(rng).map(|(n, _, _)| *n).unwrap();
    walk.push(next);

    for _ in 2..length {
        let cur = *walk.last().unwrap();
        let prev = walk[walk.len() - 2];
        let neighbors = &adj.neighbors[cur];

        if neighbors.is_empty() {
            break;
        }

        // Compute transition probabilities with bias
        let mut weights: Vec<f64> = Vec::with_capacity(neighbors.len());
        for &(nbr, _, _w) in neighbors {
            let bias = if nbr == prev {
                1.0 / p // return to previous
            } else if adj.neighbors[prev].iter().any(|(n, _, _)| *n == nbr) {
                1.0 // same distance (BFS-like)
            } else {
                1.0 / q // go further (DFS-like)
            };
            weights.push(bias);
        }

        let dist = WeightedIndex::new(&weights)
            .unwrap_or_else(|_| WeightedIndex::new(vec![1.0f64; neighbors.len()]).unwrap());

        let idx = dist.sample(rng);
        walk.push(neighbors[idx].0);
    }

    walk
}

/// Skip-gram with negative sampling (SGNS) training.
///
/// Simple implementation inspired by word2vec. Trains an embedding matrix
/// where co-occurring nodes in walks have similar representations.
fn train_skip_gram(walks: &[Vec<usize>], num_nodes: usize, config: &EmbedConfig) -> Vec<Vec<f32>> {
    let dim = config.dimensions;
    let lr = config.lr as f32;
    let window = config.window;
    let num_neg = config.negative_samples;

    let mut rng = thread_rng();

    // Initialize embeddings: W (center) and C (context)
    let normal = Normal::new(0.0, 1.0 / (dim as f64).sqrt()).unwrap();
    let mut w_emb: Vec<Vec<f32>> = (0..num_nodes)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng) as f32).collect())
        .collect();
    let mut c_emb: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; num_nodes];

    // Node frequency for negative sampling (use degree as proxy)
    // Since we don't have a separate frequency table, use uniform for now
    let neg_dist = WeightedIndex::new(vec![1.0f64; num_nodes]).unwrap();

    let mut total_loss = 0.0f32;
    let mut loss_count = 0usize;

    for walk in walks {
        let len = walk.len();
        if len < 2 {
            continue;
        }

        for i in 0..len {
            let center = walk[i];
            let w_min = if i > window { i - window } else { 0 };
            let w_max = (i + window + 1).min(len);

            for j in w_min..w_max {
                if i == j {
                    continue;
                }
                let context = walk[j];

                // Positive sample: sigmoid(w · c) should be ~1
                let dot = dot_product(&w_emb[center], &c_emb[context]);
                let sig = sigmoid(dot);
                let grad = lr * (1.0 - sig);
                total_loss += -((sig + 1e-10).ln());
                loss_count += 1;

                // Update center and context vectors
                for d in 0..dim {
                    let g = grad * c_emb[context][d];
                    c_emb[context][d] += grad * w_emb[center][d];
                    w_emb[center][d] += g;
                }

                // Negative samples: sigmoid(w · c_neg) should be ~0
                for _ in 0..num_neg {
                    let neg = neg_dist.sample(&mut rng);
                    if neg == context {
                        continue;
                    }
                    let dot_neg = dot_product(&w_emb[center], &c_emb[neg]);
                    let sig_neg = sigmoid(dot_neg);
                    let grad_neg = lr * (0.0 - sig_neg);

                    for d in 0..dim {
                        let g = grad_neg * c_emb[neg][d];
                        c_emb[neg][d] += grad_neg * w_emb[center][d];
                        w_emb[center][d] += g;
                    }
                }
            }
        }
    }

    if loss_count > 0 {
        tracing::debug!(
            "node2vec SGNS training: avg loss = {:.4} over {} pairs",
            total_loss / loss_count as f32,
            loss_count
        );
    }

    w_emb
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x > 6.0 {
        1.0
    } else if x < -6.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── LINE ──────────────────────────────────────────────────────────────

/// LINE embedding: jointly optimize 1st-order and 2nd-order proximity.
///
/// 1st-order: directly connected nodes should have similar embeddings.
/// 2nd-order: nodes with similar neighborhoods should be similar.
fn train_line(adj: &Adjacency, edges: &[RmpEdge], config: &EmbedConfig) -> Vec<Vec<f32>> {
    let n = adj.num_nodes;
    let dim = config.dimensions;
    let lr = config.lr as f32;
    let num_neg = config.negative_samples;
    let epochs = config.epochs;

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0 / (dim as f64).sqrt()).unwrap();

    // 1st-order embeddings
    let mut u1: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng) as f32).collect())
        .collect();

    // 2nd-order embeddings (context)
    let mut u2: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng) as f32).collect())
        .collect();
    let mut u2_ctx: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; n];

    // Node degree distribution for negative sampling
    let degree_sum: f64 = adj.degree.iter().map(|&d| d as f64).sum::<f64>().max(1.0);
    let neg_weights: Vec<f64> = adj
        .degree
        .iter()
        .map(|&d| ((d as f64) / degree_sum).powf(0.75))
        .collect();
    let neg_dist = WeightedIndex::new(&neg_weights)
        .unwrap_or_else(|_| WeightedIndex::new(vec![1.0f64; n]).unwrap());

    let num_edges = edges.len();
    if num_edges == 0 {
        return u1;
    }

    let mut total_loss = 0.0f32;
    let mut loss_count = 0usize;

    for _epoch in 0..epochs {
        let mut edge_order: Vec<usize> = (0..num_edges).collect();
        edge_order.shuffle(&mut rng);

        let progress_lr = lr; // could decay with epoch

        for &ei in &edge_order {
            let e = &edges[ei];
            let u = e.from as usize;
            let v = e.to as usize;
            if u >= n || v >= n {
                continue;
            }

            let w = e.weight_m.max(1.0) as f32;

            // ── 1st-order proximity ──
            // Optimize: -w * sigmoid(u1_u · u1_v) for positive pair
            let dot1 = dot_product(&u1[u], &u1[v]);
            let sig1 = sigmoid(dot1);
            let loss1 = -w * (sig1 + 1e-10).ln();
            total_loss += loss1;
            loss_count += 1;

            let grad1 = progress_lr * w * (1.0 - sig1);
            for d in 0..dim {
                let g = grad1 * u1[v][d];
                u1[v][d] += grad1 * u1[u][d];
                u1[u][d] += g;
            }

            // Negative samples for 1st order
            for _ in 0..num_neg {
                let neg = neg_dist.sample(&mut rng);
                if neg == v {
                    continue;
                }
                let dot_neg = dot_product(&u1[u], &u1[neg]);
                let sig_neg = sigmoid(dot_neg);
                let grad_neg = progress_lr * w * (0.0 - sig_neg);
                for d in 0..dim {
                    let g = grad_neg * u1[neg][d];
                    u1[neg][d] += grad_neg * u1[u][d];
                    u1[u][d] += g;
                }
            }

            // ── 2nd-order proximity ──
            // Context: node u should predict neighbor v
            let dot2 = dot_product(&u2[u], &u2_ctx[v]);
            let sig2 = sigmoid(dot2);
            let grad2 = progress_lr * w * (1.0 - sig2);
            for d in 0..dim {
                let g = grad2 * u2_ctx[v][d];
                u2_ctx[v][d] += grad2 * u2[u][d];
                u2[u][d] += g;
            }

            for _ in 0..num_neg {
                let neg = neg_dist.sample(&mut rng);
                if neg == v {
                    continue;
                }
                let dot_neg = dot_product(&u2[u], &u2_ctx[neg]);
                let sig_neg = sigmoid(dot_neg);
                let grad_neg = progress_lr * w * (0.0 - sig_neg);
                for d in 0..dim {
                    let g = grad_neg * u2_ctx[neg][d];
                    u2_ctx[neg][d] += grad_neg * u2[u][d];
                    u2[u][d] += g;
                }
            }
        }
    }

    if loss_count > 0 {
        tracing::debug!(
            "LINE training: avg 1st-order loss = {:.4} over {} samples",
            total_loss / loss_count as f32,
            loss_count
        );
    }

    // Concatenate 1st and 2nd order embeddings → 2*dim
    // Or average them → dim
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut vec = Vec::with_capacity(dim);
        for d in 0..dim {
            vec.push((u1[i][d] + u2[i][d]) / 2.0);
        }
        // Normalize
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for v in &mut vec {
            *v /= norm;
        }
        result.push(vec);
    }

    result
}

// ── FastRP ────────────────────────────────────────────────────────────

/// Fast Random Projection embedding (Chen et al. 2019).
///
/// Approximates the node similarity matrix via sparse random projection.
/// Much faster than node2vec/LINE for large graphs.
fn train_fastrp(adj: &Adjacency, config: &EmbedConfig) -> Vec<Vec<f32>> {
    let n = adj.num_nodes;
    let dim = config.dimensions;
    let mut rng = thread_rng();

    // Generate random projection matrix S ∈ {-1, 0, 1}^(n × dim)
    // Each entry: P(-1/s) = P(1/s) = 1/(2s), P(0) = 1 - 1/s
    // where s = sqrt(n) (sparsity parameter)
    let s = (n as f64).sqrt().max(1.0);
    let prob = 1.0 / (2.0 * s);

    let mut projection: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; n];
    for i in 0..n {
        for d in 0..dim {
            let r: f64 = rng.gen();
            if r < prob {
                projection[i][d] = -1.0;
            } else if r < 2.0 * prob {
                projection[i][d] = 1.0;
            }
            // else stays 0
        }
    }

    // Power iteration: H^{(0)} = S, H^{(k+1)} = A * H^{(k)} + S
    // Captures k-hop neighborhood structure
    let num_iters = 3; // captures up to 3-hop neighborhoods
    let mut h = projection.clone();

    for _ in 0..num_iters {
        let mut new_h = vec![vec![0.0f32; dim]; n];
        for i in 0..n {
            // Self-loop: add projection
            for d in 0..dim {
                new_h[i][d] += projection[i][d];
            }
            // Aggregate neighbors
            let neighbors = &adj.neighbors[i];
            let deg = neighbors.len().max(1) as f32;
            for &(nbr, _, _) in neighbors {
                for d in 0..dim {
                    new_h[i][d] += h[nbr][d] / deg;
                }
            }
        }
        h = new_h;
    }

    // Normalize each row
    for i in 0..n {
        let norm = h[i].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for v in &mut h[i] {
            *v /= norm;
        }
    }

    h
}

// ── Spatial + Structural features ─────────────────────────────────────

/// Generate handcrafted spatial + structural features for each node.
///
/// Features per node:
/// - lat, lon (normalized)
/// - degree
/// - log(degree)
/// - avg_edge_weight (mean distance to neighbors)
/// - max_edge_weight
/// - min_edge_weight
/// - std_edge_weight
/// - clustering_coefficient_approx
/// - betweenness_centrality_approx (sampled)
/// - eigenvector_centrality_approx
///
/// Padded/truncated to config.dimensions.
fn spatial_features(
    nodes: &[RmpNode],
    edges: &[RmpEdge],
    adj: &Adjacency,
    config: &EmbedConfig,
) -> Vec<Vec<f32>> {
    let n = nodes.len();
    let dim = config.dimensions;

    // Compute lat/lon bounds for normalization
    let lat_min = nodes.iter().map(|n| n.lat).fold(f64::INFINITY, f64::min);
    let lat_max = nodes
        .iter()
        .map(|n| n.lat)
        .fold(f64::NEG_INFINITY, f64::max);
    let lon_min = nodes.iter().map(|n| n.lon).fold(f64::INFINITY, f64::min);
    let lon_max = nodes
        .iter()
        .map(|n| n.lon)
        .fold(f64::NEG_INFINITY, f64::max);
    let lat_range = (lat_max - lat_min).max(1e-10);
    let lon_range = (lon_max - lon_min).max(1e-10);

    // Compute edge weight statistics per node
    let mut node_weights: Vec<Vec<f64>> = vec![Vec::new(); n];
    for e in edges {
        node_weights[e.from as usize].push(e.weight_m);
        node_weights[e.to as usize].push(e.weight_m);
    }

    // Approximate betweenness centrality via sampling
    let bc = approx_betweenness(adj, 200.min(n));

    // Approximate eigenvector centrality via power iteration
    let ec = approx_eigenvector_centrality(adj, 20);

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let lat_norm = ((nodes[i].lat - lat_min) / lat_range) as f32;
        let lon_norm = ((nodes[i].lon - lon_min) / lon_range) as f32;
        let deg = adj.degree(i) as f32;
        let log_deg = (deg as f32 + 1.0).ln();

        let weights = &node_weights[i];
        let avg_w = if weights.is_empty() {
            0.0
        } else {
            (weights.iter().sum::<f64>() / weights.len() as f64) as f32
        };
        let max_w = weights.iter().cloned().fold(0.0f64, f64::max) as f32;
        let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min) as f32;
        let std_w = if weights.len() > 1 {
            let mean = weights.iter().sum::<f64>() / weights.len() as f64;
            let variance =
                weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / weights.len() as f64;
            variance.sqrt() as f32
        } else {
            0.0f32
        };

        let clustering = approx_clustering_coefficient(adj, i);

        let mut features = vec![
            lat_norm,
            lon_norm,
            deg / (n as f32).sqrt().max(1.0),
            log_deg,
            avg_w / 1000.0, // normalize to km
            max_w / 1000.0,
            min_w / 1000.0,
            std_w / 1000.0,
            clustering,
            bc[i],
            ec[i],
        ];

        // Pad or truncate to desired dimensions
        features.resize(dim, 0.0f32);

        // Normalize the feature vector
        let norm = features
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);
        for v in &mut features {
            *v /= norm;
        }

        result.push(features);
    }

    result
}

/// Approximate local clustering coefficient for a single node.
fn approx_clustering_coefficient(adj: &Adjacency, node: usize) -> f32 {
    let neighbors: Vec<usize> = adj.neighbors[node].iter().map(|(n, _, _)| *n).collect();
    let k = neighbors.len();
    if k < 2 {
        return 0.0;
    }

    let neighbor_set: std::collections::HashSet<usize> = neighbors.iter().copied().collect();
    let mut triangles = 0usize;
    for &n1 in &neighbors {
        for &(n2, _, _) in &adj.neighbors[n1] {
            if neighbor_set.contains(&n2) && n2 > n1 {
                triangles += 1;
            }
        }
    }

    let max_triangles = k * (k - 1) / 2;
    triangles as f32 / max_triangles as f32
}

/// Approximate betweenness centrality via random sampling (Brandes approximation).
fn approx_betweenness(adj: &Adjacency, sample_size: usize) -> Vec<f32> {
    let n = adj.num_nodes;
    let mut bc = vec![0.0f32; n];

    if n == 0 {
        return bc;
    }

    let mut rng = thread_rng();
    let actual_samples = sample_size.min(n);

    for _ in 0..actual_samples {
        let source = rng.gen_range(0..n);
        if adj.degree(source) == 0 {
            continue;
        }

        // BFS from source
        let mut dist = vec![i32::MAX; n];
        let mut sigma = vec![0.0f64; n];
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut stack = Vec::new();

        dist[source] = 0;
        sigma[source] = 1.0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &(nbr, _, _) in &adj.neighbors[v] {
                if dist[nbr] == i32::MAX {
                    dist[nbr] = dist[v] + 1;
                    queue.push_back(nbr);
                }
                if dist[nbr] == dist[v] + 1 {
                    sigma[nbr] += sigma[v];
                    pred[nbr].push(v);
                }
            }
        }

        // Back-propagation
        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                let contribution = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                delta[v] += contribution;
            }
            if w != source {
                bc[w] += delta[w] as f32;
            }
        }
    }

    // Normalize
    let max_bc = bc.iter().cloned().fold(0.0f32, f32::max).max(1e-10);
    for v in &mut bc {
        *v /= max_bc;
    }

    bc
}

/// Approximate eigenvector centrality via power iteration.
fn approx_eigenvector_centrality(adj: &Adjacency, iterations: usize) -> Vec<f32> {
    let n = adj.num_nodes;
    let mut ec = vec![1.0f32 / (n as f32).sqrt(); n];

    for _ in 0..iterations {
        let mut new_ec = vec![0.0f32; n];
        for i in 0..n {
            for &(nbr, _, _) in &adj.neighbors[i] {
                new_ec[i] += ec[nbr];
            }
        }
        let norm = new_ec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for v in &mut new_ec {
            *v /= norm;
        }
        ec = new_ec;
    }

    ec
}

// ── Edge embedding from node embeddings ───────────────────────────────

/// Derive edge embeddings from node embeddings using Hadamard product.
fn embed_edges(edges: &[RmpEdge], node_embs: &[Vec<f32>]) -> Vec<EdgeEmbedding> {
    edges
        .iter()
        .enumerate()
        .map(|(ei, e)| {
            let from_emb = &node_embs[e.from as usize];
            let to_emb = &node_embs[e.to as usize];
            let vector: Vec<f32> = from_emb
                .iter()
                .zip(to_emb.iter())
                .map(|(a, b)| a * b) // Hadamard product
                .collect();

            EdgeEmbedding {
                edge_idx: ei,
                from: e.from,
                to: e.to,
                vector,
            }
        })
        .collect()
}

// ── Main entry point ──────────────────────────────────────────────────

/// Generate road network node embeddings.
///
/// This is the main public API. Takes a road network (nodes + edges from .rmp)
/// and a configuration, returns node and optionally edge embeddings.
pub fn embed_graph(
    nodes: &[RmpNode],
    edges: &[RmpEdge],
    config: &EmbedConfig,
) -> anyhow::Result<EmbeddingResult> {
    if nodes.is_empty() {
        anyhow::bail!("Cannot embed empty road network (0 nodes)");
    }

    let adj = Adjacency::build(nodes, edges);
    let method_name = match config.method {
        EmbedMethod::Node2Vec => "node2vec",
        EmbedMethod::Line => "LINE",
        EmbedMethod::FastRp => "FastRP",
        EmbedMethod::Spatial => "spatial",
    };

    tracing::info!(
        "Generating {} embeddings: {} nodes, {} edges, dim={}",
        method_name,
        nodes.len(),
        edges.len(),
        config.dimensions,
    );

    let start = std::time::Instant::now();

    let node_vectors = match config.method {
        EmbedMethod::Node2Vec => {
            tracing::debug!(
                "node2vec: generating {} walks of length {} (p={}, q={})",
                nodes.len() * config.num_walks,
                config.walk_length,
                config.p,
                config.q,
            );
            let walks = generate_walks(&adj, config);
            tracing::debug!(
                "node2vec: generated {} walks, training skip-gram...",
                walks.len()
            );
            train_skip_gram(&walks, nodes.len(), config)
        }
        EmbedMethod::Line => {
            tracing::debug!("LINE: training {} epochs...", config.epochs);
            train_line(&adj, edges, config)
        }
        EmbedMethod::FastRp => {
            tracing::debug!("FastRP: random projection with 3 power iterations...");
            train_fastrp(&adj, config)
        }
        EmbedMethod::Spatial => {
            tracing::debug!("Spatial: computing structural features...");
            spatial_features(nodes, edges, &adj, config)
        }
    };

    let elapsed = start.elapsed();
    tracing::info!(
        "{} embedding complete: {:.2}s ({:.0} nodes/sec)",
        method_name,
        elapsed.as_secs_f64(),
        nodes.len() as f64 / elapsed.as_secs_f64().max(0.001),
    );

    // Build node embeddings with metadata
    let node_embs: Vec<NodeEmbedding> = nodes
        .iter()
        .enumerate()
        .zip(node_vectors.iter())
        .map(|((idx, node), vector)| NodeEmbedding {
            node_idx: idx,
            lat: node.lat,
            lon: node.lon,
            vector: vector.clone(),
        })
        .collect();

    // Optionally derive edge embeddings
    let edge_embs = if config.include_edges {
        Some(embed_edges(edges, &node_vectors))
    } else {
        None
    };

    Ok(EmbeddingResult {
        method: method_name.to_string(),
        dimensions: config.dimensions,
        num_nodes: nodes.len(),
        num_edges: edges.len(),
        nodes: node_embs,
        edges: edge_embs,
    })
}
