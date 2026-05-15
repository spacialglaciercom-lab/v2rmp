//! Zone partition via spectral clustering (no GNN).
//!
//! Ported from `rmp.ca/backend/app/main.py` partition logic:
//!   - normalized Laplacian spectral clustering
//!   - KMeans fallback for large graphs
//!   - greedy balance post-processing
//!   - KNN graph from points
//!   - convex hull polygons per zone
//!
//! MCP tools: `partition`, `partition_from_points`

use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};

// ── Request / Response models ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInput {
    pub u: usize,
    pub v: usize,
    #[serde(default = "default_one")]
    pub length: f64,
    #[serde(default = "default_one")]
    pub intersection_density: f64,
    #[serde(default = "default_one")]
    pub cul_de_sac_penalty: f64,
    #[serde(default = "default_one")]
    pub width_penalty: f64,
}

fn default_one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointInput {
    pub lat: f64,
    pub lon: f64,
    #[serde(default = "default_one")]
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneOutput {
    pub zone_id: usize,
    pub node_ids: Vec<usize>,
    pub estimated_time: f64,
    pub estimated_distance: Option<f64>,
    pub zone_polygon: Option<Vec<[f64; 2]>>, // exterior ring [[lon, lat]...]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionResponse {
    pub zones: Vec<ZoneOutput>,
    pub warnings: Vec<String>,
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Haversine distance in km.
pub fn haversine_km(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let r = 6371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    r * c
}

fn complexity_factor(e: &EdgeInput) -> f64 {
    e.intersection_density * e.cul_de_sac_penalty * e.width_penalty
}

fn edge_weight(e: &EdgeInput, balance_metric: &str) -> f64 {
    if balance_metric == "distance" {
        e.length
    } else {
        e.length * complexity_factor(e)
    }
}

/// Round to 6 decimal places (matches Python `round(x, 6)`).
fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

// ── Adjacency list → normalized Laplacian embedding ──────────────────

type AdjList = Vec<Vec<(usize, f64)>>;

fn build_adjacency(edges: &[EdgeInput], n: usize, balance_metric: &str) -> (AdjList, Vec<f64>) {
    let mut adj: AdjList = vec![vec![]; n];
    let mut degrees = vec![0.0f64; n];
    let mut seen = std::collections::HashSet::new();
    for e in edges {
        if e.u == e.v || e.u >= n || e.v >= n {
            continue;
        }
        let key = (e.u.min(e.v), e.u.max(e.v));
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        let w = edge_weight(e, balance_metric);
        adj[e.u].push((e.v, w));
        adj[e.v].push((e.u, w));
        degrees[e.u] += w;
        degrees[e.v] += w;
    }
    (adj, degrees)
}

/// Power iteration to approximate leading k eigenvectors of D^{-1/2} A D^{-1/2}.
fn sparse_spectral_embedding(adj: &AdjList, degrees: &[f64], k: usize) -> Option<DMatrix<f64>> {
    let n = adj.len();
    if n <= 1 || k == 0 {
        return None;
    }

    let d_inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|&x| if x > 0.0 { 1.0 / x.sqrt() } else { 1.0 })
        .collect();

    let actual_k = k.min(n);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let normal = rand_distr::Normal::new(0.0, 1.0).ok()?;

    let mut v_mat = DMatrix::<f64>::zeros(n, actual_k);
    for j in 0..actual_k {
        for i in 0..n {
            v_mat[(i, j)] = normal.sample(&mut rng);
        }
    }

    for _iter in 0..50 {
        // v_mat = L_shifted * v_mat  where L_shifted[i,j] = d_inv_sqrt[i] * A[i,j] * d_inv_sqrt[j]
        let mut v_new = DMatrix::<f64>::zeros(n, actual_k);
        for j in 0..actual_k {
            for i in 0..n {
                let mut sum = 0.0;
                for &(nb, w) in &adj[i] {
                    sum += d_inv_sqrt[i] * w * d_inv_sqrt[nb] * v_mat[(nb, j)];
                }
                v_new[(i, j)] = sum;
            }
        }
        v_mat = v_new;

        // QR orthogonalize
        let qr = nalgebra::linalg::QR::new(v_mat.clone());
        v_mat = qr.q();

        // Normalize columns
        for j in 0..actual_k {
            let mut norm_sq = 0.0f64;
            for i in 0..n {
                norm_sq += v_mat[(i, j)].powi(2);
            }
            let norm = norm_sq.sqrt().max(1e-12);
            for i in 0..n {
                v_mat[(i, j)] /= norm;
            }
        }
    }

    // Normalize rows (Ng et al.)
    for i in 0..n {
        let mut norm_sq = 0.0f64;
        for j in 0..actual_k {
            norm_sq += v_mat[(i, j)].powi(2);
        }
        let norm = norm_sq.sqrt().max(1e-12);
        for j in 0..actual_k {
            v_mat[(i, j)] /= norm;
        }
    }

    Some(v_mat)
}

// ── KMeans (Lloyd's algorithm) ──────────────────────────────────────

fn kmeans_cluster(data: &DMatrix<f64>, k: usize, max_iters: usize) -> Vec<usize> {
    let n = data.nrows();
    let dim = data.ncols();
    if k >= n {
        return (0..n).collect();
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let mut centroids = DMatrix::<f64>::zeros(k, dim);
    for (ci, &idx) in indices.iter().take(k).enumerate() {
        centroids.set_row(ci, &data.row(idx));
    }

    let mut labels = vec![0usize; n];

    for _iter in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_dist = f64::MAX;
            for c in 0..k {
                let mut dist = 0.0f64;
                for d in 0..dim {
                    let delta = data[(i, d)] - centroids[(c, d)];
                    dist += delta * delta;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                changed = true;
                labels[i] = best_c;
            }
        }
        if !changed {
            break;
        }

        // Update
        let mut new_centroids = DMatrix::<f64>::zeros(k, dim);
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            for d in 0..dim {
                new_centroids[(c, d)] += data[(i, d)];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    new_centroids[(c, d)] /= counts[c] as f64;
                }
            }
        }
        centroids = new_centroids;
    }

    labels
}

fn kmeans_1d(values: &[f64], k: usize) -> Vec<usize> {
    let n = values.len();
    if k >= n {
        return (0..n).collect();
    }
    let data = DMatrix::from_column_slice(n, 1, values);
    kmeans_cluster(&data, k, 20)
}

// ── Spectral partition ────────────────────────────────────────────────

const SPECTRAL_MAX_NODES: usize = 8000;

fn spectral_partition(adj: &AdjList, degrees: &[f64], k: usize) -> Vec<usize> {
    let n = adj.len();
    if k >= n {
        return (0..n).collect();
    }

    // 1-D degree-based for large graphs
    if n > SPECTRAL_MAX_NODES {
        return kmeans_1d(degrees, k);
    }

    // Try spectral embedding
    if let Some(embedding) = sparse_spectral_embedding(adj, degrees, k) {
        return kmeans_cluster(&embedding, k, 20);
    }

    // Fallback
    kmeans_1d(degrees, k)
}

// ── Zone weight computation ───────────────────────────────────────────

fn zone_weights_from_labels(
    labels: &[usize],
    edge_weights: &[(usize, usize, f64)],
    k: usize,
) -> Vec<f64> {
    let mut zone_weights = vec![0.0f64; k];
    for &(u, v, w) in edge_weights {
        if u >= labels.len() || v >= labels.len() {
            continue;
        }
        let zu = labels[u];
        let zv = labels[v];
        if zu < k && zv < k {
            if zu == zv {
                zone_weights[zu] += w;
            } else {
                zone_weights[zu] += w * 0.5;
                zone_weights[zv] += w * 0.5;
            }
        }
    }
    zone_weights
}

fn zone_weights_from_node_weights(labels: &[usize], node_weights: &[f64], k: usize) -> Vec<f64> {
    let mut zone_weights = vec![0.0f64; k];
    for (i, &w) in node_weights.iter().enumerate() {
        if i < labels.len() {
            let z = labels[i];
            if z < k {
                zone_weights[z] += w;
            }
        }
    }
    zone_weights
}

// ── Balance post-processing (greedy node moves) ──────────────────────

fn balance_postprocess(
    mut labels: Vec<usize>,
    edge_weights: &[(usize, usize, f64)],
    k: usize,
    max_iters: usize,
) -> Vec<usize> {
    let n = labels.len();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

    let mut adj: AdjList = vec![vec![]; n];
    for &(u, v, w) in edge_weights {
        if u < n && v < n {
            adj[u].push((v, w));
            adj[v].push((u, w));
        }
    }

    let mut zone_weights = zone_weights_from_labels(&labels, edge_weights, k);
    let total: f64 = zone_weights.iter().sum();
    let target = if total > 0.0 { total / k as f64 } else { 0.0 };
    let mut current_imb: f64 = zone_weights.iter().map(|&w| (w - target).powi(2)).sum();

    for _iter in 0..max_iters {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let mut improved = false;
        for u in 0..n {
            if u % 1000 == 0 && std::time::Instant::now() >= deadline {
                break;
            }
            let z_old = labels[u];
            let half_total: f64 = adj[u].iter().map(|(_, w)| w).sum::<f64>() * 0.5;
            if half_total <= 0.0 {
                continue;
            }

            let trial_old = zone_weights[z_old] - half_total;
            let mut best_z = z_old;
            let mut best_imb = current_imb;

            for z_new in 0..k {
                if z_new == z_old {
                    continue;
                }
                let trial_new = zone_weights[z_new] + half_total;
                let imb = current_imb
                    - (zone_weights[z_old] - target).powi(2)
                    - (zone_weights[z_new] - target).powi(2)
                    + (trial_old - target).powi(2)
                    + (trial_new - target).powi(2);
                if imb < best_imb {
                    best_imb = imb;
                    best_z = z_new;
                }
            }

            if best_z != z_old {
                zone_weights[z_old] -= half_total;
                zone_weights[best_z] += half_total;
                labels[u] = best_z;
                current_imb = best_imb;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    labels
}

fn balance_postprocess_node_weights(
    mut labels: Vec<usize>,
    node_weights: &[f64],
    k: usize,
    max_iters: usize,
) -> Vec<usize> {
    let n = labels.len();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let weights = if node_weights.len() >= n {
        node_weights.to_vec()
    } else {
        vec![1.0; n]
    };

    let mut zone_weights = zone_weights_from_node_weights(&labels, &weights, k);
    let total: f64 = zone_weights.iter().sum();
    let target = if total > 0.0 { total / k as f64 } else { 0.0 };
    let mut current_imb: f64 = zone_weights.iter().map(|&w| (w - target).powi(2)).sum();

    for _iter in 0..max_iters {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let mut improved = false;
        for u in 0..n {
            if u % 1000 == 0 && std::time::Instant::now() >= deadline {
                break;
            }
            let wu = *weights.get(u).unwrap_or(&0.0);
            if wu <= 0.0 {
                continue;
            }
            let z_old = labels[u];
            let trial_old = zone_weights[z_old] - wu;
            let mut best_z = z_old;
            let mut best_imb = current_imb;

            for z_new in 0..k {
                if z_new == z_old {
                    continue;
                }
                let trial_new = zone_weights[z_new] + wu;
                let imb = current_imb
                    - (zone_weights[z_old] - target).powi(2)
                    - (zone_weights[z_new] - target).powi(2)
                    + (trial_old - target).powi(2)
                    + (trial_new - target).powi(2);
                if imb < best_imb {
                    best_imb = imb;
                    best_z = z_new;
                }
            }

            if best_z != z_old {
                zone_weights[z_old] -= wu;
                zone_weights[best_z] += wu;
                labels[u] = best_z;
                current_imb = best_imb;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    labels
}

// ── Convex hull (Graham scan) ────────────────────────────────────────

fn convex_hull_ring(coords: &[[f64; 2]], indices: &[usize]) -> Option<Vec<[f64; 2]>> {
    if indices.len() < 3 {
        return None;
    }
    let points: Vec<[f64; 2]> = indices
        .iter()
        .filter_map(|&i| coords.get(i).copied())
        .collect();
    if points.len() < 3 {
        return None;
    }

    let mut pts = points.clone();

    // Find lowest (min y, then min x)
    let mut min_idx = 0;
    for i in 1..pts.len() {
        if pts[i][1] < pts[min_idx][1]
            || (pts[i][1] == pts[min_idx][1] && pts[i][0] < pts[min_idx][0])
        {
            min_idx = i;
        }
    }
    pts.swap(0, min_idx);
    let pivot = pts[0];

    // Sort by polar angle
    pts[1..].sort_by(|a, b| {
        let cross = (a[0] - pivot[0]) * (b[1] - pivot[1])
            - (a[1] - pivot[1]) * (b[0] - pivot[0]);
        if cross > 0.0 {
            std::cmp::Ordering::Less
        } else if cross < 0.0 {
            std::cmp::Ordering::Greater
        } else {
            let da = (a[0] - pivot[0]).powi(2) + (a[1] - pivot[1]).powi(2);
            let db = (b[0] - pivot[0]).powi(2) + (b[1] - pivot[1]).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    let mut hull: Vec<[f64; 2]> = vec![pts[0], pts[1]];
    for p in pts.iter().skip(2) {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(*p);
    }

    hull.push(hull[0]);
    Some(hull)
}

// ── KNN graph from points ─────────────────────────────────────────────

fn points_to_knn_graph(points: &[PointInput], knn_neighbors: usize) -> Vec<EdgeInput> {
    let n = points.len();
    if n == 0 || knn_neighbors < 1 {
        return vec![];
    }

    let k_query = (knn_neighbors + 1).min(n);
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d = haversine_km(points[i].lon, points[i].lat, points[j].lon, points[j].lat);
                (d, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for &(_d, j) in dists.iter().take(k_query - 1) {
            let key = (i.min(j), i.max(j));
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);
            let length = haversine_km(points[i].lon, points[i].lat, points[j].lon, points[j].lat);
            if length > 0.0 {
                edges.push(EdgeInput {
                    u: i,
                    v: j,
                    length,
                    intersection_density: 1.0,
                    cul_de_sac_penalty: 1.0,
                    width_penalty: 1.0,
                });
            }
        }
    }

    edges
}

// ── Main partition function ───────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn partition_graph(
    edges: &[EdgeInput],
    node_count: usize,
    truck_count: usize,
    balance_metric: &str,
    node_coords: Option<&[[f64; 2]]>,
    node_weights: Option<&[f64]>,
    include_polygons: bool,
) -> PartitionResponse {
    let n = node_count;
    let k_eff = truck_count.min(n);
    let mut warnings: Vec<String> = Vec::new();
    let use_node_weights = node_weights.is_some() && node_weights.map_or(false, |w| w.len() >= n);

    // Pure points path: no edges → KMeans on coordinates
    if edges.is_empty() {
        if let Some(coords) = node_coords {
            if coords.len() == n {
                let flat: Vec<f64> = coords.iter().flat_map(|c| [c[0], c[1]]).collect();
                let coords_mat = DMatrix::from_row_slice(n, 2, &flat);
                let labels = kmeans_cluster(&coords_mat, k_eff, 20);

                let labels = if use_node_weights {
                    balance_postprocess_node_weights(labels, node_weights.unwrap(), k_eff, 50)
                } else {
                    labels
                };

                return build_zone_output(
                    &labels, truck_count, k_eff,
                    node_weights, edges, node_coords, include_polygons, &warnings,
                );
            }
        }

        let labels: Vec<usize> = (0..n).map(|i| i % k_eff).collect();
        return build_zone_output(
            &labels, truck_count, k_eff,
            node_weights, edges, node_coords, include_polygons, &warnings,
        );
    }

    // Validate node coverage
    let referenced: std::collections::HashSet<usize> = edges
        .iter()
        .flat_map(|e| [e.u, e.v])
        .collect();
    let unreferenced: Vec<usize> = (0..n).filter(|i| !referenced.contains(i)).collect();
    if !unreferenced.is_empty() {
        warnings.push(format!(
            "{} node(s) not referenced by any edge (e.g. {:?}). \
             These will be distributed across zones but carry no weight.",
            unreferenced.len(),
            &unreferenced[..unreferenced.len().min(5)]
        ));
    }

    let (adj, adj_degs) = build_adjacency(edges, n, balance_metric);
    let edge_weights: Vec<(usize, usize, f64)> = edges
        .iter()
        .map(|e| (e.u, e.v, edge_weight(e, balance_metric)))
        .collect();
    let total_weight: f64 = edge_weights.iter().map(|(_, _, w)| w).sum();

    let labels = if total_weight == 0.0 {
        let mut labels: Vec<usize> = (0..n).map(|i| i % k_eff).collect();
        if use_node_weights {
            labels = balance_postprocess_node_weights(labels, node_weights.unwrap(), k_eff, 50);
        }
        labels
    } else {
        let connected: Vec<usize> = (0..n).filter(|&i| adj_degs[i] > 0.0).collect();
        let isolates: Vec<usize> = (0..n).filter(|&i| adj_degs[i] <= 0.0).collect();

        if connected.is_empty() {
            let mut labels: Vec<usize> = (0..n).map(|i| i % k_eff).collect();
            if use_node_weights {
                labels = balance_postprocess_node_weights(labels, node_weights.unwrap(), k_eff, 50);
            }
            labels
        } else if connected.len() < n {
            // Sub-graph on connected nodes
            let sub_k = k_eff.min(connected.len());
            let idx_map: std::collections::HashMap<usize, usize> = connected
                .iter()
                .enumerate()
                .map(|(compact, &orig)| (orig, compact))
                .collect();

            let mut sub_adj: AdjList = vec![vec![]; connected.len()];
            let mut sub_degs = vec![0.0f64; connected.len()];
            let mut sub_edges = Vec::new();
            let mut seen = std::collections::HashSet::new();

            for e in edges {
                if let (Some(&cu), Some(&cv)) = (idx_map.get(&e.u), idx_map.get(&e.v)) {
                    let key = (cu.min(cv), cu.max(cv));
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);
                    let w = edge_weight(e, balance_metric);
                    sub_adj[cu].push((cv, w));
                    sub_adj[cv].push((cu, w));
                    sub_degs[cu] += w;
                    sub_degs[cv] += w;
                    sub_edges.push((cu, cv, w));
                }
            }

            if connected.len() > SPECTRAL_MAX_NODES {
                warnings.push(format!(
                    "Graph has {} connected nodes; using fast degree-based clustering.",
                    connected.len()
                ));
            }

            let mut sub_labels = spectral_partition(&sub_adj, &sub_degs, sub_k);

            if use_node_weights {
                let sub_nw: Vec<f64> = connected.iter().map(|&i| node_weights.unwrap()[i]).collect();
                sub_labels = balance_postprocess_node_weights(sub_labels, &sub_nw, sub_k, 50);
            } else {
                sub_labels = balance_postprocess(sub_labels, &sub_edges, sub_k, 50);
            }

            let mut labels = vec![0usize; n];
            for (compact, &orig) in connected.iter().enumerate() {
                labels[orig] = sub_labels[compact];
            }

            let mut zone_counts = vec![0usize; k_eff];
            for &l in labels.iter() {
                if l < k_eff {
                    zone_counts[l] += 1;
                }
            }
            for iso in isolates {
                let lightest = zone_counts
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &c)| c)
                    .map(|(z, _)| z)
                    .unwrap_or(0);
                labels[iso] = lightest;
                zone_counts[lightest] += 1;
            }
            labels
        } else {
            if n > SPECTRAL_MAX_NODES {
                warnings.push(format!(
                    "Graph has {} nodes; using fast degree-based clustering.", n
                ));
            }
            let mut labels = spectral_partition(&adj, &adj_degs, k_eff);
            if use_node_weights {
                labels = balance_postprocess_node_weights(labels, node_weights.unwrap(), k_eff, 50);
            } else {
                labels = balance_postprocess(labels, &edge_weights, k_eff, 50);
            }
            labels
        }
    };

    build_zone_output(
        &labels, truck_count, k_eff,
        node_weights, edges, node_coords, include_polygons, &warnings,
    )
}

fn build_zone_output(
    labels: &[usize],
    truck_count: usize,
    k_eff: usize,
    node_weights: Option<&[f64]>,
    edges: &[EdgeInput],
    node_coords: Option<&[[f64; 2]]>,
    include_polygons: bool,
    warnings: &[String],
) -> PartitionResponse {
    let n = labels.len();
    let mut zones_out = Vec::new();

    for z in 0..truck_count {
        if z < k_eff {
            let node_ids: Vec<usize> = (0..n).filter(|&i| labels[i] == z).collect();

            let weight_sum = if let Some(nw) = node_weights {
                node_ids.iter().map(|&i| nw[i]).sum()
            } else if !edges.is_empty() {
                let ew: Vec<(usize, usize, f64)> = edges
                    .iter()
                    .map(|e| (e.u, e.v, e.length))
                    .collect();
                zone_weights_from_labels(labels, &ew, k_eff)[z]
            } else {
                node_ids.len() as f64
            };

            let dist_sum: Option<f64> = None; // distance only meaningful in Python's dual-computation path

            let zone_polygon = if include_polygons {
                node_coords.and_then(|coords| convex_hull_ring(coords, &node_ids))
            } else {
                None
            };

            zones_out.push(ZoneOutput {
                zone_id: z,
                node_ids,
                estimated_time: round6(weight_sum),
                estimated_distance: dist_sum.map(round6),
                zone_polygon,
            });
        } else {
            zones_out.push(ZoneOutput {
                zone_id: z,
                node_ids: vec![],
                estimated_time: 0.0,
                estimated_distance: None,
                zone_polygon: None,
            });
        }
    }

    PartitionResponse {
        zones: zones_out,
        warnings: warnings.to_vec(),
    }
}

// ── High-level API ────────────────────────────────────────────────────

/// Partition from edge list.
pub fn partition_edges(
    edges: &[EdgeInput],
    node_count: usize,
    truck_count: usize,
    balance_metric: &str,
) -> PartitionResponse {
    partition_graph(edges, node_count, truck_count, balance_metric, None, None, false)
}

/// Partition from points.
pub fn partition_from_points(
    points: &[PointInput],
    truck_count: usize,
    balance_metric: &str,
    knn_neighbors: usize,
    include_polygons: bool,
) -> PartitionResponse {
    let edges = points_to_knn_graph(points, knn_neighbors);
    let node_count = points.len();
    let coords: Vec<[f64; 2]> = points.iter().map(|p| [p.lon, p.lat]).collect();

    let (node_weights, use_node_weights) = match balance_metric {
        "count" => (Some(vec![1.0; node_count]), true),
        "weight" => (Some(points.iter().map(|p| p.weight).collect()), true),
        _ => (None, false),
    };

    partition_graph(
        &edges,
        node_count,
        truck_count,
        "distance",
        Some(&coords),
        if use_node_weights {
            node_weights.as_deref()
        } else {
            None
        },
        include_polygons,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_km() {
        let d = haversine_km(-73.5674, 45.5017, -74.0060, 40.7128);
        assert!((d - 530.0).abs() < 20.0, "got {d}");
    }

    #[test]
    fn test_convex_hull_triangle() {
        let coords: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let hull = convex_hull_ring(&coords, &[0, 1, 2]).unwrap();
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_partition_empty() {
        let edges = vec![];
        let res = partition_edges(&edges, 5, 2, "time");
        assert_eq!(res.zones.len(), 2);
        assert!(!res.zones[0].node_ids.is_empty());
    }

    #[test]
    fn test_partition_simple_chain() {
        let edges: Vec<EdgeInput> = (0..4)
            .map(|i| EdgeInput {
                u: i,
                v: i + 1,
                length: 1.0,
                intersection_density: 1.0,
                cul_de_sac_penalty: 1.0,
                width_penalty: 1.0,
            })
            .collect();
        let res = partition_edges(&edges, 5, 2, "time");
        assert_eq!(res.zones.len(), 2);
        let total_nodes: usize = res.zones.iter().map(|z| z.node_ids.len()).sum();
        assert_eq!(total_nodes, 5);
    }

    #[test]
    fn test_partition_from_points() {
        let points: Vec<PointInput> = (0..10)
            .map(|i| PointInput {
                lat: 45.5 + i as f64 * 0.01,
                lon: -73.5 - i as f64 * 0.01,
                weight: 1.0,
            })
            .collect();
        let res = partition_from_points(&points, 3, "count", 3, false);
        assert_eq!(res.zones.len(), 3);
        let total: usize = res.zones.iter().map(|z| z.node_ids.len()).sum();
        assert_eq!(total, 10);
    }
}
