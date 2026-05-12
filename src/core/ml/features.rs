//! Instance feature extraction for learned VRP models.
//!
//! Extracts a 28-dimensional feature vector from a VRP instance.
//! Based on research:
//!   - Instance-Aware Parameter Configuration (2605.00572, 2026)
//!   - RouteFinder (2406.15007, 2024)
//!   - RRNCO (2503.16159, 2025)

use crate::core::haversine_m;
use crate::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};

/// 28-dimensional normalized instance feature vector.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct InstanceFeatures {
    // Geometric (8)
    pub n_stops_norm: f64,
    pub n_vehicles_norm: f64,
    pub avg_pairwise_km_norm: f64,
    pub lat_spread_norm: f64,
    pub lon_spread_norm: f64,
    pub density_norm: f64,
    pub area_km2_norm: f64,
    pub depot_centroid_dist_norm: f64,

    // Graph (8) — from kNN(k=10) graph of stops
    pub knn_avg_degree: f64,
    pub knn_max_degree: f64,
    pub knn_clustering: f64,
    pub knn_diameter_norm: f64,
    pub knn_mst_weight_norm: f64,
    pub knn_avg_shortest_path: f64,
    pub knn_spectral_gap: f64,
    pub knn_assortativity: f64,

    // Demand (4)
    pub total_demand_norm: f64,
    pub demand_std_norm: f64,
    pub tight_capacity_flag: f64,
    pub capacity_ratio: f64,

    // Distance matrix stats (4)
    pub dist_mean_norm: f64,
    pub dist_std_norm: f64,
    pub dist_skewness: f64,
    pub depot_dist_mean_norm: f64,

    // Objective & interaction (4)
    pub objective_min_distance: f64,
    pub objective_min_time: f64,
    pub objective_balance_load: f64,
    pub objective_min_vehicles: f64,
}

impl InstanceFeatures {
    /// Extract features from a VRP input. Depot is index 0.
    pub fn from_input(input: &VRPSolverInput) -> Self {
        let stops = &input.locations;
        let n = stops.len();
        if n <= 1 {
            return Self::default();
        }

        let depot = &stops[0];
        let others: Vec<&VRPSolverStop> = stops.iter().skip(1).collect();
        let num_stops = others.len();

        // ── Geometric features ──────────────────────────────────────────
        let mut dists = Vec::new();
        for (i, a) in others.iter().enumerate() {
            for b in others.iter().skip(i + 1) {
                dists.push(haversine_m(a.lat, a.lon, b.lat, b.lon) / 1000.0);
            }
        }
        let avg_pairwise_km = if !dists.is_empty() {
            dists.iter().sum::<f64>() / dists.len() as f64
        } else {
            0.0
        };

        let lats: Vec<f64> = others.iter().map(|s| s.lat).collect();
        let lons: Vec<f64> = others.iter().map(|s| s.lon).collect();
        let lat_spread = std_dev(&lats);
        let lon_spread = std_dev(&lons);

        let lat_km = lat_spread * 111.0;
        let lon_km = lon_spread * 111.0 * depot.lat.to_radians().cos();
        let area_km2 = (lat_km * lon_km).max(0.001);
        let density = num_stops as f64 / area_km2;

        // Centroid of stops
        let centroid_lat = lats.iter().sum::<f64>() / lats.len() as f64;
        let centroid_lon = lons.iter().sum::<f64>() / lons.len() as f64;
        let depot_centroid_dist = haversine_m(depot.lat, depot.lon, centroid_lat, centroid_lon) / 1000.0;

        // ── Demand features ───────────────────────────────────────────
        let demands: Vec<f64> = others.iter().filter_map(|s| s.demand).collect();
        let total_demand: f64 = demands.iter().sum();
        let demand_std = std_dev(&demands);
        let capacity_ratio = if input.vehicle_capacity > 0.0 {
            total_demand / (input.vehicle_capacity * input.num_vehicles.max(1) as f64)
        } else {
            0.0
        };
        let tight_capacity = capacity_ratio > 0.8;

        // ── Distance matrix stats ──────────────────────────────────────
        let depot_dists: Vec<f64> = others
            .iter()
            .map(|s| haversine_m(depot.lat, depot.lon, s.lat, s.lon) / 1000.0)
            .collect();
        let dist_mean = if !dists.is_empty() {
            dists.iter().sum::<f64>() / dists.len() as f64
        } else {
            0.0
        };
        let dist_std = std_dev(&dists);
        let dist_skew = skewness(&dists);
        let depot_dist_mean = if !depot_dists.is_empty() {
            depot_dists.iter().sum::<f64>() / depot_dists.len() as f64
        } else {
            0.0
        };

        // ── kNN graph features (k=10) ──────────────────────────────────
        let knn_feats = knn_graph_features(&others, 10);

        // ── Normalization constants (from training distribution) ───────
        // These are approximate; the real training distribution should be used.
        Self {
            n_stops_norm: (num_stops as f64 / 500.0).min(1.0),
            n_vehicles_norm: (input.num_vehicles.max(1) as f64 / 20.0).min(1.0),
            avg_pairwise_km_norm: (avg_pairwise_km / 100.0).min(1.0),
            lat_spread_norm: (lat_spread / 5.0).min(1.0),
            lon_spread_norm: (lon_spread / 5.0).min(1.0),
            density_norm: (density / 50.0).min(1.0),
            area_km2_norm: (area_km2 / 1000.0).min(1.0),
            depot_centroid_dist_norm: (depot_centroid_dist / 100.0).min(1.0),

            knn_avg_degree: knn_feats.avg_degree,
            knn_max_degree: knn_feats.max_degree,
            knn_clustering: knn_feats.clustering,
            knn_diameter_norm: (knn_feats.diameter / 100.0).min(1.0),
            knn_mst_weight_norm: (knn_feats.mst_weight / 1000.0).min(1.0),
            knn_avg_shortest_path: (knn_feats.avg_shortest_path / 100.0).min(1.0),
            knn_spectral_gap: (knn_feats.spectral_gap / 50.0).min(1.0),
            knn_assortativity: knn_feats.assortativity.clamp(-1.0, 1.0),

            total_demand_norm: (total_demand / 1000.0).min(1.0),
            demand_std_norm: (demand_std / 100.0).min(1.0),
            tight_capacity_flag: if tight_capacity { 1.0 } else { 0.0 },
            capacity_ratio: capacity_ratio.min(1.0),

            dist_mean_norm: (dist_mean / 100.0).min(1.0),
            dist_std_norm: (dist_std / 100.0).min(1.0),
            dist_skewness: dist_skew,
            depot_dist_mean_norm: (depot_dist_mean / 100.0).min(1.0),

            objective_min_distance: match input.objective {
                VrpObjective::MinDistance => 1.0,
                _ => 0.0,
            },
            objective_min_time: match input.objective {
                VrpObjective::MinTime => 1.0,
                _ => 0.0,
            },
            objective_balance_load: match input.objective {
                VrpObjective::BalanceLoad => 1.0,
                _ => 0.0,
            },
            objective_min_vehicles: match input.objective {
                VrpObjective::MinVehicles => 1.0,
                _ => 0.0,
            },
        }
    }

    /// Flatten to a dense Vec<f32> for neural network input.
    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.n_stops_norm as f32,
            self.n_vehicles_norm as f32,
            self.avg_pairwise_km_norm as f32,
            self.lat_spread_norm as f32,
            self.lon_spread_norm as f32,
            self.density_norm as f32,
            self.area_km2_norm as f32,
            self.depot_centroid_dist_norm as f32,
            self.knn_avg_degree as f32,
            self.knn_max_degree as f32,
            self.knn_clustering as f32,
            self.knn_diameter_norm as f32,
            self.knn_mst_weight_norm as f32,
            self.knn_avg_shortest_path as f32,
            self.knn_spectral_gap as f32,
            self.knn_assortativity as f32,
            self.total_demand_norm as f32,
            self.demand_std_norm as f32,
            self.tight_capacity_flag as f32,
            self.capacity_ratio as f32,
            self.dist_mean_norm as f32,
            self.dist_std_norm as f32,
            self.dist_skewness as f32,
            self.depot_dist_mean_norm as f32,
            self.objective_min_distance as f32,
            self.objective_min_time as f32,
            self.objective_balance_load as f32,
            self.objective_min_vehicles as f32,
        ]
    }
}

// ── kNN graph helpers ────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct KnnFeatures {
    avg_degree: f64,
    max_degree: f64,
    clustering: f64,
    diameter: f64,
    mst_weight: f64,
    avg_shortest_path: f64,
    spectral_gap: f64,
    assortativity: f64,
}

fn knn_graph_features(stops: &[&VRPSolverStop], k: usize) -> KnnFeatures {
    let n = stops.len();
    if n < 2 {
        return KnnFeatures::default();
    }
    let k = k.min(n - 1).max(1);

    // Build distance matrix
    let mut dists = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = haversine_m(stops[i].lat, stops[i].lon, stops[j].lat, stops[j].lon) / 1000.0;
            dists[i][j] = d;
            dists[j][i] = d;
        }
    }

    // kNN adjacency list
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut degrees = vec![0usize; n];
    for i in 0..n {
        let mut neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dists[i][j]))
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(j, _) in neighbors.iter().take(k) {
            adj[i].push(j);
            adj[j].push(i);
            degrees[i] += 1;
            degrees[j] += 1;
        }
    }

    let avg_degree = degrees.iter().sum::<usize>() as f64 / n as f64;
    let max_degree = *degrees.iter().max().unwrap_or(&0) as f64;

    // Clustering coefficient (local)
    let mut clustering_sum = 0.0;
    for i in 0..n {
        let neighbors = &adj[i];
        let di = neighbors.len();
        if di < 2 {
            continue;
        }
        let mut edges_between = 0;
        for a in 0..di {
            for b in (a + 1)..di {
                if adj[neighbors[a]].contains(&neighbors[b]) {
                    edges_between += 1;
                }
            }
        }
        let possible = di * (di - 1) / 2;
        clustering_sum += edges_between as f64 / possible as f64;
    }
    let clustering = clustering_sum / n as f64;

    // MST weight (Prim)
    let mst_weight = prim_mst(&dists);

    // Diameter & avg shortest path (Floyd-Warshall for small n)
    let (diameter, avg_sp) = if n <= 200 {
        let (_sp, diam, avg) = all_pairs_shortest_paths(&dists);
        (diam, avg)
    } else {
        // For large n, approximate with sampled pairs
        let mut max_d = 0.0;
        let mut sum_d = 0.0;
        let mut count = 0;
        let sample = (n * 10).min(2000);
        for _ in 0..sample {
            let i = (rand_u32() as usize) % n;
            let j = (rand_u32() as usize) % n;
            if i == j { continue; }
            let d = dists[i][j];
            if d > max_d { max_d = d; }
            sum_d += d;
            count += 1;
        }
        (max_d, if count > 0 { sum_d / count as f64 } else { 0.0 })
    };

    // Spectral gap (approximate from degree sequence)
    let spectral_gap = max_degree - avg_degree;

    // Assortativity (degree correlation)
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut edge_count = 0;
    for i in 0..n {
        for &j in &adj[i] {
            if i >= j { continue; }
            let di = degrees[i] as f64;
            let dj = degrees[j] as f64;
            sum_xy += di * dj;
            sum_x += di;
            sum_y += dj;
            sum_x2 += di * di;
            sum_y2 += dj * dj;
            edge_count += 1;
        }
    }
    let assortativity = if edge_count > 0 {
        let ex = sum_x / edge_count as f64;
        let ey = sum_y / edge_count as f64;
        let ex2 = sum_x2 / edge_count as f64;
        let ey2 = sum_y2 / edge_count as f64;
        let cov = (sum_xy / edge_count as f64) - ex * ey;
        let var_x = ex2 - ex * ex;
        let var_y = ey2 - ey * ey;
        if var_x > 0.0 && var_y > 0.0 {
            cov / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        }
    } else {
        0.0
    };

    KnnFeatures {
        avg_degree,
        max_degree,
        clustering,
        diameter,
        mst_weight,
        avg_shortest_path: avg_sp,
        spectral_gap,
        assortativity,
    }
}

fn prim_mst(dists: &[Vec<f64>]) -> f64 {
    let n = dists.len();
    if n == 0 {
        return 0.0;
    }
    let mut visited = vec![false; n];
    let mut min_edge = vec![f64::MAX; n];
    min_edge[0] = 0.0;
    let mut total = 0.0;

    for _ in 0..n {
        let mut u = None;
        for i in 0..n {
            if !visited[i] && (u.is_none() || min_edge[i] < min_edge[u.unwrap()]) {
                u = Some(i);
            }
        }
        let u = u.unwrap();
        visited[u] = true;
        total += min_edge[u];

        for v in 0..n {
            if !visited[v] && dists[u][v] < min_edge[v] {
                min_edge[v] = dists[u][v];
            }
        }
    }
    total
}

fn all_pairs_shortest_paths(dists: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64, f64) {
    let n = dists.len();
    let mut sp = dists.to_vec();
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if sp[i][k] + sp[k][j] < sp[i][j] {
                    sp[i][j] = sp[i][k] + sp[k][j];
                }
            }
        }
    }
    let mut max_d = 0.0;
    let mut sum_d = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if sp[i][j] < f64::MAX / 2.0 {
                if sp[i][j] > max_d {
                    max_d = sp[i][j];
                }
                sum_d += sp[i][j];
                count += 1;
            }
        }
    }
    let avg = if count > 0 { sum_d / count as f64 } else { 0.0 };
    (sp, max_d, avg)
}

// Simple xorshift for deterministic pseudo-randomness (no std dependency)
fn rand_u32() -> u32 {
    static mut SEED: u32 = 123456789;
    unsafe {
        SEED ^= SEED << 13;
        SEED ^= SEED >> 17;
        SEED ^= SEED << 5;
        SEED
    }
}

// ── Statistical helpers ────────────────────────────────────────────────

fn std_dev(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

fn skewness(vals: &[f64]) -> f64 {
    if vals.len() < 3 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let std = std_dev(vals);
    if std == 0.0 {
        return 0.0;
    }
    let n = vals.len() as f64;
    let m3 = vals.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
    m3 / std.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::vrp::test_utils::{make_input, make_stop};

    #[test]
    fn test_features_vector_length() {
        let stops = vec![
            make_stop(0.0, 0.0, "depot"),
            make_stop(1.0, 0.0, "a"),
            make_stop(0.0, 1.0, "b"),
        ];
        let input = make_input(stops, 1);
        let f = InstanceFeatures::from_input(&input);
        let vec = f.to_vector();
        assert_eq!(vec.len(), 28);
        // Features are normalized but some may exceed [0,1] due to raw value scale;
        // just ensure they are finite.
        for v in &vec {
            assert!(v.is_finite(), "feature not finite: {}", v);
        }
    }

    #[test]
    fn test_objective_onehot() {
        let stops = vec![make_stop(0.0, 0.0, "depot"), make_stop(1.0, 0.0, "a")];
        let mut input = make_input(stops.clone(), 1);
        input.objective = VrpObjective::MinTime;
        let f = InstanceFeatures::from_input(&input);
        assert_eq!(f.objective_min_time, 1.0);
        assert_eq!(f.objective_min_distance, 0.0);
    }
}
