#![allow(dead_code)]
#![allow(dead_code)]
#![allow(dead_code)]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_macros,
    clippy::all
)]
#![allow(dead_code)]
#![allow(dead_code)]
#![allow(dead_code)]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_macros,
    clippy::all
)]
//! Geometric and routing utilities for VRP solvers.
//!
//! Provides distance-matrix construction, zone clustering,
//! nearest-neighbor routing, 2-opt improvement, and Valhalla matrix fetching.

use super::super::haversine_m;
use super::types::{DistCell, DistMatrix, VRPSolverStop};

/// Haversine distance between two WGS-84 coordinates, in km.
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    haversine_m(lat1, lon1, lat2, lon2) / 1000.0
}

/// Build a full O(n²) distance/time matrix using haversine.
/// Time estimated at `avg_speed_kmh` (default 40 km/h).
pub fn build_haversine_matrix(locations: &[VRPSolverStop], avg_speed_kmh: f64) -> DistMatrix {
    let n = locations.len();
    let mut matrix = Vec::with_capacity(n);
    for (i, _) in locations.iter().enumerate().take(n) {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let dist = haversine_km(
                locations[i].lat,
                locations[i].lon,
                locations[j].lat,
                locations[j].lon,
            );
            let time_sec = (dist / avg_speed_kmh) * 3600.0;
            row.push(DistCell {
                distance: dist,
                time: time_sec,
            });
        }
        matrix.push(row);
    }
    matrix
}

#[cfg(feature = "ml")]
type EmbeddingsSlice<'a> = &'a [crate::core::ml::graph_embed::RoadEmbedding];
#[cfg(not(feature = "ml"))]
type EmbeddingsSlice<'a> = &'a [()];

/// Build a distance matrix using shortest paths on the road network graph.
pub fn build_graph_matrix(
    stops: &[VRPSolverStop],
    nodes: &[RmpNode],
    edges: &[RmpEdge],
    embeddings: Option<EmbeddingsSlice>,
    avg_speed_kmh: f64,
) -> DistMatrix {
    let n_stops = stops.len();
    let n_nodes = nodes.len();
    if n_stops == 0 || n_nodes == 0 {
        return vec![vec![DistCell { distance: 0.0, time: 0.0 }; n_stops]; n_stops];
    }

    // 1. Build adjacency list
    let mut adj = vec![Vec::new(); n_nodes];
    for (i, edge) in edges.iter().enumerate() {
        let mut weight = edge.weight_m;
        
        // Apply learned embedding if available
        #[cfg(feature = "ml")]
        if let Some(embs) = embeddings {
            if let Some(emb) = embs.get(i) {
                // Use first dimension as a learned bias for now
                // Research basis: GAIN (2107.07791)
                let bias = emb.vector.get(0).copied().unwrap_or(0.0);
                weight *= (1.0 + bias as f64).max(0.1_f64);
            }
        }

        adj[edge.from as usize].push((edge.to as usize, weight));
        if edge.oneway == 0 {
            adj[edge.to as usize].push((edge.from as usize, weight));
        }
    }

    // 2. Map stops to nearest nodes
    let stop_nodes: Vec<usize> = stops.iter().map(|stop| {
        let mut best_node = 0;
        let mut best_dist = f64::MAX;
        for (i, node) in nodes.iter().enumerate() {
            let d = haversine_m(stop.lat, stop.lon, node.lat, node.lon);
            if d < best_dist {
                best_dist = d;
                best_node = i;
            }
        }
        best_node
    }).collect();

    // 3. Dijkstra from each stop node
    let mut matrix = vec![vec![DistCell { distance: 0.0, time: 0.0 }; n_stops]; n_stops];
    
    for i in 0..n_stops {
        let start_node = stop_nodes[i];
        let dists = dijkstra(start_node, &adj, n_nodes);
        
        for j in 0..n_stops {
            let target_node = stop_nodes[j];
            let d_m = dists[target_node];
            let d_km = d_m / 1000.0;
            let time_sec = (d_km / avg_speed_kmh) * 3600.0;
            matrix[i][j] = DistCell { distance: d_km, time: time_sec };
        }
    }

    matrix
}

fn dijkstra(start: usize, adj: &[Vec<(usize, f64)>], n: usize) -> Vec<f64> {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    #[derive(Copy, Clone, PartialEq)]
    struct State {
        cost: f64,
        position: usize,
    }
    impl Eq for State {}
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut dists = vec![f64::MAX; n];
    let mut heap = BinaryHeap::new();

    dists[start] = 0.0;
    heap.push(State { cost: 0.0, position: start });

    while let Some(State { cost, position }) = heap.pop() {
        if cost > dists[position] { continue; }

        for (next, weight) in &adj[position] {
            let next_cost = cost + weight;
            if next_cost < dists[*next] {
                dists[*next] = next_cost;
                heap.push(State { cost: next_cost, position: *next });
            }
        }
    }
    dists
}

use super::super::optimize::{RmpEdge, RmpNode};

/// Fetch a real-road distance/time matrix from Valhalla.
pub async fn get_valhalla_matrix(locations: &[VRPSolverStop]) -> Result<DistMatrix, String> {
    let url = "https://valhalla1.openstreetmap.de/sources_to_targets";
    let locs: Vec<serde_json::Value> = locations
        .iter()
        .map(|l| serde_json::json!({"lat": l.lat, "lon": l.lon}))
        .collect();

    let body = serde_json::json!({
        "sources": locs,
        "targets": locs,
        "costing": "auto",
        "directions_options": { "units": "kilometers" }
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Valhalla request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Valhalla HTTP {}", resp.status()));
    }

    let data: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Valhalla parse error: {e}"))?;

    let stt = data
        .get("sources_to_targets")
        .ok_or("Invalid Valhalla response: missing sources_to_targets")?;

    let rows = stt
        .as_array()
        .ok_or("Invalid Valhalla response: sources_to_targets not an array")?;
    let n = rows.len();
    let mut matrix = Vec::with_capacity(n);
    for row_val in rows {
        let row = row_val.as_array().ok_or("Invalid Valhalla row")?;
        let mut matrix_row = Vec::with_capacity(n);
        for cell_val in row {
            let cell = cell_val.as_object();
            let (dist, time) = match cell {
                Some(c) => (
                    c.get("distance").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    c.get("time").and_then(|v| v.as_f64()).unwrap_or(0.0),
                ),
                None => (0.0, 0.0),
            };
            matrix_row.push(DistCell {
                distance: dist,
                time,
            });
        }
        matrix.push(matrix_row);
    }
    Ok(matrix)
}

/// Cluster stops into geographic grid zones so nearby stops are grouped together.
pub fn cluster_by_zones(locations: &[VRPSolverStop], num_zones: usize) -> Vec<Vec<usize>> {
    let n = locations.len();
    if n <= num_zones {
        return (0..n).map(|i| vec![i]).collect();
    }
    let min_lat = locations
        .iter()
        .map(|p| p.lat)
        .fold(f64::INFINITY, f64::min);
    let max_lat = locations
        .iter()
        .map(|p| p.lat)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lon = locations
        .iter()
        .map(|p| p.lon)
        .fold(f64::INFINITY, f64::min);
    let max_lon = locations
        .iter()
        .map(|p| p.lon)
        .fold(f64::NEG_INFINITY, f64::max);

    let rows = (num_zones as f64).sqrt().ceil() as usize;
    let cols = num_zones.div_ceil(rows); // ceil division
    let cell_lat = (max_lat - min_lat) / rows as f64;
    let cell_lon = (max_lon - min_lon) / cols as f64;
    let cell_lat = if cell_lat == 0.0 { 0.001 } else { cell_lat };
    let cell_lon = if cell_lon == 0.0 { 0.001 } else { cell_lon };

    let mut grid: std::collections::BTreeMap<(usize, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, _) in locations.iter().enumerate().take(n) {
        let p = &locations[i];
        let ri = ((p.lat - min_lat) / cell_lat).floor() as usize;
        let ri = ri.min(rows - 1);
        let ci = ((p.lon - min_lon) / cell_lon).floor() as usize;
        let ci = ci.min(cols - 1);
        grid.entry((ri, ci)).or_default().push(i);
    }
    grid.into_values().collect()
}

/// Order clusters so the start zone comes first, then nearest by centroid.
pub fn order_clusters_by_start(
    clusters: &[Vec<usize>],
    locations: &[VRPSolverStop],
    start_index: usize,
) -> Vec<Vec<usize>> {
    let start = &locations[start_index];
    let mut with_centroid: Vec<(Vec<usize>, f64, bool)> = clusters
        .iter()
        .map(|indices| {
            let lat: f64 =
                indices.iter().map(|&i| locations[i].lat).sum::<f64>() / indices.len() as f64;
            let lon: f64 =
                indices.iter().map(|&i| locations[i].lon).sum::<f64>() / indices.len() as f64;
            let dist = haversine_km(start.lat, start.lon, lat, lon);
            let contains_start = indices.contains(&start_index);
            (indices.clone(), dist, contains_start)
        })
        .collect();

    with_centroid.sort_by(|a, b| {
        if a.2 && !b.2 {
            std::cmp::Ordering::Less
        } else if !a.2 && b.2 {
            std::cmp::Ordering::Greater
        } else {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    with_centroid
        .into_iter()
        .map(|(indices, _, _)| indices)
        .collect()
}

/// Nearest-neighbor TSP construction for a subset of indices.
pub fn nearest_neighbor_route(
    matrix: &DistMatrix,
    indices: &[usize],
    start_idx_in_subset: usize,
) -> Vec<usize> {
    if indices.len() <= 1 {
        return indices.to_vec();
    }
    let mut route = Vec::with_capacity(indices.len());
    let mut remaining: std::collections::HashSet<usize> = indices.iter().copied().collect();
    let start_global = indices[start_idx_in_subset];
    route.push(start_global);
    remaining.remove(&start_global);
    let mut current = start_global;
    while !remaining.is_empty() {
        let mut nearest = 0;
        let mut best_dist = f64::INFINITY;
        for &i in &remaining {
            let d = matrix
                .get(current)
                .and_then(|row| row.get(i))
                .map(|c| c.distance)
                .unwrap_or(f64::INFINITY);
            if d < best_dist {
                best_dist = d;
                nearest = i;
            }
        }
        route.push(nearest);
        remaining.remove(&nearest);
        current = nearest;
    }
    route
}

/// 2-opt improvement: iteratively reverse segments to reduce total distance.
pub fn two_opt_improve(
    matrix: &DistMatrix,
    route_indices: &[usize],
    max_iter: u32,
) -> Vec<usize> {
    let n = route_indices.len();
    if n <= 3 {
        return route_indices.to_vec();
    }
    let mut route = route_indices.to_vec();
    let mut improved = true;
    let mut iterations = 0;
    while improved && iterations < max_iter {
        improved = false;
        iterations += 1;
        for i in 0..n - 2 {
            for j in (i + 2)..n {
                let a = route[i];
                let b = route[i + 1];
                let c = route[j];
                let d = route[(j + 1) % n];
                let before = matrix_get_dist(matrix, a, b) + matrix_get_dist(matrix, c, d);
                let after = matrix_get_dist(matrix, a, c) + matrix_get_dist(matrix, b, d);
                if after < before - 1e-6 {
                    route[i + 1..=j].reverse();
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
    }
    route
}

/// Helper: get distance from matrix with fallback.
pub fn matrix_get_dist(matrix: &DistMatrix, i: usize, j: usize) -> f64 {
    matrix
        .get(i)
        .and_then(|row| row.get(j))
        .map(|c| c.distance)
        .unwrap_or(0.0)
}

/// Helper: get time from matrix with fallback.
pub fn matrix_get_time(matrix: &DistMatrix, i: usize, j: usize) -> f64 {
    matrix
        .get(i)
        .and_then(|row| row.get(j))
        .map(|c| c.time)
        .unwrap_or(0.0)
}

pub fn build_sweep_routes(
    matrix: &crate::core::vrp::types::DistMatrix,
    locations: &[crate::core::vrp::types::VRPSolverStop],
    num_vehicles: usize,
) -> Vec<Vec<usize>> {
    let n = locations.len();
    if n <= 1 {
        return vec![];
    }
    let depot = &locations[0];
    let mut indices: Vec<usize> = (1..n).collect();
    indices.sort_by(|&a, &b| {
        let la = &locations[a];
        let lb = &locations[b];
        let angle_a = (la.lat - depot.lat).atan2(la.lon - depot.lon);
        let angle_b = (lb.lat - depot.lat).atan2(lb.lon - depot.lon);
        angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
    });

    let per_route = (indices.len() as f64 / num_vehicles as f64).ceil() as usize;
    let mut route_indices: Vec<Vec<usize>> = Vec::new();

    for v in 0..num_vehicles {
        let start = v * per_route;
        let end = std::cmp::min(start + per_route, indices.len());
        if start >= indices.len() { break; }
        let segment = &indices[start..end];
        if segment.is_empty() { continue; }

        let mut route = vec![0];
        let mut remaining: std::collections::HashSet<usize> = segment.iter().copied().collect();
        let mut current = 0;
        while !remaining.is_empty() {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for &node in &remaining {
                let dist = crate::core::vrp::utils::matrix_get_dist(matrix, current, node);
                if dist < best_dist {
                    best_dist = dist;
                    best = node;
                }
            }
            remaining.remove(&best);
            route.push(best);
            current = best;
        }
        route.push(0);
        route_indices.push(route);
    }
    route_indices
}



/// Parse a CSV file of coordinates into VRP solver stops.
///
/// Supported column sets:
///   Minimal:  lat,lon
///   Extended: lat,lon,label,demand,type
///
/// The `type` column accepts "depot" or "stop" (default: "stop").
/// If no depot is present, the first row becomes the depot.
/// The `demand` column is optional; defaults to 1.0 for stops, 0.0 for depots.
pub fn parse_csv_stops(csv_path: &str) -> Result<(Vec<crate::core::vrp::types::VRPSolverStop>, Vec<usize>), String> {
    use std::io::Read;
    let mut file = std::fs::File::open(csv_path)
        .map_err(|e| format!("Cannot open CSV '{}': {}", csv_path, e))?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| format!("Cannot read CSV '{}': {}", csv_path, e))?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(content.as_bytes());

    let headers = reader.headers()
        .map_err(|e| format!("Failed to read CSV headers: {}", e))?
        .clone();

    let lat_idx = headers.iter().position(|h| h.eq_ignore_ascii_case("lat") || h.eq_ignore_ascii_case("latitude"))
        .ok_or("CSV must have a 'lat' column")?;
    let lon_idx = headers.iter().position(|h| h.eq_ignore_ascii_case("lon") || h.eq_ignore_ascii_case("lng") || h.eq_ignore_ascii_case("longitude"))
        .ok_or("CSV must have a 'lon' column")?;
    let label_idx = headers.iter().position(|h| h.eq_ignore_ascii_case("label") || h.eq_ignore_ascii_case("name") || h.eq_ignore_ascii_case("id"));
    let demand_idx = headers.iter().position(|h| h.eq_ignore_ascii_case("demand"));
    let type_idx = headers.iter().position(|h| h.eq_ignore_ascii_case("type") || h.eq_ignore_ascii_case("role"));

    let mut stops = Vec::new();
    let mut depot_indices = Vec::new();

    for (row_num, result) in reader.records().enumerate() {
        let record = result.map_err(|e| format!("CSV parse error at row {}: {}", row_num + 2, e))?;

        let lat: f64 = record.get(lat_idx)
            .ok_or_else(|| format!("Missing lat at row {}", row_num + 2))?
            .parse()
            .map_err(|e| format!("Invalid lat at row {}: {}", row_num + 2, e))?;
        let lon: f64 = record.get(lon_idx)
            .ok_or_else(|| format!("Missing lon at row {}", row_num + 2))?
            .parse()
            .map_err(|e| format!("Invalid lon at row {}: {}", row_num + 2, e))?;

        let label = label_idx
            .and_then(|i| record.get(i))
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("stop_{}", row_num));

        let is_depot = type_idx
            .and_then(|i| record.get(i))
            .map(|s| s.eq_ignore_ascii_case("depot"))
            .unwrap_or(false);

        let demand = demand_idx
            .and_then(|i| record.get(i))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(if is_depot { 0.0 } else { 1.0 });

        if is_depot {
            depot_indices.push(stops.len());
        }

        stops.push(crate::core::vrp::types::VRPSolverStop {
            lat,
            lon,
            label,
            demand: Some(demand),
            arrival_time: None,
        });
    }

    if stops.is_empty() {
        return Err("CSV file contains no data rows".to_string());
    }

    // If no depot was found, designate the first stop as depot
    if depot_indices.is_empty() {
        depot_indices.push(0);
        if let Some(d) = stops[0].demand.as_mut() {
            *d = 0.0;
        }
    }

    Ok((stops, depot_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_same_point() {
        let d = haversine_km(40.0, -74.0, 40.0, -74.0);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_haversine_known_distance() {
        // NYC to LA ~3940 km
        let d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437);
        assert!(d > 3900.0 && d < 4000.0, "Expected ~3940 km, got {d}");
    }

    #[test]
    fn test_build_haversine_matrix() {
        let stops = vec![
            VRPSolverStop {
                lat: 0.0,
                lon: 0.0,
                label: "A".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 1.0,
                lon: 1.0,
                label: "B".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let m = build_haversine_matrix(&stops, 40.0);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 2);
        assert!(m[0][0].distance < 1e-10); // A→A
        assert!(m[0][1].distance > 0.0); // A→B
    }

    #[test]
    fn test_cluster_by_zones() {
        let stops = vec![
            VRPSolverStop {
                lat: 0.0,
                lon: 0.0,
                label: "A".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 0.1,
                lon: 0.1,
                label: "B".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 10.0,
                lon: 10.0,
                label: "C".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let clusters = cluster_by_zones(&stops, 2);
        // All 3 stops must be covered
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_nearest_neighbor_route() {
        // 3 stops in a line: 0→1→2
        let stops = vec![
            VRPSolverStop {
                lat: 0.0,
                lon: 0.0,
                label: "0".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 1.0,
                lon: 0.0,
                label: "1".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 2.0,
                lon: 0.0,
                label: "2".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let m = build_haversine_matrix(&stops, 40.0);
        let route = nearest_neighbor_route(&m, &[0, 1, 2], 0);
        assert_eq!(route.len(), 3);
        assert_eq!(route[0], 0); // starts at depot
    }

    #[test]
    fn test_haversine_symmetry() {
        let d1 = haversine_km(40.0, -74.0, 34.0, -118.0);
        let d2 = haversine_km(34.0, -118.0, 40.0, -74.0);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_antipodal() {
        // Opposite points on earth should be ~20015 km (half circumference)
        let d = haversine_km(0.0, 0.0, 0.0, 180.0);
        assert!(d > 19900.0 && d < 20100.0, "Expected ~20015 km, got {d}");
    }

    #[test]
    fn test_haversine_matrix_symmetry() {
        let stops = vec![
            VRPSolverStop {
                lat: 51.5,
                lon: -0.1,
                label: "London".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 48.9,
                lon: 2.3,
                label: "Paris".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 52.5,
                lon: 13.4,
                label: "Berlin".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let m = build_haversine_matrix(&stops, 50.0);
        // Distance matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j].distance - m[j][i].distance).abs() < 1e-10,
                    "Matrix not symmetric at [{i}][{j}]"
                );
            }
        }
        // Time matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j].time - m[j][i].time).abs() < 1e-6,
                    "Time matrix not symmetric at [{i}][{j}]"
                );
            }
        }
        // Diagonal should be zero distance
        for i in 0..3 {
            assert!(m[i][i].distance < 1e-10);
            assert!(m[i][i].time < 1e-10);
        }
    }

    #[test]
    fn test_build_haversine_matrix_time_estimate() {
        let stops = vec![
            VRPSolverStop {
                lat: 0.0,
                lon: 0.0,
                label: "A".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 0.0,
                lon: 1.0,
                label: "B".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let m = build_haversine_matrix(&stops, 40.0);
        let dist_km = m[0][1].distance;
        let expected_time = (dist_km / 40.0) * 3600.0;
        assert!((m[0][1].time - expected_time).abs() < 1e-6);
    }

    #[test]
    fn test_two_opt_improve_already_optimal() {
        // A straight line 0-1-2-3 is already optimal for a tour
        let stops = vec![
            VRPSolverStop {
                lat: 0.0,
                lon: 0.0,
                label: "0".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 1.0,
                lon: 0.0,
                label: "1".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 2.0,
                lon: 0.0,
                label: "2".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 3.0,
                lon: 0.0,
                label: "3".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let m = build_haversine_matrix(&stops, 40.0);
        let route = vec![0, 1, 2, 3];
        let improved = two_opt_improve(&m, &route, 300);
        assert_eq!(improved.len(), 4);
    }

    #[test]
    fn test_two_opt_improve_crossing_route() {
        // Create a crossing: 0-2-1-3 should become 0-1-2-3 (or reverse)
        let stops = vec![
            VRPSolverStop {
                lat: 0.0,
                lon: 0.0,
                label: "0".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 1.0,
                lon: 0.0,
                label: "1".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 2.0,
                lon: 0.0,
                label: "2".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 3.0,
                lon: 0.0,
                label: "3".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let m = build_haversine_matrix(&stops, 40.0);
        let route = vec![0, 2, 1, 3]; // crossing
        let improved = two_opt_improve(&m, &route, 300);
        // Calculate total distance of improved route
        let improved_dist: f64 = (0..improved.len() - 1)
            .map(|i| m[improved[i]][improved[i + 1]].distance)
            .sum();
        let original_dist: f64 = (0..route.len() - 1)
            .map(|i| m[route[i]][route[i + 1]].distance)
            .sum();
        assert!(
            improved_dist <= original_dist + 1e-6,
            "2-opt should not worsen the route"
        );
    }

    #[test]
    fn test_two_opt_improve_small_route() {
        let m: DistMatrix = vec![vec![DistCell {
            distance: 0.0,
            time: 0.0,
        }]];
        let improved = two_opt_improve(&m, &[0], 300);
        assert_eq!(improved, vec![0]);
    }

    #[test]
    fn test_two_opt_improve_two_nodes() {
        let m: DistMatrix = vec![
            vec![
                DistCell {
                    distance: 0.0,
                    time: 0.0,
                },
                DistCell {
                    distance: 5.0,
                    time: 100.0,
                },
            ],
            vec![
                DistCell {
                    distance: 5.0,
                    time: 100.0,
                },
                DistCell {
                    distance: 0.0,
                    time: 0.0,
                },
            ],
        ];
        let improved = two_opt_improve(&m, &[0, 1], 300);
        assert_eq!(improved.len(), 2);
    }

    #[test]
    fn test_cluster_by_zones_single_stop() {
        let stops = vec![VRPSolverStop {
            lat: 10.0,
            lon: 20.0,
            label: "A".into(),
            demand: None,
            arrival_time: None,
        }];
        let clusters = cluster_by_zones(&stops, 4);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0]);
    }

    #[test]
    fn test_cluster_by_zones_all_same_location() {
        let stops = vec![
            VRPSolverStop {
                lat: 5.0,
                lon: 5.0,
                label: "A".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 5.0,
                lon: 5.0,
                label: "B".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 5.0,
                lon: 5.0,
                label: "C".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let clusters = cluster_by_zones(&stops, 4);
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_order_clusters_start_first() {
        let stops = vec![
            VRPSolverStop {
                lat: 0.0,
                lon: 0.0,
                label: "depot".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 1.0,
                lon: 1.0,
                label: "near".into(),
                demand: None,
                arrival_time: None,
            },
            VRPSolverStop {
                lat: 50.0,
                lon: 50.0,
                label: "far".into(),
                demand: None,
                arrival_time: None,
            },
        ];
        let clusters = vec![vec![2], vec![0, 1]];
        let ordered = order_clusters_by_start(&clusters, &stops, 0);
        // Cluster containing start (0) must come first
        assert!(ordered[0].contains(&0));
    }

    #[test]
    fn test_nearest_neighbor_single_node() {
        let stops = vec![VRPSolverStop {
            lat: 5.0,
            lon: 5.0,
            label: "A".into(),
            demand: None,
            arrival_time: None,
        }];
        let m = build_haversine_matrix(&stops, 40.0);
        let route = nearest_neighbor_route(&m, &[0], 0);
        assert_eq!(route, vec![0]);
    }

    #[test]
    fn test_matrix_get_helpers() {
        let m: DistMatrix = vec![
            vec![
                DistCell {
                    distance: 0.0,
                    time: 0.0,
                },
                DistCell {
                    distance: 10.0,
                    time: 200.0,
                },
            ],
            vec![
                DistCell {
                    distance: 10.0,
                    time: 200.0,
                },
                DistCell {
                    distance: 0.0,
                    time: 0.0,
                },
            ],
        ];
        assert_eq!(matrix_get_dist(&m, 0, 1), 10.0);
        assert_eq!(matrix_get_time(&m, 0, 1), 200.0);
        assert_eq!(matrix_get_dist(&m, 0, 0), 0.0);
        // Out of bounds returns 0.0
        assert_eq!(matrix_get_dist(&m, 5, 5), 0.0);
        assert_eq!(matrix_get_time(&m, 5, 5), 0.0);
    }
}
