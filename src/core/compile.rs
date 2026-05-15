use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

use super::clean::{clean_geojson, CleanOptions};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileRequest {
    pub input_geojson: String,
    pub output_rmp: String,
    pub compress: bool,
    pub road_classes: Vec<String>,
    pub clean_options: Option<CleanOptions>,
    #[serde(default)]
    pub prune_disconnected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    pub input_size_bytes: u64,
    pub output_size_bytes: u64,
    pub node_count: usize,
    pub edge_count: usize,
    pub elapsed_ms: u64,
}

/// .rmp binary format header (magic + version).
const RMP_MAGIC: &[u8; 4] = b"RMP1";

/// Compile a GeoJSON road network into the .rmp binary format (filesystem).
pub fn run_compile(req: &CompileRequest) -> anyhow::Result<CompileResult> {
    let start = Instant::now();

    let mut input_data = Vec::new();
    {
        let mut file = std::fs::File::open(&req.input_geojson)
            .with_context(|| format!("Failed to open input GeoJSON: {}", req.input_geojson))?;
        file.read_to_end(&mut input_data)?;
    }
    let input_size_bytes = input_data.len() as u64;

    let mut geojson: geojson::FeatureCollection = serde_json::from_slice(&input_data)
        .with_context(|| "Failed to parse GeoJSON FeatureCollection")?;

    if let Some(ref clean_opts) = req.clean_options {
        let (cleaned_fc, _stats, _warnings) =
            clean_geojson(&geojson, clean_opts).with_context(|| "Failed to clean GeoJSON")?;
        geojson = cleaned_fc;
    }

    let (buf, node_count, edge_count) = build_rmp_buffer(&geojson, req.prune_disconnected)?;

    // Write output file
    std::fs::write(&req.output_rmp, &buf)
        .with_context(|| format!("Failed to write output file: {}", req.output_rmp))?;

    let output_size_bytes = buf.len() as u64;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(CompileResult {
        input_size_bytes,
        output_size_bytes,
        node_count,
        edge_count,
        elapsed_ms,
    })
}

/// Build a `.rmp` binary buffer from a parsed GeoJSON FeatureCollection.
///
/// Shared by `run_compile` (and the in-memory compile path).
fn build_rmp_buffer(
    geojson: &geojson::FeatureCollection,
    prune_disconnected: bool,
) -> anyhow::Result<(Vec<u8>, usize, usize)> {
    let mut node_map: HashMap<u64, u32> = HashMap::new();
    let mut nodes: Vec<(f64, f64)> = Vec::new();
    let mut edges: Vec<(u32, u32, f64, u8)> = Vec::new();

    for feature in &geojson.features {
        let geometry = match feature.geometry.as_ref() {
            Some(g) => g,
            None => continue,
        };

        let oneway = feature
            .properties
            .as_ref()
            .and_then(|props| props.get("oneway"))
            .and_then(|v| v.as_str())
            .map(|s| {
                if matches!(s, "yes" | "1" | "true") {
                    1u8
                } else {
                    0u8
                }
            })
            .unwrap_or(0);

        let line_strings: Vec<&Vec<Vec<f64>>> = match &geometry.value {
            geojson::Value::LineString(coords) => vec![coords],
            geojson::Value::MultiLineString(multi) => multi.iter().collect(),
            _ => continue,
        };

        for coords in line_strings {
            if coords.len() < 2 {
                continue;
            }

            let coord_points: Vec<(f64, f64)> = coords
                .iter()
                .filter(|p| p.len() >= 2)
                .map(|p| (p[1], p[0]))
                .collect();

            if coord_points.len() < 2 {
                continue;
            }

            let mut last_node_id = None;
            for i in 0..coord_points.len() - 1 {
                let (lat1, lon1) = coord_points[i];
                let (lat2, lon2) = coord_points[i + 1];

                let from_node = match last_node_id {
                    Some(id) => id,
                    None => get_or_create_node(&mut node_map, &mut nodes, lat1, lon1),
                };
                let to_node = get_or_create_node(&mut node_map, &mut nodes, lat2, lon2);
                last_node_id = Some(to_node);

                let weight_m = super::haversine_m(lat1, lon1, lat2, lon2);
                edges.push((from_node, to_node, weight_m, oneway));
            }
        }
    }

    // Prune disconnected subgraphs
    if prune_disconnected && !nodes.is_empty() {
        prune_disconnected_components(&mut nodes, &mut edges);
    }

    let node_count = nodes.len();
    let edge_count = edges.len();

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(RMP_MAGIC);
    buf.extend_from_slice(&(node_count as u32).to_le_bytes());
    buf.extend_from_slice(&(edge_count as u32).to_le_bytes());

    for (lat, lon) in &nodes {
        buf.extend_from_slice(&lat.to_le_bytes());
        buf.extend_from_slice(&lon.to_le_bytes());
    }

    for (from, to, weight_m, oneway) in &edges {
        buf.extend_from_slice(&from.to_le_bytes());
        buf.extend_from_slice(&to.to_le_bytes());
        buf.extend_from_slice(&weight_m.to_le_bytes());
        buf.push(*oneway);
    }

    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    Ok((buf, node_count, edge_count))
}

/// Prune disconnected subgraphs, keeping only the largest component.
fn prune_disconnected_components(
    nodes: &mut Vec<(f64, f64)>,
    edges: &mut Vec<(u32, u32, f64, u8)>,
) {
    let n = nodes.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(from, to, _, _) in edges.iter() {
        adj[from as usize].push(to);
        adj[to as usize].push(from);
    }

    let mut visited = vec![false; n];
    let mut components: Vec<Vec<u32>> = Vec::new();

    for i in 0..n {
        if !visited[i] {
            let mut comp = Vec::new();
            let mut stack = vec![i as u32];
            visited[i] = true;
            while let Some(node) = stack.pop() {
                comp.push(node);
                for &neighbor in &adj[node as usize] {
                    if !visited[neighbor as usize] {
                        visited[neighbor as usize] = true;
                        stack.push(neighbor);
                    }
                }
            }
            components.push(comp);
        }
    }

    if components.len() <= 1 {
        return;
    }

    components.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let largest = &components[0];

    let mut old_to_new = vec![None; n];
    let mut new_nodes = Vec::with_capacity(largest.len());
    for &old_id in largest {
        old_to_new[old_id as usize] = Some(new_nodes.len() as u32);
        new_nodes.push(nodes[old_id as usize]);
    }

    let mut new_edges = Vec::new();
    for &(from, to, weight, oneway) in edges.iter() {
        if let (Some(nf), Some(nt)) = (old_to_new[from as usize], old_to_new[to as usize]) {
            new_edges.push((nf, nt, weight, oneway));
        }
    }

    *nodes = new_nodes;
    *edges = new_edges;
}

/// Quick validation: check if a file starts with the RMP magic bytes.
#[allow(dead_code)]
pub fn is_rmp_file(data: &[u8]) -> bool {
    data.len() >= 4 && &data[..4] == RMP_MAGIC
}

/// Get or create a node ID for the given (lat, lon) coordinates.
fn get_or_create_node(
    node_map: &mut HashMap<u64, u32>,
    nodes: &mut Vec<(f64, f64)>,
    lat: f64,
    lon: f64,
) -> u32 {
    let lat_i = (lat * 1e6) as i32;
    let lon_i = (lon * 1e6) as i32;
    let key = ((lat_i as u64) << 32) | (lon_i as u32 as u64);

    *node_map.entry(key).or_insert_with(|| {
        let id = nodes.len() as u32;
        nodes.push((lat, lon));
        id
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_compile_error_on_missing_input() {
        let req = CompileRequest {
            input_geojson: "non_existent_file.geojson".to_string(),
            output_rmp: "output.rmp".to_string(),
            compress: false,
            road_classes: vec![],
            clean_options: None,
            prune_disconnected: false,
        };
        let result = run_compile(&req);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to open input GeoJSON"));
    }
}
