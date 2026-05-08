use anyhow::Result;
use geojson::{Feature, FeatureCollection, Geometry, Value};
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use serde_json::{Map, Value as JsonValue};
use std::collections::{HashMap, HashSet};

use super::super::clean::{haversine_m, node_id};

#[derive(Debug, Clone)]
pub struct Node {
    pub lon: f64,
    pub lat: f64,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub length_m: f64,
    pub coords: Vec<[f64; 2]>,
    pub properties: Map<String, JsonValue>,
}

pub type RoadGraph = UnGraph<Node, Edge>;

/// Build graph from repaired features
pub fn build_graph(features: &[Feature], decimals: u32) -> Result<RoadGraph> {
    let mut graph = RoadGraph::new_undirected();
    let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

    for feature in features {
        let geom = match &feature.geometry {
            Some(g) => g,
            None => continue,
        };

        let coords = match &geom.value {
            Value::LineString(ref c) => c.clone(),
            _ => continue,
        };

        if coords.len() < 2 {
            continue;
        }

        // Calculate length
        let mut length_m = 0.0;
        for i in 0..coords.len() - 1 {
            if coords[i].len() >= 2 && coords[i + 1].len() >= 2 {
                length_m += haversine_m(
                    coords[i][1],
                    coords[i][0],
                    coords[i + 1][1],
                    coords[i + 1][0],
                );
            }
        }

        // Get or create nodes
        let start = &coords[0];
        let end = &coords[coords.len() - 1];

        if start.len() < 2 || end.len() < 2 {
            continue;
        }

        let start_id = node_id(start[0], start[1], decimals);
        let end_id = node_id(end[0], end[1], decimals);

        let start_idx = *node_map.entry(start_id.clone()).or_insert_with(|| {
            graph.add_node(Node {
                lon: start[0],
                lat: start[1],
            })
        });

        let end_idx = *node_map.entry(end_id.clone()).or_insert_with(|| {
            graph.add_node(Node {
                lon: end[0],
                lat: end[1],
            })
        });

        // Add edge
        let edge_coords: Vec<[f64; 2]> = coords
            .iter()
            .filter(|c| c.len() >= 2)
            .map(|c| [c[0], c[1]])
            .collect();

        let properties = feature.properties.clone().unwrap_or_default();

        graph.add_edge(
            start_idx,
            end_idx,
            Edge {
                length_m,
                coords: edge_coords,
                properties,
            },
        );
    }

    Ok(graph)
}

/// Remove self-loops from graph
pub fn remove_selfloops(graph: &mut RoadGraph) -> usize {
    let mut removed = 0;
    let edges: Vec<_> = graph.edge_indices().collect();

    for edge_idx in edges {
        if let Some((a, b)) = graph.edge_endpoints(edge_idx) {
            if a == b {
                graph.remove_edge(edge_idx);
                removed += 1;
            }
        }
    }

    removed
}

/// Remove edges shorter than min_length_m
pub fn remove_short_edges(graph: &mut RoadGraph, min_length_m: f64) -> usize {
    let mut removed = 0;
    let edges: Vec<_> = graph.edge_indices().collect();

    for edge_idx in edges {
        if let Some(edge) = graph.edge_weight(edge_idx) {
            if edge.length_m < min_length_m {
                graph.remove_edge(edge_idx);
                removed += 1;
            }
        }
    }

    removed
}

/// Deduplicate edges with same geometry
pub fn dedupe_edges(graph: &mut RoadGraph) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut seen: HashMap<(NodeIndex, NodeIndex), HashSet<u64>> = HashMap::new();
    let mut to_remove = Vec::new();

    for edge_idx in graph.edge_indices() {
        if let Some((a, b)) = graph.edge_endpoints(edge_idx) {
            if let Some(edge) = graph.edge_weight(edge_idx) {
                // Hash the coordinates
                let mut hasher = DefaultHasher::new();
                for coord in &edge.coords {
                    coord[0].to_bits().hash(&mut hasher);
                    coord[1].to_bits().hash(&mut hasher);
                }
                let hash = hasher.finish();

                let key = if a <= b { (a, b) } else { (b, a) };
                let hashes = seen.entry(key).or_insert_with(HashSet::new);

                if hashes.contains(&hash) {
                    to_remove.push(edge_idx);
                } else {
                    hashes.insert(hash);
                }
            }
        }
    }

    let removed = to_remove.len();
    for edge_idx in to_remove {
        graph.remove_edge(edge_idx);
    }

    removed
}

/// Remove edges missing required attributes
pub fn remove_edges_missing_attrs(graph: &mut RoadGraph, required_attrs: &[String]) -> usize {
    let mut to_remove = Vec::new();

    for edge_idx in graph.edge_indices() {
        if let Some(edge) = graph.edge_weight(edge_idx) {
            for attr in required_attrs {
                match edge.properties.get(attr) {
                    Some(JsonValue::Null) | None => {
                        to_remove.push(edge_idx);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    let removed = to_remove.len();
    for edge_idx in to_remove {
        graph.remove_edge(edge_idx);
    }

    removed
}

/// Merge a list of property values for parallel edge merging.
/// - Numeric values: average them
/// - String values: join with "; " (deduplicated)
/// - Other types: keep first value
fn merge_property_values(values: &[JsonValue]) -> JsonValue {
    if values.is_empty() {
        return JsonValue::Null;
    }
    if values.len() == 1 {
        return values[0].clone();
    }

    // Try numeric average
    let non_null: Vec<_> = values.iter().filter(|v| !v.is_null()).collect();
    if non_null.is_empty() {
        return JsonValue::Null;
    }
    if non_null.len() == 1 {
        return non_null[0].clone();
    }

    // Check if all are numbers
    let all_numbers = non_null.iter().all(|v| v.is_number());
    if all_numbers {
        let sum: f64 = non_null.iter().filter_map(|v| v.as_f64()).sum();
        let count = non_null.len() as f64;
        return JsonValue::from(
            serde_json::Number::from_f64(sum / count).unwrap_or(serde_json::Number::from(0)),
        );
    }

    // Check if all are strings
    let all_strings = non_null.iter().all(|v| v.is_string());
    if all_strings {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for v in &non_null {
            if let Some(s) = v.as_str() {
                if seen.insert(s.to_string()) {
                    result.push(s.to_string());
                }
            }
        }
        return JsonValue::String(result.join("; "));
    }

    // Default: keep first value
    non_null[0].clone()
}

/// Merge parallel edges (same node pair).
/// When `merge_properties` is true, combines properties using merge_property_values.
pub fn merge_parallel_edges(graph: &mut RoadGraph, merge_properties: bool) -> usize {
    let mut merged = 0;
    let mut processed: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();

    let node_pairs: Vec<_> = graph
        .edge_indices()
        .filter_map(|e| graph.edge_endpoints(e))
        .collect();

    for (a, b) in node_pairs {
        let key = if a <= b { (a, b) } else { (b, a) };
        if processed.contains(&key) {
            continue;
        }
        processed.insert(key);

        let edges: Vec<_> = if graph.is_directed() {
            graph
                .edges_connecting(a, b)
                .chain(graph.edges_connecting(b, a))
                .map(|e| e.id())
                .collect()
        } else {
            graph.edges_connecting(a, b).map(|e| e.id()).collect()
        };

        if edges.len() <= 1 {
            continue;
        }

        // Collect properties from all parallel edges
        let mut all_props: Vec<Map<String, JsonValue>> = Vec::new();
        let mut total_length = 0.0f64;

        for &edge_idx in &edges {
            if let Some(edge) = graph.edge_weight(edge_idx) {
                total_length += edge.length_m;
                all_props.push(edge.properties.clone());
            }
        }

        // Merge properties if requested
        let merged_props = if merge_properties && !all_props.is_empty() {
            let mut result = Map::new();
            let mut all_keys: HashSet<String> = HashSet::new();
            for props in &all_props {
                for key in props.keys() {
                    all_keys.insert(key.clone());
                }
            }
            for key in all_keys {
                let values: Vec<JsonValue> = all_props
                    .iter()
                    .filter_map(|p| p.get(&key).cloned())
                    .filter(|v| !v.is_null())
                    .collect();
                if !values.is_empty() {
                    result.insert(key, merge_property_values(&values));
                }
            }
            result
        } else {
            // Keep first edge's properties
            all_props.first().cloned().unwrap_or_default()
        };

        // Update first edge
        let first_edge_idx = edges[0];
        if let Some(edge) = graph.edge_weight_mut(first_edge_idx) {
            edge.length_m = total_length;
            edge.properties = merged_props;
        }

        // Remove other edges
        for &edge_idx in &edges[1..] {
            graph.remove_edge(edge_idx);
            merged += 1;
        }
    }

    merged
}

/// Remove isolated nodes (degree 0)
pub fn remove_isolates(graph: &mut RoadGraph) -> usize {
    let isolates: Vec<_> = graph
        .node_indices()
        .filter(|&n| graph.edges(n).count() == 0)
        .collect();

    let removed = isolates.len();
    for node in isolates {
        graph.remove_node(node);
    }

    removed
}

/// Keep only the largest N connected components
pub fn keep_largest_components(graph: &mut RoadGraph, max_components: usize) -> usize {
    use petgraph::algo::connected_components;
    use petgraph::visit::Dfs;

    let num_components = connected_components(&*graph);

    if num_components <= max_components {
        return 0;
    }

    // Find all components
    let mut visited = HashSet::new();
    let mut components: Vec<Vec<NodeIndex>> = Vec::new();

    for node in graph.node_indices() {
        if visited.contains(&node) {
            continue;
        }

        let mut component = Vec::new();
        let mut dfs = Dfs::new(&*graph, node);

        while let Some(n) = dfs.next(&*graph) {
            visited.insert(n);
            component.push(n);
        }

        components.push(component);
    }

    // Sort by size (largest first)
    components.sort_by_key(|b| std::cmp::Reverse(b.len()));

    // Keep largest N components
    let mut keep_nodes: HashSet<NodeIndex> = HashSet::new();
    for component in components.iter().take(max_components) {
        keep_nodes.extend(component);
    }

    // Remove nodes not in largest components
    let all_nodes: Vec<_> = graph.node_indices().collect();
    for node in all_nodes {
        if !keep_nodes.contains(&node) {
            graph.remove_node(node);
        }
    }

    num_components - max_components
}

/// Convert graph back to GeoJSON, optionally including point features
pub fn graph_to_geojson(
    graph: &RoadGraph,
    simplify_tolerance_m: f64,
    point_features: Option<&[Feature]>,
) -> Result<FeatureCollection> {
    let mut features = Vec::new();

    for edge_idx in graph.edge_indices() {
        if let Some(edge) = graph.edge_weight(edge_idx) {
            let mut coords = edge.coords.clone();

            // Simplify if requested
            if simplify_tolerance_m > 0.0 && coords.len() > 2 {
                coords = simplify_coords(&coords, simplify_tolerance_m);
            }

            if coords.len() < 2 {
                continue;
            }

            let geojson_coords: Vec<Vec<f64>> = coords.iter().map(|c| vec![c[0], c[1]]).collect();

            features.push(Feature {
                bbox: None,
                geometry: Some(Geometry {
                    bbox: None,
                    value: Value::LineString(geojson_coords),
                    foreign_members: None,
                }),
                id: None,
                properties: Some(edge.properties.clone()),
                foreign_members: None,
            });
        }
    }

    // Append point features if provided
    if let Some(points) = point_features {
        features.extend(points.iter().cloned());
    }

    Ok(FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    })
}

/// Simple Douglas-Peucker simplification
fn simplify_coords(coords: &[[f64; 2]], tolerance_m: f64) -> Vec<[f64; 2]> {
    if coords.len() <= 2 {
        return coords.to_vec();
    }

    use geo::algorithm::simplify::Simplify;
    use geo::LineString;

    let line: LineString<f64> = coords.iter().map(|c| (c[0], c[1])).collect();
    let tolerance_deg = tolerance_m / 111_320.0;
    let simplified = line.simplify(&tolerance_deg);

    simplified.0.iter().map(|c| [c.x, c.y]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn make_simple_graph() -> RoadGraph {
        let mut g = RoadGraph::new_undirected();
        let a = g.add_node(Node { lon: 0.0, lat: 0.0 });
        let b = g.add_node(Node { lon: 1.0, lat: 0.0 });
        let c = g.add_node(Node { lon: 2.0, lat: 0.0 });
        g.add_edge(
            a,
            b,
            Edge {
                length_m: 100.0,
                coords: vec![[0.0, 0.0], [1.0, 0.0]],
                properties: Map::new(),
            },
        );
        g.add_edge(
            b,
            c,
            Edge {
                length_m: 100.0,
                coords: vec![[1.0, 0.0], [2.0, 0.0]],
                properties: Map::new(),
            },
        );
        g
    }

    #[test]
    fn test_remove_selfloops() {
        let mut g = make_simple_graph();
        let a = g.add_node(Node { lon: 5.0, lat: 5.0 });
        g.add_edge(
            a,
            a,
            Edge {
                length_m: 0.0,
                coords: vec![[5.0, 5.0], [5.0, 5.0]],
                properties: Map::new(),
            },
        );
        assert_eq!(g.edge_count(), 3);
        let removed = remove_selfloops(&mut g);
        assert_eq!(removed, 1);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_remove_short_edges() {
        let mut g = RoadGraph::new_undirected();
        let a = g.add_node(Node { lon: 0.0, lat: 0.0 });
        let b = g.add_node(Node {
            lon: 0.001,
            lat: 0.0,
        });
        let c = g.add_node(Node { lon: 1.0, lat: 0.0 });
        g.add_edge(
            a,
            b,
            Edge {
                length_m: 0.05,
                coords: vec![[0.0, 0.0], [0.001, 0.0]],
                properties: Map::new(),
            },
        );
        g.add_edge(
            b,
            c,
            Edge {
                length_m: 100.0,
                coords: vec![[0.001, 0.0], [1.0, 0.0]],
                properties: Map::new(),
            },
        );
        let removed = remove_short_edges(&mut g, 1.0);
        assert_eq!(removed, 1);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_remove_isolates() {
        let mut g = make_simple_graph();
        let _isolated = g.add_node(Node {
            lon: 99.0,
            lat: 99.0,
        });
        assert_eq!(g.node_count(), 4);
        let removed = remove_isolates(&mut g);
        assert_eq!(removed, 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_keep_largest_components() {
        let mut g = RoadGraph::new_undirected();
        let a = g.add_node(Node { lon: 0.0, lat: 0.0 });
        let b = g.add_node(Node { lon: 1.0, lat: 0.0 });
        let c = g.add_node(Node {
            lon: 10.0,
            lat: 0.0,
        });
        let d = g.add_node(Node {
            lon: 11.0,
            lat: 0.0,
        });
        let e = g.add_node(Node {
            lon: 20.0,
            lat: 0.0,
        });
        g.add_edge(
            a,
            b,
            Edge {
                length_m: 100.0,
                coords: vec![[0.0, 0.0], [1.0, 0.0]],
                properties: Map::new(),
            },
        );
        g.add_edge(
            c,
            d,
            Edge {
                length_m: 100.0,
                coords: vec![[10.0, 0.0], [11.0, 0.0]],
                properties: Map::new(),
            },
        );
        g.add_edge(
            d,
            e,
            Edge {
                length_m: 100.0,
                coords: vec![[11.0, 0.0], [20.0, 0.0]],
                properties: Map::new(),
            },
        );
        let removed = keep_largest_components(&mut g, 1);
        assert_eq!(removed, 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_dedupe_edges() {
        let mut g = RoadGraph::new_undirected();
        let a = g.add_node(Node { lon: 0.0, lat: 0.0 });
        let b = g.add_node(Node { lon: 1.0, lat: 0.0 });
        g.add_edge(
            a,
            b,
            Edge {
                length_m: 100.0,
                coords: vec![[0.0, 0.0], [1.0, 0.0]],
                properties: Map::new(),
            },
        );
        g.add_edge(
            a,
            b,
            Edge {
                length_m: 100.0,
                coords: vec![[0.0, 0.0], [1.0, 0.0]],
                properties: Map::new(),
            },
        );
        assert_eq!(g.edge_count(), 2);
        let removed = dedupe_edges(&mut g);
        assert_eq!(removed, 1);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_merge_parallel_edges() {
        let mut g = RoadGraph::new_undirected();
        let a = g.add_node(Node { lon: 0.0, lat: 0.0 });
        let b = g.add_node(Node { lon: 1.0, lat: 0.0 });
        let mut props1 = Map::new();
        props1.insert(
            "name".to_string(),
            serde_json::Value::String("Main St".to_string()),
        );
        let mut props2 = Map::new();
        props2.insert(
            "name".to_string(),
            serde_json::Value::String("Main St Alt".to_string()),
        );
        g.add_edge(
            a,
            b,
            Edge {
                length_m: 100.0,
                coords: vec![[0.0, 0.0], [1.0, 0.0]],
                properties: props1,
            },
        );
        g.add_edge(
            a,
            b,
            Edge {
                length_m: 80.0,
                coords: vec![[0.0, 0.0], [1.0, 0.1]],
                properties: props2,
            },
        );
        assert_eq!(g.edge_count(), 2);
        let merged = merge_parallel_edges(&mut g, true);
        assert_eq!(merged, 1);
        assert_eq!(g.edge_count(), 1);
        // Length should be summed
        let edge = g.edge_weights().next().unwrap();
        assert_eq!(edge.length_m, 180.0);
        // Properties should be merged (strings joined with ";")
        assert!(edge.properties.contains_key("name"));
    }
}
