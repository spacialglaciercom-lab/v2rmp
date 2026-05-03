use anyhow::Result;
use ordered_float::OrderedFloat;
use petgraph::graph::NodeIndex;
use rstar::{RTree, AABB};
use std::collections::HashMap;

use super::super::clean::haversine_distance_m;
use super::graph::RoadGraph;

#[derive(Debug, Clone)]
struct NodePoint {
    idx: NodeIndex,
    lon: OrderedFloat<f64>,
    lat: OrderedFloat<f64>,
}

impl rstar::RTreeObject for NodePoint {
    type Envelope = AABB<[OrderedFloat<f64>; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.lon, self.lat])
    }
}

impl rstar::PointDistance for NodePoint {
    fn distance_2(&self, point: &[OrderedFloat<f64>; 2]) -> OrderedFloat<f64> {
        let dx = self.lon - point[0];
        let dy = self.lat - point[1];
        dx * dx + dy * dy
    }
}

/// Union-Find data structure for node clustering
struct UnionFind {
    parent: HashMap<NodeIndex, NodeIndex>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
        }
    }

    fn find(&mut self, x: NodeIndex) -> NodeIndex {
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(x) {
            e.insert(x);
            return x;
        }

        let parent = self.parent[&x];
        if parent != x {
            let root = self.find(parent);
            self.parent.insert(x, root);
            root
        } else {
            parent
        }
    }

    fn union(&mut self, a: NodeIndex, b: NodeIndex) {
        let ra = self.find(a);
        let rb = self.find(b);

        if ra != rb {
            // Canonical = smaller index
            let canonical = ra.min(rb);
            self.parent.insert(ra, canonical);
            self.parent.insert(rb, canonical);
        }
    }
}

/// Merge nodes within node_snap_m distance
pub fn merge_nearby_nodes(
    graph: &mut RoadGraph,
    node_snap_m: f64,
    decimals: u32,
    merge_positions: bool,
) -> Result<usize> {
    if graph.node_count() == 0 {
        return Ok(0);
    }

    // Build spatial index
    let node_points: Vec<NodePoint> = graph
        .node_indices()
        .filter_map(|idx| {
            graph.node_weight(idx).map(|node| NodePoint {
                idx,
                lon: OrderedFloat(node.lon),
                lat: OrderedFloat(node.lat),
            })
        })
        .collect();

    let tree = RTree::bulk_load(node_points);

    // Find nearby nodes and union them
    let mut uf = UnionFind::new();

    for node_idx in graph.node_indices() {
        if let Some(node) = graph.node_weight(node_idx) {
            let delta_lat = node_snap_m / 111_320.0;
            let cos_lat = (node.lat.to_radians().cos()).max(0.01);
            let delta_lon = node_snap_m / (111_320.0 * cos_lat);

            let envelope = AABB::from_corners(
                [
                    OrderedFloat(node.lon - delta_lon),
                    OrderedFloat(node.lat - delta_lat),
                ],
                [
                    OrderedFloat(node.lon + delta_lon),
                    OrderedFloat(node.lat + delta_lat),
                ],
            );

            for nearby in tree.locate_in_envelope(&envelope) {
                if nearby.idx != node_idx {
                    // Check actual distance
                    if let Some(other_node) = graph.node_weight(nearby.idx) {
                        let dist_m = haversine_distance_m(
                            node.lon,
                            node.lat,
                            other_node.lon,
                            other_node.lat,
                        );

                        if dist_m <= node_snap_m {
                            uf.union(node_idx, nearby.idx);
                        }
                    }
                }
            }
        }
    }

    // Build mapping: old_idx -> canonical_idx
    let nodes: Vec<_> = graph.node_indices().collect();
    let mut mapping: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for &node_idx in &nodes {
        let canonical = uf.find(node_idx);
        mapping.insert(node_idx, canonical);
    }

    // Count merged nodes
    let merged_count = mapping.iter().filter(|(k, v)| k != v).count();

    if merged_count == 0 {
        return Ok(0);
    }

    // Calculate new positions for canonical nodes
    if merge_positions {
        let mut groups: HashMap<NodeIndex, Vec<(f64, f64)>> = HashMap::new();

        for &node_idx in &nodes {
            if let Some(node) = graph.node_weight(node_idx) {
                let canonical = mapping[&node_idx];
                groups
                    .entry(canonical)
                    .or_insert_with(Vec::new)
                    .push((node.lon, node.lat));
            }
        }

        // Update canonical node positions (average)
        for (canonical_idx, positions) in groups {
            if positions.len() > 1 {
                let avg_lon = positions.iter().map(|p| p.0).sum::<f64>() / positions.len() as f64;
                let avg_lat = positions.iter().map(|p| p.1).sum::<f64>() / positions.len() as f64;

                if let Some(node) = graph.node_weight_mut(canonical_idx) {
                    let factor = 10_f64.powi(decimals as i32);
                    node.lon = (avg_lon * factor).round() / factor;
                    node.lat = (avg_lat * factor).round() / factor;
                }
            }
        }
    }

    // Relabel graph nodes
    let mut new_graph = RoadGraph::new_undirected();
    let mut new_node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    // Add canonical nodes
    for &old_idx in &nodes {
        let canonical = mapping[&old_idx];
        if let std::collections::hash_map::Entry::Vacant(e) = new_node_map.entry(canonical) {
            if let Some(node) = graph.node_weight(canonical) {
                let new_idx = new_graph.add_node(node.clone());
                e.insert(new_idx);
            }
        }
    }

    // Add edges with updated endpoints
    for edge_idx in graph.edge_indices() {
        if let Some((old_a, old_b)) = graph.edge_endpoints(edge_idx) {
            if let Some(edge) = graph.edge_weight(edge_idx) {
                let canonical_a = mapping[&old_a];
                let canonical_b = mapping[&old_b];

                if canonical_a == canonical_b {
                    // Skip self-loops created by merging
                    continue;
                }

                if let (Some(&new_a), Some(&new_b)) = (
                    new_node_map.get(&canonical_a),
                    new_node_map.get(&canonical_b),
                ) {
                    let mut new_edge = edge.clone();

                    // Update edge endpoints in coordinates
                    if let (Some(start_node), Some(end_node)) =
                        (new_graph.node_weight(new_a), new_graph.node_weight(new_b))
                    {
                        if !new_edge.coords.is_empty() {
                            new_edge.coords[0] = [start_node.lon, start_node.lat];
                            if new_edge.coords.len() > 1 {
                                let last_idx = new_edge.coords.len() - 1;
                                new_edge.coords[last_idx] = [end_node.lon, end_node.lat];
                            }

                            // Recalculate length
                            let mut length_m = 0.0;
                            for i in 0..new_edge.coords.len() - 1 {
                                length_m += haversine_distance_m(
                                    new_edge.coords[i][0],
                                    new_edge.coords[i][1],
                                    new_edge.coords[i + 1][0],
                                    new_edge.coords[i + 1][1],
                                );
                            }
                            new_edge.length_m = length_m;
                        }
                    }

                    new_graph.add_edge(new_a, new_b, new_edge);
                }
            }
        }
    }

    // Replace the original graph
    *graph = new_graph;

    Ok(merged_count)
}
