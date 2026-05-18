use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Geometric/topological summary of a CPP route.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouteMetrics {
    /// Number of transitions in route_nodes (i.e. len(route)-1).
    pub total_hops: usize,
    /// Distinct edges of the original graph traversed at least once.
    pub unique_required_edges: usize,
    /// Total number of edges in the original graph.
    pub edges_in_graph: usize,
    /// unique_required_edges / edges_in_graph. Should be 1.0 for a valid solution.
    pub coverage: f64,
    /// total_hops / edges_in_graph. 1.0 = perfect Eulerian.
    pub edge_repeat_ratio: f64,
    /// Count of A -> B -> A patterns.
    pub immediate_reversals: usize,
    /// Count of closed loops formed entirely by deadhead transitions.
    pub null_deadhead_cycles: usize,
    /// Fraction of total route distance spent on deadhead transitions.
    pub deadhead_distance_ratio: f64,
}

/// Canonicalize an edge for a hash set/map.
pub fn canon(u: usize, v: usize, directed: bool) -> (usize, usize) {
    if directed {
        (u, v)
    } else {
        if u < v {
            (u, v)
        } else {
            (v, u)
        }
    }
}

/// Count of A -> B -> A patterns where A != B.
pub fn immediate_reversal_count(route_nodes: &[usize]) -> usize {
    if route_nodes.len() < 3 {
        return 0;
    }
    let mut count = 0;
    for i in 0..route_nodes.len() - 2 {
        if route_nodes[i] == route_nodes[i + 2] && route_nodes[i] != route_nodes[i + 1] {
            count += 1;
        }
    }
    count
}

/// Count closed loops made entirely of deadhead/bridge transitions.
pub fn null_deadhead_cycle_count(
    route_nodes: &[usize],
    required_edges: &HashSet<(usize, usize)>,
    directed: bool,
) -> usize {
    if route_nodes.len() < 3 || (required_edges.is_empty()) {
        return 0;
    }

    let mut cycles = 0;
    let mut run_nodes = Vec::new();
    let mut in_run = false;

    for i in 0..route_nodes.len() - 1 {
        let u = route_nodes[i];
        let v = route_nodes[i + 1];
        let is_deadhead = !required_edges.contains(&canon(u, v, directed));

        if is_deadhead {
            if !in_run {
                run_nodes.push(u);
                in_run = true;
            }
            run_nodes.push(v);
        } else {
            if in_run {
                let mut unique_nodes = HashSet::new();
                for &node in &run_nodes {
                    if !unique_nodes.insert(node) {
                        cycles += 1;
                        break;
                    }
                }
                run_nodes.clear();
                in_run = false;
            }
        }
    }

    if in_run {
        let mut unique_nodes = HashSet::new();
        for &node in &run_nodes {
            if !unique_nodes.insert(node) {
                cycles += 1;
                break;
            }
        }
    }

    cycles
}

/// Haversine distance in km between two WGS-84 points.
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0088;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    r * c
}

/// Fraction of total route distance spent on deadhead/bridge transitions.
pub fn deadhead_distance_ratio(
    route_nodes: &[usize],
    required_edges: &HashSet<(usize, usize)>,
    edge_lengths: &HashMap<(usize, usize), f64>,
    node_coords: &HashMap<usize, (f64, f64)>,
    directed: bool,
) -> f64 {
    if route_nodes.len() < 2 {
        return 0.0;
    }

    let mut total_dist = 0.0;
    let mut deadhead_dist = 0.0;
    let mut seen_required: HashMap<(usize, usize), usize> = HashMap::new();

    for i in 0..route_nodes.len() - 1 {
        let u = route_nodes[i];
        let v = route_nodes[i + 1];
        let c = canon(u, v, directed);

        let length = if let Some(&l) = edge_lengths.get(&c) {
            l
        } else {
            if let (Some(&(lat1, lon1)), Some(&(lat2, lon2))) =
                (node_coords.get(&u), node_coords.get(&v))
            {
                haversine_km(lat1, lon1, lat2, lon2)
            } else {
                0.0
            }
        };

        total_dist += length;
        if !required_edges.contains(&c) {
            deadhead_dist += length;
        } else {
            let count = seen_required.entry(c).or_insert(0);
            *count += 1;
            if *count > 1 {
                deadhead_dist += length;
            }
        }
    }

    if total_dist <= 0.0 {
        0.0
    } else {
        deadhead_dist / total_dist
    }
}

/// Compute all metrics for a single route.
pub fn compute_route_metrics(
    route_nodes: &[usize],
    required_edges: &HashSet<(usize, usize)>,
    edges_in_graph: usize,
    edge_lengths: &HashMap<(usize, usize), f64>,
    node_coords: &HashMap<usize, (f64, f64)>,
    directed: bool,
) -> RouteMetrics {
    let mut seen = HashSet::new();
    for i in 0..route_nodes.len().saturating_sub(1) {
        let pair = canon(route_nodes[i], route_nodes[i + 1], directed);
        if required_edges.contains(&pair) {
            seen.insert(pair);
        }
    }

    let total_hops = route_nodes.len().saturating_sub(1);
    let coverage = if edges_in_graph > 0 {
        seen.len() as f64 / edges_in_graph as f64
    } else {
        1.0
    };
    let repeat = if edges_in_graph > 0 {
        total_hops as f64 / edges_in_graph as f64
    } else {
        0.0
    };

    RouteMetrics {
        total_hops,
        unique_required_edges: seen.len(),
        edges_in_graph,
        coverage,
        edge_repeat_ratio: repeat,
        immediate_reversals: immediate_reversal_count(route_nodes),
        null_deadhead_cycles: null_deadhead_cycle_count(route_nodes, required_edges, directed),
        deadhead_distance_ratio: deadhead_distance_ratio(
            route_nodes,
            required_edges,
            edge_lengths,
            node_coords,
            directed,
        ),
    }
}
