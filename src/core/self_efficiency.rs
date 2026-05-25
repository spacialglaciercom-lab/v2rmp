#![allow(dead_code)]
//! Self-Efficiency Tools for V2RMP
//!
//! This module implements three key MCP tools aimed at making the V2RMP system
//! fully autonomous and self-improving:
//!
//! 1. `geocode_to_bbox` - Resolves natural language locations into precise coordinates
//! 2. `autonomous_retraining` - Monitors feedback and triggers model retraining
//! 3. `network_integrity_auditor` - Validates route traversability using graph theory

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

use petgraph::algo::connected_components;
use petgraph::graph::NodeIndex;
use petgraph::visit::Dfs;

use crate::core::clean::graph::RoadGraph;
use crate::core::extract::BBoxRequest;

/// Result from geocoding a location name
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeocodeResult {
    /// Bounding box for the location
    pub bbox: BBoxRequest,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Display name of the location
    pub display_name: String,
    /// Center coordinates (lon, lat)
    pub center_lon: f64,
    pub center_lat: f64,
}

/// Result from network integrity audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkAuditResult {
    /// Whether the network passed all checks
    pub passed: bool,
    /// Number of strongly connected components
    pub num_strong_components: usize,
    /// Whether the graph is strongly connected (for directed graphs)
    pub is_strongly_connected: bool,
    /// Whether the graph is weakly connected (for undirected view)
    pub is_weakly_connected: bool,
    /// List of isolated nodes
    pub isolated_nodes: Vec<usize>,
    /// List of disconnected components (node indices in each)
    pub disconnected_components: Vec<Vec<usize>>,
    /// U-turn heatmap: intersections with high penalty detours
    pub uturn_heatmap: Vec<UturnIssue>,
    /// Physical constraint violations
    pub constraint_violations: Vec<ConstraintViolation>,
    /// Suggested repairs
    pub suggested_repairs: Vec<String>,
}

/// U-turn issue detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UturnIssue {
    /// Node index
    pub node_index: usize,
    /// Estimated detour distance in meters
    pub detour_m: f64,
    /// Severity (0.0 to 1.0)
    pub severity: f64,
}

/// Physical constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// Edge or node index
    pub index: usize,
    /// Type of constraint (e.g., "bridge_height", "weight_limit")
    pub constraint_type: String,
    /// Description of the violation
    pub description: String,
    /// Severity
    pub severity: f64,
}

/// Feedback entry for retraining
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    pub timestamp: String,
    pub instance_features: Vec<f64>,
    pub solver_id: String,
    pub total_distance_km: f64,
    pub elapsed_ms: u64,
    pub gap_to_bks: f64,
}

/// Retraining status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainingStatus {
    pub should_retrain: bool,
    pub reason: String,
    pub feedback_count: usize,
    pub avg_gap_to_bks: f64,
    pub gap_increase_pct: f64,
    pub new_models_generated: Vec<String>,
}

/// Geocode a location name to a bounding box
///
/// This function would typically call an external geocoding API like Nominatim.
/// For now, it includes a basic local lookup for common locations.
pub fn geocode_to_bbox(location_name: &str) -> Result<GeocodeResult> {
    // Trim and normalize the input
    let location = location_name.trim().to_lowercase();

    // Built-in known locations (can be extended)
    let known_locations: Vec<(Vec<&str>, BBoxRequest, &str, f64, f64)> = vec![
        (
            vec!["brooklyn", "ny", "new york", "brooklyn ny", "brooklyn, ny"],
            BBoxRequest {
                min_lon: -74.042091,
                min_lat: 40.570744,
                max_lon: -73.847567,
                max_lat: 40.739446,
            },
            "Brooklyn, NY, USA",
            -73.944158,
            40.678178,
        ),
        (
            vec![
                "plateau",
                "le plateau",
                "plateau montreal",
                "le plateau, montreal",
                "plateau, montreal",
            ],
            BBoxRequest {
                min_lon: -73.628,
                min_lat: 45.490,
                max_lon: -73.540,
                max_lat: 45.570,
            },
            "Le Plateau-Mont-Royal, Montreal, QC, Canada",
            -73.577,
            45.533,
        ),
        (
            vec!["montreal", "mtl", "montréal"],
            BBoxRequest {
                min_lon: -73.793,
                min_lat: 45.380,
                max_lon: -73.400,
                max_lat: 45.695,
            },
            "Montreal, QC, Canada",
            -73.567,
            45.502,
        ),
        (
            vec!["manhattan", "manhattan ny", "manhattan, ny"],
            BBoxRequest {
                min_lon: -74.047285,
                min_lat: 40.705324,
                max_lon: -73.906736,
                max_lat: 40.878095,
            },
            "Manhattan, NY, USA",
            -73.965,
            40.783,
        ),
        (
            vec!["paris", "paris, france"],
            BBoxRequest {
                min_lon: 2.224,
                min_lat: 48.815,
                max_lon: 2.469,
                max_lat: 48.902,
            },
            "Paris, France",
            2.352,
            48.857,
        ),
        (
            vec!["london", "london, uk", "london, england"],
            BBoxRequest {
                min_lon: -0.510,
                min_lat: 51.286,
                max_lon: 0.337,
                max_lat: 51.692,
            },
            "London, United Kingdom",
            -0.128,
            51.507,
        ),
        (
            vec!["berlin", "berlin, germany"],
            BBoxRequest {
                min_lon: 13.088,
                min_lat: 52.338,
                max_lon: 13.761,
                max_lat: 52.675,
            },
            "Berlin, Germany",
            13.405,
            52.520,
        ),
        (
            vec!["san francisco", "sf", "san francisco, ca"],
            BBoxRequest {
                min_lon: -122.524,
                min_lat: 37.708,
                max_lon: -122.375,
                max_lat: 37.831,
            },
            "San Francisco, CA, USA",
            -122.419,
            37.775,
        ),
    ];

    // Try to match against known locations
    for (aliases, bbox, display_name, center_lon, center_lat) in known_locations {
        for alias in aliases {
            if location.contains(alias) {
                return Ok(GeocodeResult {
                    bbox,
                    confidence: 0.95,
                    display_name: display_name.to_string(),
                    center_lon,
                    center_lat,
                });
            }
        }
    }

    // Try to parse as coordinates (lat,lon or lon,lat)
    if let Ok(result) = parse_coordinates(&location) {
        return Ok(result);
    }

    // Try to parse as bbox string
    if let Ok(result) = parse_bbox_string(&location) {
        return Ok(result);
    }

    // If no match, return a sensible default with low confidence
    Ok(GeocodeResult {
        bbox: BBoxRequest {
            min_lon: -73.6,
            min_lat: 45.4,
            max_lon: -73.5,
            max_lat: 45.6,
        },
        confidence: 0.1,
        display_name: format!("Unknown location: {}", location_name),
        center_lon: -73.55,
        center_lat: 45.5,
    })
}

/// Parse a string that looks like coordinates
fn parse_coordinates(location: &str) -> Result<GeocodeResult> {
    // Try to parse as "lat,lon" or "lon,lat"
    let parts: Vec<&str> = location.split(',').collect();
    if parts.len() == 2 {
        if let (Ok(val1), Ok(val2)) = (
            parts[0].trim().parse::<f64>(),
            parts[1].trim().parse::<f64>(),
        ) {
            // Try both orders: lat,lon and lon,lat
            let mut lat = val1;
            let mut lon = val2;

            // Check if val1 looks like lat and val2 looks like lon
            if !((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon)) {
                // Try swapping them
                lat = val2;
                lon = val1;
            }

            // Check if the swapped values are valid
            if (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) {
                let bbox = BBoxRequest {
                    min_lon: lon - 0.01,
                    min_lat: lat - 0.01,
                    max_lon: lon + 0.01,
                    max_lat: lat + 0.01,
                };
                return Ok(GeocodeResult {
                    bbox,
                    confidence: 0.99,
                    display_name: format!("Point at {}, {}", lat, lon),
                    center_lon: lon,
                    center_lat: lat,
                });
            }
        }
    }
    anyhow::bail!("Not a valid coordinate format");
}

/// Parse a bbox string "min_lon,min_lat,max_lon,max_lat"
fn parse_bbox_string(location: &str) -> Result<GeocodeResult> {
    let parts: Vec<&str> = location.split(',').collect();
    if parts.len() == 4 {
        if let (Ok(min_lon), Ok(min_lat), Ok(max_lon), Ok(max_lat)) = (
            parts[0].trim().parse::<f64>(),
            parts[1].trim().parse::<f64>(),
            parts[2].trim().parse::<f64>(),
            parts[3].trim().parse::<f64>(),
        ) {
            let center_lon = (min_lon + max_lon) / 2.0;
            let center_lat = (min_lat + max_lat) / 2.0;
            let bbox = BBoxRequest {
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            };
            return Ok(GeocodeResult {
                bbox,
                confidence: 0.99,
                display_name: format!(
                    "Bounding box: {},{} to {},{}",
                    min_lon, min_lat, max_lon, max_lat
                ),
                center_lon,
                center_lat,
            });
        }
    }
    anyhow::bail!("Not a valid bbox format");
}

/// Check network integrity and physical realism
pub fn audit_network_integrity(graph: &RoadGraph) -> Result<NetworkAuditResult> {
    let mut result = NetworkAuditResult {
        passed: true,
        num_strong_components: 0,
        is_strongly_connected: true,
        is_weakly_connected: true,
        isolated_nodes: Vec::new(),
        disconnected_components: Vec::new(),
        uturn_heatmap: Vec::new(),
        constraint_violations: Vec::new(),
        suggested_repairs: Vec::new(),
    };

    // Check 1: Find isolated nodes (degree 0)
    for node_idx in graph.node_indices() {
        if graph.edges(node_idx).count() == 0 {
            result.isolated_nodes.push(node_idx.index());
        }
    }

    if !result.isolated_nodes.is_empty() {
        result.passed = false;
        result.suggested_repairs.push(format!(
            "Remove {} isolated nodes",
            result.isolated_nodes.len()
        ));
    }

    // Check 2: Connected components analysis
    let num_components = connected_components(graph);
    result.num_strong_components = num_components;
    result.is_strongly_connected = num_components <= 1;

    if num_components > 1 {
        result.passed = false;
        result.is_weakly_connected = false;

        // Find all components
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        for node in graph.node_indices() {
            if visited.contains(&node) {
                continue;
            }
            let mut component = Vec::new();
            let mut dfs = Dfs::new(graph, node);
            while let Some(n) = dfs.next(graph) {
                visited.insert(n);
                component.push(n.index());
            }
            result.disconnected_components.push(component);
        }

        result.suggested_repairs.push(format!(
            "Prune {} disconnected components, keeping largest",
            num_components - 1
        ));
    }

    // Check 3: U-turn heatmap (simplified - detect sharp angles)
    // For each node, check if there are edges that form sharp angles
    // Note: This is a simplified check - we count edges per node as a proxy for U-turn issues
    for node_idx in graph.node_indices() {
        let edge_count = graph.edges(node_idx).count();
        if edge_count >= 3 {
            // Nodes with many edges may have U-turn issues
            // This is a heuristic - a more detailed check would analyze actual angles
            let angle_issues = if edge_count > 4 { 1 } else { 0 };
            if angle_issues > 0 {
                result.uturn_heatmap.push(UturnIssue {
                    node_index: node_idx.index(),
                    detour_m: edge_count as f64 * 5.0, // Simplified estimation
                    severity: (edge_count as f64 / 10.0).min(1.0),
                });
            }
        }
    }

    // Check 4: Physical constraints from properties
    for edge_idx in graph.edge_indices() {
        if let Some(edge) = graph.edge_weight(edge_idx) {
            // Check for bridge height limits
            if let Some(Value::Number(ref height)) = edge.properties.get("max_height") {
                if let Some(height_m) = height.as_f64() {
                    if height_m < 3.0 {
                        result.constraint_violations.push(ConstraintViolation {
                            index: edge_idx.index(),
                            constraint_type: "bridge_height".to_string(),
                            description: format!("Low bridge: {}m", height_m),
                            severity: 0.8,
                        });
                    }
                }
            }
            // Check for weight limits
            if let Some(Value::Number(ref weight)) = edge.properties.get("max_weight") {
                if let Some(weight_kg) = weight.as_f64() {
                    if weight_kg < 5000.0 {
                        result.constraint_violations.push(ConstraintViolation {
                            index: edge_idx.index(),
                            constraint_type: "weight_limit".to_string(),
                            description: format!("Low weight limit: {}kg", weight_kg),
                            severity: 0.6,
                        });
                    }
                }
            }
        }
    }

    if !result.constraint_violations.is_empty() {
        result.passed = false;
        result.suggested_repairs.push(format!(
            "Review {} physical constraint violations",
            result.constraint_violations.len()
        ));
    }

    Ok(result)
}

/// Calculate the angle between two edges at a common node
#[allow(dead_code)]
fn calculate_angle(coords1: &[[f64; 2]], coords2: &[[f64; 2]]) -> f64 {
    use std::f64::consts::PI;

    if coords1.len() < 2 || coords2.len() < 2 {
        return 0.0;
    }

    // Get direction vectors from the node
    // For simplicity, use the first and last points
    let dx1 = coords1.last().unwrap()[0] - coords1[0][0];
    let dy1 = coords1.last().unwrap()[1] - coords1[0][1];
    let dx2 = coords2.last().unwrap()[0] - coords2[0][0];
    let dy2 = coords2.last().unwrap()[1] - coords2[0][1];

    // Normalize vectors
    let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();

    if len1 < 1e-10 || len2 < 1e-10 {
        return 0.0;
    }

    let nx1 = dx1 / len1;
    let ny1 = dy1 / len1;
    let nx2 = dx2 / len2;
    let ny2 = dy2 / len2;

    // Dot product
    let dot = nx1 * nx2 + ny1 * ny2;

    // Clamp to [-1, 1] to avoid numerical errors
    let dot = dot.clamp(-1.0, 1.0);

    // Angle in radians
    let angle_rad = dot.acos();

    // Convert to degrees
    angle_rad * (180.0 / PI)
}

/// Load feedback entries from a JSONL file
pub fn load_feedback_data(path: &str) -> Result<Vec<FeedbackEntry>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let mut entries = Vec::new();
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: FeedbackEntry = serde_json::from_str(&line)?;
        entries.push(entry);
    }

    Ok(entries)
}

/// Check if autonomous retraining should be triggered
pub fn check_autonomous_retraining(
    feedback_path: &str,
    count_threshold: usize,
    gap_threshold_pct: f64,
) -> Result<RetrainingStatus> {
    let entries = load_feedback_data(feedback_path)?;
    let feedback_count = entries.len();

    // Calculate statistics
    let mut total_gap = 0.0;
    let mut total_count = 0;

    for entry in &entries {
        total_gap += entry.gap_to_bks;
        total_count += 1;
    }

    let avg_gap_to_bks = if total_count > 0 {
        total_gap / total_count as f64
    } else {
        0.0
    };

    // Check trigger A: Count-based
    let should_retrain_count = feedback_count >= count_threshold;

    // Check trigger B: Performance-based (if we have enough history)
    // For now, we use a simple heuristic: if average gap is above threshold
    let should_retrain_performance = avg_gap_to_bks > gap_threshold_pct;

    // Also check if gap has increased by 5% compared to historical
    // (This would require tracking historical averages - simplified for now)
    let gap_increase_pct = if avg_gap_to_bks > 0.0 {
        // Simplified: just use the current average as a proxy
        avg_gap_to_bks
    } else {
        0.0
    };

    let should_retrain = should_retrain_count || should_retrain_performance;

    let reason = if should_retrain_count {
        format!(
            "Feedback count ({}) >= threshold ({})",
            feedback_count, count_threshold
        )
    } else if should_retrain_performance {
        format!(
            "Average gap to BKS ({:.2}%) > threshold ({:.2}%)",
            avg_gap_to_bks, gap_threshold_pct
        )
    } else {
        "No retraining triggers met".to_string()
    };

    Ok(RetrainingStatus {
        should_retrain,
        reason,
        feedback_count,
        avg_gap_to_bks,
        gap_increase_pct,
        new_models_generated: Vec::new(),
    })
}

/// Execute the training pipeline and generate new models
#[cfg(feature = "ml")]
pub fn execute_retraining_pipeline(
    _feedback_path: &str,
    training_script: &str,
    output_dir: &str,
) -> Result<Vec<String>> {
    use std::process::Command;

    // Check that training script exists
    if !Path::new(training_script).exists() {
        anyhow::bail!("Training script not found at: {}", training_script);
    }

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    // Run the training pipeline
    let status = Command::new(training_script)
        .status()
        .context("Failed to execute training pipeline")?;

    if !status.success() {
        anyhow::bail!(
            "Training pipeline failed with exit code: {:?}",
            status.code()
        );
    }

    // Find generated .safetensors files
    let mut new_models = Vec::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                if let Some(name) = path.file_name() {
                    new_models.push(name.to_string_lossy().into_owned());
                }
            }
        }
    }

    Ok(new_models)
}

/// Validate a route for physical realism before optimization
pub fn validate_route_physical_realm(
    graph: &RoadGraph,
    task_edges: &[usize],
    task_stops: &[(f64, f64)],
) -> Result<NetworkAuditResult> {
    // First, run the general network audit
    let mut audit_result = audit_network_integrity(graph)?;

    // Check if all task edges exist in the graph
    let mut missing_edges = 0;
    for &edge_idx in task_edges {
        if graph.edge_indices().count() <= edge_idx {
            missing_edges += 1;
        }
    }

    if missing_edges > 0 {
        audit_result.passed = false;
        audit_result
            .suggested_repairs
            .push(format!("{} task edges not found in graph", missing_edges));
    }

    // Check if all stops can be snapped to the network
    use crate::core::haversine_m;

    let mut unsnapped_stops = 0;
    for &(lat, lon) in task_stops {
        let mut min_dist = f64::INFINITY;
        for node_idx in graph.node_indices() {
            if let Some(node) = graph.node_weight(node_idx) {
                let dist = haversine_m(lat, lon, node.lat, node.lon);
                if dist < min_dist {
                    min_dist = dist;
                }
                if dist < 100.0 {
                    // Found a close enough node
                    break;
                }
            }
        }
        if min_dist > 100.0 {
            unsnapped_stops += 1;
        }
    }

    if unsnapped_stops > 0 {
        audit_result.passed = false;
        audit_result.suggested_repairs.push(format!(
            "{} stops cannot be snapped to network (too far)",
            unsnapped_stops
        ));
    }

    Ok(audit_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    // Re-export Node and Edge for tests
    use crate::core::clean::graph::{Edge, Node};

    #[test]
    fn test_geocode_known_locations() {
        let result = geocode_to_bbox("Brooklyn, NY").unwrap();
        assert_eq!(result.display_name, "Brooklyn, NY, USA");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_geocode_le_plateau() {
        let result = geocode_to_bbox("Le Plateau, Montreal").unwrap();
        assert_eq!(
            result.display_name,
            "Le Plateau-Mont-Royal, Montreal, QC, Canada"
        );
        assert!(result.confidence > 0.9);
        // Check bbox is in Montreal area
        assert!(result.bbox.min_lon < -73.5);
        assert!(result.bbox.max_lon > -73.6);
    }

    #[test]
    fn test_geocode_coordinates() {
        // Test with a coordinate that won't match any known location
        // Use a location in the middle of the ocean
        let result = geocode_to_bbox("-100.0,30.0").unwrap();
        assert!(result.confidence > 0.9);
        assert!((result.center_lat - 30.0).abs() < 0.01);
        assert!((result.center_lon - (-100.0)).abs() < 0.01);
    }

    #[test]
    fn test_geocode_bbox_string() {
        // Test with min_lon,min_lat,max_lon,max_lat order
        let result = geocode_to_bbox("-73.6,45.4,-73.5,45.6").unwrap();
        assert!(result.confidence > 0.9); // Lower threshold for now
        assert!((result.bbox.min_lon - (-73.6)).abs() < 0.02);
        assert!((result.bbox.min_lat - 45.4).abs() < 0.02);
    }

    #[test]
    fn test_audit_empty_graph() {
        let graph = RoadGraph::new_undirected();
        let result = audit_network_integrity(&graph).unwrap();
        assert!(result.passed);
        assert_eq!(result.num_strong_components, 0);
    }

    #[test]
    fn test_audit_single_node() {
        let mut graph = RoadGraph::new_undirected();
        graph.add_node(Node { lon: 0.0, lat: 0.0 });
        let result = audit_network_integrity(&graph).unwrap();
        assert!(!result.passed);
        assert_eq!(result.isolated_nodes.len(), 1);
    }

    #[test]
    fn test_audit_connected_graph() {
        let mut graph = RoadGraph::new_undirected();
        let a = graph.add_node(Node { lon: 0.0, lat: 0.0 });
        let b = graph.add_node(Node { lon: 1.0, lat: 0.0 });
        graph.add_edge(
            a,
            b,
            Edge {
                length_m: 100.0,
                coords: vec![[0.0, 0.0], [1.0, 0.0]],
                properties: serde_json::Map::new(),
            },
        );
        let result = audit_network_integrity(&graph).unwrap();
        assert!(result.passed);
        assert_eq!(result.num_strong_components, 1);
        assert!(result.isolated_nodes.is_empty());
    }

    #[test]
    fn test_retraining_check_empty() {
        // Create a temporary empty feedback file
        let temp_path = "/tmp/test_feedback_empty.jsonl";
        std::fs::write(temp_path, "").unwrap();

        let status = check_autonomous_retraining(temp_path, 500, 5.0).unwrap();
        assert!(!status.should_retrain);
        assert_eq!(status.feedback_count, 0);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_retraining_check_with_data() {
        // Create a temporary feedback file with some entries
        let temp_path = "/tmp/test_feedback_data.jsonl";
        let entries = vec![
            FeedbackEntry {
                timestamp: "2024-01-01T00:00:00Z".to_string(),
                instance_features: vec![0.1; 30],
                solver_id: "clarke_wright".to_string(),
                total_distance_km: 42.5,
                elapsed_ms: 1200,
                gap_to_bks: 3.5,
            },
            FeedbackEntry {
                timestamp: "2024-01-02T00:00:00Z".to_string(),
                instance_features: vec![0.2; 30],
                solver_id: "sweep".to_string(),
                total_distance_km: 45.0,
                elapsed_ms: 1500,
                gap_to_bks: 8.0, // High gap
            },
        ];

        // Write entries to file
        let mut file = File::create(temp_path).unwrap();
        for entry in &entries {
            writeln!(file, "{}", serde_json::to_string(entry).unwrap()).unwrap();
        }

        let status = check_autonomous_retraining(temp_path, 2, 5.0).unwrap();
        assert!(status.should_retrain);
        assert_eq!(status.feedback_count, 2);
        assert!(status.avg_gap_to_bks > 5.0);

        std::fs::remove_file(temp_path).ok();
    }
}
