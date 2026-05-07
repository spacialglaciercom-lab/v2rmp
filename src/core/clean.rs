use anyhow::Result;
use serde::{Deserialize, Serialize};

mod geometry;
mod graph;
mod spatial;
mod stats;

pub use stats::CleanStats;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CleanOptions {
    pub make_valid: bool,
    pub drop_invalid: bool,
    pub remove_selfloops: bool,
    pub min_length_m: f64,
    pub node_snap_m: f64,
    pub node_precision_decimals: u32,
    pub merge_node_positions: bool,
    pub dedupe_edges: bool,
    pub remove_isolates: bool,
    pub max_components: usize,
    pub required_attrs: Option<Vec<String>>,
    pub merge_parallel_edges: bool,
    pub merge_parallel_edge_properties: bool,
    pub property_merge_strategy: String, // "first" or "merge"
    pub simplify_tolerance_m: f64,
    pub include_polygons: bool,
    pub include_points: bool,
}

impl Default for CleanOptions {
    fn default() -> Self {
        Self {
            make_valid: true,
            drop_invalid: true,
            remove_selfloops: true,
            min_length_m: 0.1,
            node_snap_m: 1.0,
            node_precision_decimals: 6,
            merge_node_positions: true,
            dedupe_edges: true,
            remove_isolates: true,
            max_components: 1,
            required_attrs: None,
            merge_parallel_edges: false,
            merge_parallel_edge_properties: false,
            property_merge_strategy: "first".to_string(),
            simplify_tolerance_m: 0.0,
            include_polygons: false,
            include_points: false,
        }
    }
}

/// Main cleaning pipeline: GeoJSON -> cleaned GeoJSON
pub fn clean_geojson(
    geojson: &geojson::FeatureCollection,
    options: &CleanOptions,
) -> Result<(geojson::FeatureCollection, CleanStats, Vec<String>)> {
    let mut stats = CleanStats::default();
    let mut warnings = Vec::new();
    stats.input_features = geojson.features.len();

    // Stage 1: Repair geometries
    let (repaired, point_features) =
        geometry::repair_features(&geojson.features, options, &mut stats)?;

    if repaired.is_empty() && point_features.is_empty() {
        return Ok((
            geojson::FeatureCollection {
                bbox: None,
                features: vec![],
                foreign_members: None,
            },
            stats,
            warnings,
        ));
    }

    // Check for high invalid drop ratio
    if stats.input_features > 0 {
        let drop_ratio = stats.invalid_dropped as f64 / stats.input_features as f64;
        if drop_ratio > 0.10 {
            let msg = format!(
                "Over 10% of features were dropped as invalid ({}/{} = {:.1}%). \
                 Consider checking CRS and geometry validity.",
                stats.invalid_dropped,
                stats.input_features,
                drop_ratio * 100.0
            );
            warnings.push(msg);
        }
    }

    // Stage 2: Build graph (only from LineString features, not points)
    let mut graph = graph::build_graph(&repaired, options.node_precision_decimals)?;

    // Stages 3–11
    if options.remove_selfloops {
        stats.selfloops_removed = graph::remove_selfloops(&mut graph);
    }

    if options.min_length_m > 0.0 {
        stats.short_edges_removed = graph::remove_short_edges(&mut graph, options.min_length_m);
    }

    if options.node_snap_m > 0.0 {
        stats.nodes_merged = spatial::merge_nearby_nodes(
            &mut graph,
            options.node_snap_m,
            options.node_precision_decimals,
            options.merge_node_positions,
        )?;
    }

    if options.dedupe_edges {
        stats.duplicate_edges_removed = graph::dedupe_edges(&mut graph);
    }

    if let Some(ref attrs) = options.required_attrs {
        stats.incomplete_edges_removed = graph::remove_edges_missing_attrs(&mut graph, attrs);
    }

    if options.merge_parallel_edges {
        let merge_props =
            options.property_merge_strategy == "merge" || options.merge_parallel_edge_properties;
        stats.parallel_edges_merged = graph::merge_parallel_edges(&mut graph, merge_props);
    }

    if options.remove_isolates {
        stats.isolates_removed = graph::remove_isolates(&mut graph);
    }

    if options.max_components > 0 {
        stats.components_removed =
            graph::keep_largest_components(&mut graph, options.max_components);
    }

    // Stage 4: Export to GeoJSON (include point features if requested)
    let point_ref = if options.include_points {
        Some(point_features.as_slice())
    } else {
        None
    };
    let cleaned = graph::graph_to_geojson(&graph, options.simplify_tolerance_m, point_ref)?;
    stats.output_features = cleaned.features.len();

    Ok((cleaned, stats, warnings))
}

/// Re-export shared haversine from core (lat, lon order).
pub(crate) use super::haversine_m;

/// Generate node ID from coordinates
pub fn node_id(lon: f64, lat: f64, decimals: u32) -> String {
    let factor = 10_f64.powi(decimals as i32);
    let lon_rounded = (lon * factor).round() / factor;
    let lat_rounded = (lat * factor).round() / factor;
    format!("{},{}", lon_rounded, lat_rounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_zero_distance() {
        let d = haversine_m(0.0, 0.0, 0.0, 0.0);
        assert!(d < 0.01, "Same point should have zero distance, got {}", d);
    }

    #[test]
    fn test_haversine_known_distance() {
        // NYC to London: approx 5,570 km
        let d = haversine_m(40.7128, -74.006, 51.5074, -0.1278);
        assert!(
            (d - 5_570_000.0).abs() < 100_000.0,
            "NYC to London should be ~5570 km, got {} m",
            d
        );
    }

    #[test]
    fn test_haversine_symmetry() {
        let d1 = haversine_m(20.0, 10.0, 40.0, 30.0);
        let d2 = haversine_m(40.0, 30.0, 20.0, 10.0);
        assert!(
            (d1 - d2).abs() < 0.01,
            "Haversine should be symmetric: {} vs {}",
            d1,
            d2
        );
    }

    #[test]
    fn test_node_id_basic() {
        let id = node_id(1.123456, 2.654321, 6);
        assert_eq!(id, "1.123456,2.654321");
    }

    #[test]
    fn test_node_id_rounding() {
        let id = node_id(1.1234567, 2.6543218, 6);
        assert_eq!(id, "1.123457,2.654322");
    }

    #[test]
    fn test_node_id_negative() {
        let id = node_id(-73.985428, 40.748817, 6);
        assert_eq!(id, "-73.985428,40.748817");
    }

    #[test]
    fn test_node_id_snap() {
        let id1 = node_id(10.1234561, 20.9876541, 6);
        let id2 = node_id(10.1234564, 20.9876544, 6);
        assert_eq!(id1, id2, "Should snap to same node at precision 6");
    }

    #[test]
    fn test_clean_options_default() {
        let opts = CleanOptions::default();
        assert!(opts.make_valid);
        assert!(opts.drop_invalid);
        assert!(opts.remove_selfloops);
        assert_eq!(opts.min_length_m, 0.1);
        assert_eq!(opts.node_snap_m, 1.0);
        assert!(opts.dedupe_edges);
        assert!(opts.remove_isolates);
        assert_eq!(opts.max_components, 1);
        assert!(!opts.merge_parallel_edges);
        assert_eq!(opts.property_merge_strategy, "first");
    }
}
