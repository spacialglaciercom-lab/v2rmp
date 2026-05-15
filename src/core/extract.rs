use crate::core::overture::{BBox, Geometry, OvertureExtractor, OvertureSegment};
use anyhow::{Context, Result};
use geojson::{Feature, FeatureCollection, Geometry as GeoJsonGeometry, Value as GeoJsonValue};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExtractSource {
    Osm,
    Overture,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBoxRequest {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

impl From<BBoxRequest> for BBox {
    fn from(r: BBoxRequest) -> Self {
        BBox {
            min_lon: r.min_lon,
            min_lat: r.min_lat,
            max_lon: r.max_lon,
            max_lat: r.max_lat,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractRequest {
    pub source: ExtractSource,
    pub bbox: BBoxRequest,
    pub road_classes: Vec<RoadClass>,
    pub output_path: String,
    pub pbf_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RoadClass {
    Residential,
    Tertiary,
    Secondary,
    Primary,
    Trunk,
    Motorway,
    Unclassified,
    LivingStreet,
    Service,
    SecondaryLink,
    PrimaryLink,
    TrunkLink,
    MotorwayLink,
}

impl RoadClass {
    pub fn all_vehicle() -> Vec<Self> {
        vec![
            Self::Residential,
            Self::Tertiary,
            Self::Secondary,
            Self::Primary,
            Self::Trunk,
            Self::Motorway,
            Self::Unclassified,
            Self::LivingStreet,
            Self::Service,
            Self::SecondaryLink,
            Self::PrimaryLink,
            Self::TrunkLink,
            Self::MotorwayLink,
        ]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Residential => "residential",
            Self::Tertiary => "tertiary",
            Self::Secondary => "secondary",
            Self::Primary => "primary",
            Self::Trunk => "trunk",
            Self::Motorway => "motorway",
            Self::Unclassified => "unclassified",
            Self::LivingStreet => "living_street",
            Self::Service => "service",
            Self::SecondaryLink => "secondary_link",
            Self::PrimaryLink => "primary_link",
            Self::TrunkLink => "trunk_link",
            Self::MotorwayLink => "motorway_link",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractResult {
    pub nodes: usize,
    pub edges: usize,
    pub total_km: f64,
    pub output_path: String,
}

/// Extract road network from Overture Maps S3 Parquet files or OSM PBF.
pub async fn run_extract(req: &ExtractRequest) -> anyhow::Result<ExtractResult> {
    match req.source {
        ExtractSource::Overture => run_overture_extract(req).await,
        ExtractSource::Osm => run_osm_extract(req).await,
    }
}

/// Run OSM PBF extraction
async fn run_osm_extract(req: &ExtractRequest) -> anyhow::Result<ExtractResult> {
    use crate::core::osm::{segment_to_feature, BBox as OsmBBox, OsmExtractor, OverpassExtractor};

    let bbox = OsmBBox {
        min_lon: req.bbox.min_lon,
        min_lat: req.bbox.min_lat,
        max_lon: req.bbox.max_lon,
        max_lat: req.bbox.max_lat,
    };
    let classes: Vec<String> = req
        .road_classes
        .iter()
        .map(|rc| rc.as_str().to_string())
        .collect();

    // Priority: 1. CLI/Request path, 2. Env var, 3. Default file, 4. Overpass API
    let pbf_path_candidate = req.pbf_path.clone()
        .or_else(|| std::env::var("OSM_PBF_PATH").ok());

    let segments = if let Some(pbf_path) = pbf_path_candidate {
        if std::path::Path::new(&pbf_path).exists() {
            tracing::info!("Using local PBF file: {}", pbf_path);
            let extractor = OsmExtractor::new(pbf_path)?;
            extractor.extract_bbox(&bbox, &classes)?
        } else {
            tracing::info!("PBF file not found at {}, falling back to Overpass", pbf_path);
            let extractor = OverpassExtractor::new();
            extractor.extract_bbox(&bbox, &classes).await?
        }
    } else if std::path::Path::new("monaco.osm.pbf").exists() {
        tracing::info!("Using default PBF file: monaco.osm.pbf");
        let extractor = OsmExtractor::new("monaco.osm.pbf".to_string())?;
        extractor.extract_bbox(&bbox, &classes)?
    } else {
        tracing::info!("No PBF file found, using Overpass API");
        let extractor = OverpassExtractor::new();
        extractor.extract_bbox(&bbox, &classes).await?
    };

    let features: Vec<Feature> = segments.into_iter().map(segment_to_feature).collect();

    // Build graph statistics
    let (nodes, edges, total_km) = build_graph_stats(&features)?;

    // Write GeoJSON output
    let geojson = FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    let geojson_string = serde_json::to_string_pretty(&geojson)?;
    let output_path = &req.output_path;

    std::fs::File::create(output_path)?
        .write_all(geojson_string.as_bytes())
        .context("Failed to write GeoJSON output")?;

    Ok(ExtractResult {
        nodes,
        edges,
        total_km,
        output_path: output_path.clone(),
    })
}

/// Run Overture extraction from S3
async fn run_overture_extract(req: &ExtractRequest) -> anyhow::Result<ExtractResult> {
    let bbox: BBox = req.bbox.clone().into();
    let extractor = OvertureExtractor::new()?;

    tracing::info!(
        "Extracting Overture data for bbox: [{:.4}, {:.4}, {:.4}, {:.4}]",
        bbox.min_lon,
        bbox.min_lat,
        bbox.max_lon,
        bbox.max_lat
    );

    let segments = extractor.extract_bbox(&bbox).await?;

    tracing::info!("Extracted {} segments from Overture S3", segments.len());

    // Convert segments to GeoJSON features
    let features: Vec<Feature> = segments
        .into_iter()
        .filter(|seg| should_include_segment(seg, &req.road_classes))
        .map(segment_to_feature)
        .collect();

    tracing::info!(
        "After road class filtering: {} road segments",
        features.len()
    );

    // Build graph statistics
    let (nodes, edges, total_km) = build_graph_stats(&features)?;

    // Write GeoJSON output
    let geojson = FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    let geojson_string = serde_json::to_string_pretty(&geojson)?;
    let output_path = &req.output_path;

    File::create(output_path)?
        .write_all(geojson_string.as_bytes())
        .context("Failed to write GeoJSON output")?;

    Ok(ExtractResult {
        nodes,
        edges,
        total_km,
        output_path: output_path.clone(),
    })
}

/// Check if a segment should be included based on road classes
fn should_include_segment(seg: &OvertureSegment, classes: &[RoadClass]) -> bool {
    if classes.is_empty() {
        return true;
    }

    let class_str = seg.class.as_deref().unwrap_or("");

    classes.iter().any(|rc| rc.as_str() == class_str)
}

/// Convert an Overture segment to a GeoJSON feature
fn segment_to_feature(seg: OvertureSegment) -> Feature {
    let geometry_json = match seg.geometry {
        Geometry::LineString(coords) => {
            json!({
                "type": "LineString",
                "coordinates": coords
            })
        }
        Geometry::Point(lon, lat) => {
            json!({
                "type": "Point",
                "coordinates": [lon, lat]
            })
        }
    };

    // Convert JSON to geojson::Geometry
    let geometry = GeoJsonGeometry::from_json_value(geometry_json)
        .expect("Failed to convert JSON to geojson::Geometry");

    // Build properties map
    let mut props = serde_json::Map::new();
    props.insert("id".to_string(), serde_json::Value::String(seg.id.clone()));
    if let Some(ref name) = seg.name {
        props.insert("name".to_string(), serde_json::Value::String(name.clone()));
    }
    if let Some(ref class) = seg.class {
        props.insert(
            "class".to_string(),
            serde_json::Value::String(class.clone()),
        );
    }
    if let Some(ref subtype) = seg.subtype {
        props.insert(
            "subtype".to_string(),
            serde_json::Value::String(subtype.clone()),
        );
    }
    if let Some(ref surface) = seg.surface {
        props.insert(
            "surface".to_string(),
            serde_json::Value::String(surface.clone()),
        );
    }
    if let Some(ref oneway) = seg.oneway {
        props.insert(
            "oneway".to_string(),
            serde_json::Value::String(oneway.clone()),
        );
    }
    if let Some(ref junction) = seg.junction {
        props.insert(
            "junction".to_string(),
            serde_json::Value::String(junction.clone()),
        );
    }
    if let Some(ref osm_id) = seg.osm_id {
        props.insert(
            "osm_id".to_string(),
            serde_json::Value::String(osm_id.clone()),
        );
    }

    Feature {
        id: None,
        bbox: None,
        geometry: Some(geometry),
        properties: Some(props),
        foreign_members: None,
    }
}

/// Build graph statistics from features
fn build_graph_stats(features: &[Feature]) -> Result<(usize, usize, f64)> {
    // Node deduplication: snap coordinates to ~1m precision
    let mut node_map: HashMap<(i64, i64), usize> = HashMap::new();
    let mut next_node_id: usize = 0;
    let mut edge_count = 0;
    let mut total_km = 0.0;

    for f in features {
        if let Some(ref geom) = f.geometry {
            let line_strings: Vec<&Vec<Vec<f64>>> = match &geom.value {
                GeoJsonValue::LineString(coords) => vec![coords],
                GeoJsonValue::MultiLineString(multi) => multi.iter().collect(),
                _ => continue,
            };

            for coords in line_strings {
                if coords.len() < 2 {
                    continue;
                }

                // Extract coordinates as (lon, lat) pairs
                let coord_points: Vec<(f64, f64)> = coords
                    .iter()
                    .filter(|p| p.len() >= 2)
                    .map(|p| (p[0], p[1]))
                    .collect();

                // Calculate length for each segment
                for window in coord_points.windows(2) {
                    let (lon1, lat1) = window[0];
                    let (lon2, lat2) = window[1];

                    // Haversine distance in km
                    let d = super::haversine_m(lat1, lon1, lat2, lon2) / 1000.0;
                    total_km += d;

                    // Get/create node IDs
                    let _node1 = get_or_create_node(&mut node_map, &mut next_node_id, lon1, lat1);
                    let _node2 = get_or_create_node(&mut node_map, &mut next_node_id, lon2, lat2);

                    // Count edge
                    edge_count += 1;
                }
            }
        }
    }

    Ok((node_map.len(), edge_count, total_km))
}

/// Get or create a node ID for given coordinates
fn get_or_create_node(
    node_map: &mut HashMap<(i64, i64), usize>,
    next_node_id: &mut usize,
    lon: f64,
    lat: f64,
) -> usize {
    // Snap to ~1m precision (6 decimal places)
    let key = ((lon * 1e6) as i64, (lat * 1e6) as i64);
    *node_map.entry(key).or_insert_with(|| {
        let id = *next_node_id;
        *next_node_id += 1;
        id
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_distance() {
        // Distance from New York to Los Angeles (approx 3935 km)
        let ny_lat = 40.7128;
        let ny_lon = -74.0060;
        let la_lat = 34.0522;
        let la_lon = -118.2437;

        let dist = super::super::haversine_m(ny_lat, ny_lon, la_lat, la_lon) / 1000.0;
        assert!((dist - 3935.0).abs() < 10.0);
    }

    #[test]
    fn test_road_class_conversion() {
        assert_eq!(RoadClass::Motorway.as_str(), "motorway");
        assert_eq!(RoadClass::Residential.as_str(), "residential");
        assert_eq!(RoadClass::Tertiary.as_str(), "tertiary");
    }
}
