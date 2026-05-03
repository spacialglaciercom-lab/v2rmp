use crate::core::osm;
use crate::core::overture::{BBox, OvertureExtractor, OvertureSegment};
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
    /// Path to OSM PBF file (required when source is Osm)
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

/// Extract road network from Overture Maps S3 Parquet files or OSM PBF files.
pub fn run_extract(req: &ExtractRequest) -> anyhow::Result<ExtractResult> {
    match req.source {
        ExtractSource::Overture => run_overture_extract(req),
        ExtractSource::Osm => run_osm_extract(req),
    }
}

/// Run Overture extraction from S3
fn run_overture_extract(req: &ExtractRequest) -> anyhow::Result<ExtractResult> {
    let runtime = tokio::runtime::Runtime::new()
        .context("Failed to create Tokio runtime for async S3 operations")?;

    runtime.block_on(async {
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

        let features: Vec<Feature> = segments
            .into_iter()
            .filter(|seg| should_include_segment(seg, &req.road_classes))
            .map(segment_to_feature)
            .collect::<anyhow::Result<Vec<_>>>()?;

        tracing::info!(
            "After road class filtering: {} road segments",
            features.len()
        );

        finalize_extraction(features, &req.output_path)
    })
}

/// Check if a segment should be included based on road class filter.
fn should_include_segment(seg: &OvertureSegment, classes: &[RoadClass]) -> bool {
    match &seg.class {
        Some(class) => classes.iter().any(|rc| rc.as_str() == class),
        None => false,
    }
}

/// Convert an OvertureSegment to a GeoJSON Feature.
fn segment_to_feature(seg: OvertureSegment) -> anyhow::Result<Feature> {
    use crate::core::overture::Geometry as OvertureGeometry;

    // Build the GeoJSON geometry value
    let geometry_json = match seg.geometry {
        OvertureGeometry::LineString(coords) => json!({
            "type": "LineString",
            "coordinates": coords.into_iter()
                .map(|(lon, lat)| vec![lon, lat])
                .collect::<Vec<_>>()
        }),
        OvertureGeometry::Point(lon, lat) => json!({
            "type": "Point",
            "coordinates": vec![lon, lat]
        }),
    };

    // Convert JSON to geojson::Geometry
    let geometry = GeoJsonGeometry::from_json_value(geometry_json)
        ?;

    // Build properties map
    let mut props = serde_json::Map::new();
    props.insert("id".to_string(), serde_json::Value::String(seg.id));
    if let Some(name) = seg.name {
        props.insert("name".to_string(), serde_json::Value::String(name));
    }
    if let Some(class) = seg.class {
        props.insert("class".to_string(), serde_json::Value::String(class));
    }
    if let Some(subtype) = seg.subtype {
        props.insert("subtype".to_string(), serde_json::Value::String(subtype));
    }
    if let Some(surface) = seg.surface {
        props.insert("surface".to_string(), serde_json::Value::String(surface));
    }
    if let Some(oneway) = seg.oneway {
        props.insert("oneway".to_string(), serde_json::Value::String(oneway));
    }
    if let Some(junction) = seg.junction {
        props.insert("junction".to_string(), serde_json::Value::String(junction));
    }
    if let Some(osm_id) = seg.osm_id {
        props.insert("osm_id".to_string(), serde_json::Value::String(osm_id));
    }

    Ok(Feature {
        id: None,
        bbox: None,
        geometry: Some(geometry),
        properties: Some(props),
        foreign_members: None,
    })
}

/// Build graph statistics from features
fn build_graph_stats(features: &[Feature]) -> Result<(usize, usize, f64)> {
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

                let coord_points: Vec<(f64, f64)> = coords
                    .iter()
                    .filter(|p| p.len() >= 2)
                    .map(|p| (p[0], p[1]))
                    .collect();

                for window in coord_points.windows(2) {
                    let (lon1, lat1) = window[0];
                    let (lon2, lat2) = window[1];

                    let d = haversine_distance_km(lat1, lon1, lat2, lon2);
                    total_km += d;

                    let _node1 = get_or_create_node(&mut node_map, &mut next_node_id, lon1, lat1);
                    let _node2 = get_or_create_node(&mut node_map, &mut next_node_id, lon2, lat2);

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
    let key = ((lon * 1e6) as i64, (lat * 1e6) as i64);
    *node_map.entry(key).or_insert_with(|| {
        let id = *next_node_id;
        *next_node_id += 1;
        id
    })
}

/// Calculate Haversine distance between two points in km
fn haversine_distance_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_KM * c
}

/// Get highway tags from road classes, falling back to all vehicle classes if empty.
fn get_highway_tags(road_classes: &[RoadClass]) -> Vec<String> {
    if road_classes.is_empty() {
        RoadClass::all_vehicle()
            .iter()
            .map(|rc| rc.as_str().to_string())
            .collect()
    } else {
        road_classes
            .iter()
            .map(|rc| rc.as_str().to_string())
            .collect()
    }
}

/// Finalize extraction by building stats, serializing to GeoJSON, and writing to file.
fn finalize_extraction(features: Vec<Feature>, output_path: &str) -> Result<ExtractResult> {
    let (nodes, edges, total_km) = build_graph_stats(&features)?;

    let geojson = FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    let geojson_string = serde_json::to_string_pretty(&geojson)?;

    File::create(output_path)?
        .write_all(geojson_string.as_bytes())
        .context("Failed to write GeoJSON output")?;

    Ok(ExtractResult {
        nodes,
        edges,
        total_km,
        output_path: output_path.to_string(),
    })
}

/// Run OSM PBF extraction from local file
fn run_osm_extract(req: &ExtractRequest) -> anyhow::Result<ExtractResult> {
    let pbf_path = req
        .pbf_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("pbf_path is required for OSM extraction"))?;

    let bbox = osm::BBox {
        min_lon: req.bbox.min_lon,
        min_lat: req.bbox.min_lat,
        max_lon: req.bbox.max_lon,
        max_lat: req.bbox.max_lat,
    };

    tracing::info!(
        "Extracting OSM data from {} for bbox: [{:.4}, {:.4}, {:.4}, {:.4}]",
        pbf_path,
        bbox.min_lon,
        bbox.min_lat,
        bbox.max_lon,
        bbox.max_lat
    );

    let extractor = osm::OsmExtractor::new(pbf_path.clone())?;

    let highway_tags = get_highway_tags(&req.road_classes);
    let segments = extractor.extract_bbox(&bbox, &highway_tags)?;

    tracing::info!("Extracted {} segments from OSM PBF", segments.len());

    let features: Vec<Feature> = segments.into_iter().map(osm::segment_to_feature).collect::<anyhow::Result<Vec<_>>>()?;

    finalize_extraction(features, &req.output_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_distance() {
        let ny_lat = 40.7128;
        let ny_lon = -74.0060;
        let la_lat = 34.0522;
        let la_lon = -118.2437;

        let dist = haversine_distance_km(ny_lat, ny_lon, la_lat, la_lon);
        assert!((dist - 3935.0).abs() < 10.0);
    }

    #[test]
    fn test_road_class_conversion() {
        assert_eq!(RoadClass::Motorway.as_str(), "motorway");
        assert_eq!(RoadClass::Residential.as_str(), "residential");
        assert_eq!(RoadClass::Tertiary.as_str(), "tertiary");
    }

    #[test]
    fn test_should_include_segment() {
        use crate::core::overture::Geometry;
        let seg = OvertureSegment {
            id: "seg1".to_string(),
            name: None,
            class: Some("residential".to_string()),
            subtype: None,
            subclass: None,
            surface: None,
            geometry: Geometry::LineString(vec![(-74.0, 40.7), (-73.9, 40.8)]),
            oneway: None,
            junction: None,
            osm_id: None,
        };
        assert!(should_include_segment(&seg, &[RoadClass::Residential]));
        assert!(!should_include_segment(&seg, &[RoadClass::Motorway]));
    }

    #[test]
    fn test_should_include_segment_no_class() {
        use crate::core::overture::Geometry;
        let seg = OvertureSegment {
            id: "seg2".to_string(),
            name: None,
            class: None,
            subtype: None,
            subclass: None,
            surface: None,
            geometry: Geometry::LineString(vec![(-74.0, 40.7), (-73.9, 40.8)]),
            oneway: None,
            junction: None,
            osm_id: None,
        };
        assert!(!should_include_segment(&seg, &[RoadClass::Residential]));
    }

    #[test]
    fn test_build_graph_stats_basic() {
        let geometry = GeoJsonGeometry {
            bbox: None,
            value: GeoJsonValue::LineString(vec![
                vec![-74.006, 40.7128],
                vec![-73.985, 40.748],
                vec![-73.944, 40.678],
            ]),
            foreign_members: None,
        };
        let feature = Feature {
            id: None,
            bbox: None,
            geometry: Some(geometry),
            properties: Some(json!({"class": "residential"}).as_object().unwrap().clone()),
            foreign_members: None,
        };
        let fc = FeatureCollection {
            bbox: None,
            features: vec![feature],
            foreign_members: None,
        };
        let (nodes, edges, total_km) = build_graph_stats(&fc.features).unwrap();
        assert_eq!(nodes, 3);
        assert_eq!(edges, 2);
        assert!(total_km > 0.0);
        assert!(total_km > 5.0 && total_km < 20.0);
    }

    #[test]
    fn test_build_graph_stats_dedup() {
        let geom1 = GeoJsonGeometry {
            bbox: None,
            value: GeoJsonValue::LineString(vec![vec![-74.006, 40.7128], vec![-73.985, 40.748]]),
            foreign_members: None,
        };
        let feat1 = Feature {
            id: None,
            bbox: None,
            geometry: Some(geom1),
            properties: Some(json!({}).as_object().unwrap().clone()),
            foreign_members: None,
        };
        let geom2 = GeoJsonGeometry {
            bbox: None,
            value: GeoJsonValue::LineString(vec![vec![-73.985, 40.748], vec![-73.944, 40.678]]),
            foreign_members: None,
        };
        let feat2 = Feature {
            id: None,
            bbox: None,
            geometry: Some(geom2),
            properties: Some(json!({}).as_object().unwrap().clone()),
            foreign_members: None,
        };
        let (nodes, edges, _total_km) = build_graph_stats(&[feat1, feat2]).unwrap();
        assert_eq!(nodes, 3);
        assert_eq!(edges, 2);
    }

    #[test]
    fn test_bbox_request_conversion() {
        let req = BBoxRequest {
            min_lon: -122.5,
            min_lat: 37.7,
            max_lon: -122.4,
            max_lat: 37.8,
        };
        let bbox: BBox = req.into();
        assert!((bbox.min_lon - (-122.5)).abs() < f64::EPSILON);
        assert!((bbox.min_lat - 37.7).abs() < f64::EPSILON);
        assert!((bbox.max_lon - (-122.4)).abs() < f64::EPSILON);
        assert!((bbox.max_lat - 37.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extract_result_serialization() {
        let result = ExtractResult {
            nodes: 100,
            edges: 200,
            total_km: 42.5,
            output_path: "/tmp/test.geojson".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ExtractResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nodes, 100);
        assert_eq!(deserialized.edges, 200);
        assert!((deserialized.total_km - 42.5).abs() < f64::EPSILON);
        assert_eq!(deserialized.output_path, "/tmp/test.geojson");
    }
}
