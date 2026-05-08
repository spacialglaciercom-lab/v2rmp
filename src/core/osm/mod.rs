use geojson::{Feature, Geometry as GeoJsonGeometry, Value as GeoJsonValue};
use serde::{Deserialize, Serialize};

pub mod overpass_extractor;
pub mod pbf_extractor;

pub use crate::core::geo_types::BBox;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmSegment {
    pub id: i64,
    pub name: Option<String>,
    pub highway: String,
    pub oneway: Option<String>,
    pub surface: Option<String>,
    pub geometry: Vec<(f64, f64)>, // lon, lat pairs
}

/// Convert OSM segment to GeoJSON Feature
pub fn segment_to_feature(seg: OsmSegment) -> Feature {
    let coordinates: Vec<Vec<f64>> = seg
        .geometry
        .into_iter()
        .map(|(lon, lat)| vec![lon, lat])
        .collect();

    let geometry = GeoJsonGeometry {
        bbox: None,
        value: GeoJsonValue::LineString(coordinates),
        foreign_members: None,
    };

    let mut props = serde_json::Map::new();
    props.insert("id".to_string(), serde_json::Value::Number(seg.id.into()));
    props.insert("class".to_string(), serde_json::Value::String(seg.highway));

    if let Some(name) = seg.name {
        props.insert("name".to_string(), serde_json::Value::String(name));
    }
    if let Some(oneway) = seg.oneway {
        props.insert("oneway".to_string(), serde_json::Value::String(oneway));
    }
    if let Some(surface) = seg.surface {
        props.insert("surface".to_string(), serde_json::Value::String(surface));
    }

    Feature {
        id: None,
        bbox: None,
        geometry: Some(geometry),
        properties: Some(props),
        foreign_members: None,
    }
}

pub use overpass_extractor::OverpassExtractor;
pub use pbf_extractor::OsmExtractor;
