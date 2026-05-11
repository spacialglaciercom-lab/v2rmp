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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_to_feature_all_fields() {
        let seg = OsmSegment {
            id: 12345,
            name: Some("Main Street".to_string()),
            highway: "primary".to_string(),
            oneway: Some("yes".to_string()),
            surface: Some("asphalt".to_string()),
            geometry: vec![(1.0, 2.0), (3.0, 4.0)],
        };

        let feature = segment_to_feature(seg);

        assert!(feature.id.is_none());
        assert!(feature.bbox.is_none());

        let geometry = feature.geometry.unwrap();
        assert!(geometry.bbox.is_none());
        assert!(geometry.foreign_members.is_none());

        if let GeoJsonValue::LineString(coords) = geometry.value {
            assert_eq!(coords, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        } else {
            panic!("Geometry value is not a LineString");
        }

        let props = feature.properties.unwrap();
        assert_eq!(props.get("id"), Some(&serde_json::Value::Number(12345.into())));
        assert_eq!(props.get("class"), Some(&serde_json::Value::String("primary".to_string())));
        assert_eq!(props.get("name"), Some(&serde_json::Value::String("Main Street".to_string())));
        assert_eq!(props.get("oneway"), Some(&serde_json::Value::String("yes".to_string())));
        assert_eq!(props.get("surface"), Some(&serde_json::Value::String("asphalt".to_string())));
    }

    #[test]
    fn test_segment_to_feature_missing_optional_fields() {
        let seg = OsmSegment {
            id: 54321,
            name: None,
            highway: "residential".to_string(),
            oneway: None,
            surface: None,
            geometry: vec![(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)],
        };

        let feature = segment_to_feature(seg);

        let geometry = feature.geometry.unwrap();
        if let GeoJsonValue::LineString(coords) = geometry.value {
            assert_eq!(coords, vec![vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]]);
        } else {
            panic!("Geometry value is not a LineString");
        }

        let props = feature.properties.unwrap();
        assert_eq!(props.get("id"), Some(&serde_json::Value::Number(54321.into())));
        assert_eq!(props.get("class"), Some(&serde_json::Value::String("residential".to_string())));
        assert!(props.get("name").is_none());
        assert!(props.get("oneway").is_none());
        assert!(props.get("surface").is_none());
    }
}
