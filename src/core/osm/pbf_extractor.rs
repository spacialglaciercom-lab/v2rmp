use anyhow::{Context, Result};
use geojson::{Feature, Geometry as GeoJsonGeometry, Value as GeoJsonValue};
use osmpbf::{Element, ElementReader};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct BBox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

impl BBox {
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lon >= self.min_lon && lon <= self.max_lon && lat >= self.min_lat && lat <= self.max_lat
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsmSegment {
    pub id: i64,
    pub name: Option<String>,
    pub highway: String,
    pub oneway: Option<String>,
    pub surface: Option<String>,
    pub geometry: Vec<(f64, f64)>, // lon, lat pairs
}

#[derive(Debug)]
pub struct OsmExtractor {
    pbf_path: String,
}

impl OsmExtractor {
    pub fn new(pbf_path: String) -> Result<Self> {
        if !Path::new(&pbf_path).exists() {
            anyhow::bail!("PBF file not found: {}", pbf_path);
        }
        Ok(Self { pbf_path })
    }

    pub fn extract_bbox(&self, bbox: &BBox, road_classes: &[String]) -> Result<Vec<OsmSegment>> {
        let mut nodes: HashMap<i64, (f64, f64)> = HashMap::new();
        let mut segments: Vec<OsmSegment> = Vec::new();

        tracing::info!("Reading PBF file: {}", self.pbf_path);

        // First pass: collect nodes within bbox
        let file = File::open(&self.pbf_path)
            .context(format!("Failed to open PBF file: {}", self.pbf_path))?;

        let reader = ElementReader::new(file);

        reader.for_each(|element| match element {
            Element::Node(node) => {
                let lon = node.lon();
                let lat = node.lat();
                if bbox.contains(lon, lat) {
                    nodes.insert(node.id(), (lon, lat));
                }
            }
            Element::DenseNode(node) => {
                let lon = node.lon();
                let lat = node.lat();
                if bbox.contains(lon, lat) {
                    nodes.insert(node.id(), (lon, lat));
                }
            }
            _ => {}
        })?;

        tracing::info!("Collected {} nodes within bbox", nodes.len());

        // Second pass: collect ways (roads)
        let file = File::open(&self.pbf_path)?;
        let reader = ElementReader::new(file);

        reader.for_each(|element| {
            if let Element::Way(way) = element {
                // Check if it's a road
                let mut highway_val: Option<&str> = None;
                let mut name_val: Option<&str> = None;
                let mut oneway_val: Option<&str> = None;
                let mut surface_val: Option<&str> = None;

                for (key, value) in way.tags() {
                    match key {
                        "highway" => highway_val = Some(value),
                        "name" => name_val = Some(value),
                        "oneway" => oneway_val = Some(value),
                        "surface" => surface_val = Some(value),
                        _ => {}
                    }
                }

                // Filter by highway tag
                if let Some(hw) = highway_val {
                    if !road_classes.is_empty() && !road_classes.iter().any(|rc| rc == hw) {
                        return;
                    }

                    // Build geometry from node refs
                    let mut geometry: Vec<(f64, f64)> = Vec::new();
                    for node_id in way.refs() {
                        if let Some(&coords) = nodes.get(&node_id) {
                            geometry.push(coords);
                        }
                    }

                    // Only include ways with at least 2 nodes in bbox
                    if geometry.len() >= 2 {
                        segments.push(OsmSegment {
                            id: way.id(),
                            name: name_val.map(|s| s.to_string()),
                            highway: hw.to_string(),
                            oneway: oneway_val.map(|s| s.to_string()),
                            surface: surface_val.map(|s| s.to_string()),
                            geometry,
                        });
                    }
                }
            }
        })?;

        tracing::info!("Extracted {} road segments", segments.len());

        Ok(segments)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_contains() {
        let bbox = BBox {
            min_lon: -74.0,
            min_lat: 40.7,
            max_lon: -73.9,
            max_lat: 40.8,
        };
        assert!(bbox.contains(-73.95, 40.75));
        assert!(!bbox.contains(-74.1, 40.75));
        assert!(!bbox.contains(-73.95, 40.9));
    }

    #[test]
    fn test_segment_to_feature() {
        let seg = OsmSegment {
            id: 12345,
            name: Some("Main Street".to_string()),
            highway: "residential".to_string(),
            oneway: Some("yes".to_string()),
            surface: Some("asphalt".to_string()),
            geometry: vec![(-74.0, 40.7), (-73.9, 40.8)],
        };

        let feature = segment_to_feature(seg);
        assert!(feature.geometry.is_some());

        let props = feature.properties.unwrap();
        assert_eq!(props.get("class").unwrap().as_str().unwrap(), "residential");
        assert_eq!(props.get("name").unwrap().as_str().unwrap(), "Main Street");
        assert_eq!(props.get("oneway").unwrap().as_str().unwrap(), "yes");
    }

    #[test]
    fn test_osm_extractor_new_invalid_path() {
        let result = OsmExtractor::new("non_existent_file.osm.pbf".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PBF file not found"));
    }

    #[test]
    fn test_osm_extractor_new_valid_path() {
        // Use the monaco.osm.pbf file that exists in the root
        let result = OsmExtractor::new("monaco.osm.pbf".to_string());
        assert!(result.is_ok());
    }
}
