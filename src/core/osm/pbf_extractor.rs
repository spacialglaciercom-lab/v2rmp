use anyhow::{Context, Result};
use osmpbf::{Element, ElementReader};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::{BBox, OsmSegment};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osm_extractor_new_not_found() {
        let path = "non_existent_file.pbf".to_string();
        let result = OsmExtractor::new(path.clone());

        assert!(result.is_err());
        if let Err(err) = result {
            assert_eq!(err.to_string(), format!("PBF file not found: {}", path));
        }
    }
}
