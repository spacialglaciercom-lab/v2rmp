use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

use super::{BBox, OsmSegment};

/// Public Overpass API endpoints
pub const OVERPASS_ENDPOINTS: &[&str] = &[
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
];

pub struct OverpassExtractor {
    client: Client,
}

impl Default for OverpassExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl OverpassExtractor {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .user_agent("v2rmp-extractor/0.1")
            .build()
            .unwrap_or_else(|_| Client::new());

        Self { client }
    }

    /// Extract road network from Overpass API with endpoint fallback
    pub async fn extract_bbox(
        &self,
        bbox: &BBox,
        road_classes: &[String],
    ) -> Result<Vec<OsmSegment>> {
        let classes_filter = if road_classes.is_empty() {
            "way[\"highway\"]".to_string()
        } else {
            format!("way[\"highway\"~\"^{}$\"]", road_classes.join("|"))
        };

        // Overpass QL query:
        // 1. Set output format and timeout
        // 2. Query ways with highway tag in bbox
        // 3. Recurse down to get nodes (>)
        // 4. Output body and skeleton
        let query = format!(
            "[out:json][timeout:300];\n(\n  {}({},{},{},{});\n);\nout body;\n>;\nout skel qt;",
            classes_filter, bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon
        );

        let mut last_error = None;

        for endpoint in OVERPASS_ENDPOINTS {
            tracing::info!("Attempting Overpass extraction from: {}", endpoint);
            match self.query_endpoint(endpoint, &query).await {
                Ok(segments) => {
                    tracing::info!(
                        "Successfully extracted {} segments from {}",
                        segments.len(),
                        endpoint
                    );
                    return Ok(segments);
                }
                Err(e) => {
                    tracing::warn!("Overpass endpoint {} failed: {:#}", endpoint, e);
                    last_error = Some(e);
                    // Continue to next endpoint
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All Overpass endpoints failed")))
    }

    async fn query_endpoint(&self, endpoint: &str, query: &str) -> Result<Vec<OsmSegment>> {
        let response = self
            .client
            .post(endpoint)
            .form(&[("data", query)])
            .send()
            .await
            .with_context(|| format!("Failed to connect to Overpass endpoint: {}", endpoint))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!(
                "Overpass endpoint {} returned error {}: {}",
                endpoint,
                status,
                error_text
            );
        }

        let data: Value = response
            .json()
            .await
            .with_context(|| format!("Failed to parse JSON response from {}", endpoint))?;

        self.parse_overpass_json(data)
    }

    fn parse_overpass_json(&self, data: Value) -> Result<Vec<OsmSegment>> {
        let elements = data["elements"]
            .as_array()
            .context("Overpass response missing 'elements' array")?;

        let mut nodes: HashMap<i64, (f64, f64)> = HashMap::new();
        let mut ways_data: Vec<&Value> = Vec::new();

        // First pass: collect nodes
        for el in elements {
            if el["type"] == "node" {
                if let (Some(id), Some(lat), Some(lon)) =
                    (el["id"].as_i64(), el["lat"].as_f64(), el["lon"].as_f64())
                {
                    nodes.insert(id, (lon, lat));
                }
            } else if el["type"] == "way" {
                ways_data.push(el);
            }
        }

        let mut segments = Vec::new();

        // Second pass: build segments from ways
        for way in ways_data {
            let id = way["id"].as_i64().context("Way missing ID")?;
            let tags = &way["tags"];

            let highway = tags["highway"]
                .as_str()
                .unwrap_or("unclassified")
                .to_string();

            let name = tags["name"].as_str().map(|s| s.to_string());
            let oneway = tags["oneway"].as_str().map(|s| s.to_string());
            let surface = tags["surface"].as_str().map(|s| s.to_string());

            let mut geometry = Vec::new();
            if let Some(node_ids) = way["nodes"].as_array() {
                for node_id_val in node_ids {
                    if let Some(node_id) = node_id_val.as_i64() {
                        if let Some(&coords) = nodes.get(&node_id) {
                            geometry.push(coords);
                        }
                    }
                }
            }

            // Only include ways that have at least some geometry
            if geometry.len() >= 2 {
                segments.push(OsmSegment {
                    id,
                    name,
                    highway,
                    oneway,
                    surface,
                    geometry,
                });
            }
        }

        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_overpass_json() {
        let extractor = OverpassExtractor::new();
        let data = json!({
            "elements": [
                { "type": "node", "id": 1, "lat": 40.0, "lon": -74.0 },
                { "type": "node", "id": 2, "lat": 40.1, "lon": -74.1 },
                {
                    "type": "way",
                    "id": 10,
                    "nodes": [1, 2],
                    "tags": {
                        "highway": "residential",
                        "name": "Test Road"
                    }
                }
            ]
        });

        let segments = extractor.parse_overpass_json(data).unwrap();
        assert_eq!(segments.len(), 1);
        let seg = &segments[0];
        assert_eq!(seg.id, 10);
        assert_eq!(seg.highway, "residential");
        assert_eq!(seg.name.as_deref(), Some("Test Road"));
        assert_eq!(seg.geometry.len(), 2);
        assert_eq!(seg.geometry[0], (-74.0, 40.0));
        assert_eq!(seg.geometry[1], (-74.1, 40.1));
    }
}
