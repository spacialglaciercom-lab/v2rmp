use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileRequest {
    pub input_geojson: String,
    pub output_rmp: String,
    pub compress: bool,
    pub road_classes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    pub input_size_bytes: u64,
    pub output_size_bytes: u64,
    pub node_count: usize,
    pub edge_count: usize,
    pub elapsed_ms: u64,
}

/// .rmp binary format header (magic + version).
const RMP_MAGIC: &[u8; 4] = b"RMP1";

/// Compile a GeoJSON road network into the .rmp binary format.
///
/// Binary layout:
///   [4]  magic "RMP1"
///   [4]  node count (u32 LE)
///   [4]  edge count (u32 LE)
///   [N]  node entries: lat(f64) lon(f64) = 16 bytes each
///   [E]  edge entries: from(u32 LE) to(u32 LE) weight_m(f64 LE) oneway(u8) = 17 bytes each
///   [4]  CRC32 checksum (LE)
pub fn run_compile(req: &CompileRequest) -> anyhow::Result<CompileResult> {
    let start = Instant::now();

    // 1. Read the GeoJSON file
    let mut input_data = Vec::new();
    {
        let mut file = std::fs::File::open(&req.input_geojson)
            .with_context(|| format!("Failed to open input GeoJSON: {}", req.input_geojson))?;
        file.read_to_end(&mut input_data)?;
    }
    let input_size_bytes = input_data.len() as u64;

    // 2. Parse into FeatureCollection
    let geojson: geojson::FeatureCollection = serde_json::from_slice(&input_data)
        .with_context(|| "Failed to parse GeoJSON FeatureCollection")?;

    // 3. Deduplicate nodes by snapping coordinates to 1e6 precision
    //    and 4. Build adjacency list of edges
    let mut node_map: HashMap<u64, u32> = HashMap::new();
    let mut nodes: Vec<(f64, f64)> = Vec::new(); // (lat, lon) in original precision
    let mut edges: Vec<(u32, u32, f64, u8)> = Vec::new(); // (from, to, weight_m, oneway)

    for feature in &geojson.features {
        let geometry = match feature.geometry.as_ref() {
            Some(g) => g,
            None => continue,
        };

        // Determine oneway from properties
        let oneway = feature
            .properties
            .as_ref()
            .and_then(|props| props.get("oneway"))
            .and_then(|v| v.as_str())
            .map(|s| {
                if matches!(s, "yes" | "1" | "true") {
                    1u8
                } else {
                    0u8
                }
            })
            .unwrap_or(0);

        let line_strings: Vec<&Vec<Vec<f64>>> = match &geometry.value {
            geojson::Value::LineString(coords) => vec![coords],
            geojson::Value::MultiLineString(multi) => multi.iter().collect(),
            _ => continue,
        };

        for coords in line_strings {
            if coords.len() < 2 {
                continue;
            }

            let coord_points: Vec<(f64, f64)> = coords
                .iter()
                .filter(|p| p.len() >= 2)
                .map(|p| (p[1], p[0])) // (lat, lon)
                .collect();

            if coord_points.len() < 2 {
                continue;
            }

            let mut it = coord_points.iter();
            if let Some(&(mut lat1, mut lon1)) = it.next() {
                let mut from_node = get_or_create_node(&mut node_map, &mut nodes, lat1, lon1);
                for &(lat2, lon2) in it {
                    let weight_m = haversine_distance_m(lat1, lon1, lat2, lon2);
                    let to_node = get_or_create_node(&mut node_map, &mut nodes, lat2, lon2);
                    edges.push((from_node, to_node, weight_m, oneway));
                    from_node = to_node;
                    lat1 = lat2;
                    lon1 = lon2;
                }
            }
        }
    }

    let node_count = nodes.len();
    let edge_count = edges.len();

    // 5. Write the binary .rmp format
    let mut buf: Vec<u8> = Vec::new();

    // Magic
    buf.extend_from_slice(RMP_MAGIC);

    // Node count
    buf.extend_from_slice(&(node_count as u32).to_le_bytes());

    // Edge count
    buf.extend_from_slice(&(edge_count as u32).to_le_bytes());

    // Node entries: lat(f64 LE) lon(f64 LE) = 16 bytes each
    for (lat, lon) in &nodes {
        buf.extend_from_slice(&lat.to_le_bytes());
        buf.extend_from_slice(&lon.to_le_bytes());
    }

    // Edge entries: from(u32 LE) to(u32 LE) weight_m(f64 LE) oneway(u8) = 17 bytes each
    for (from, to, weight_m, oneway) in &edges {
        buf.extend_from_slice(&from.to_le_bytes());
        buf.extend_from_slice(&to.to_le_bytes());
        buf.extend_from_slice(&weight_m.to_le_bytes());
        buf.push(*oneway);
    }

    // CRC32 checksum (LE)
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    // Write output file
    std::fs::write(&req.output_rmp, &buf)
        .with_context(|| format!("Failed to write output file: {}", req.output_rmp))?;

    let output_size_bytes = buf.len() as u64;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(CompileResult {
        input_size_bytes,
        output_size_bytes,
        node_count,
        edge_count,
        elapsed_ms,
    })
}

/// Quick validation: check if a file starts with the RMP magic bytes.
pub fn is_rmp_file(data: &[u8]) -> bool {
    data.len() >= 4 && &data[..4] == RMP_MAGIC
}

/// Get or create a node ID for the given (lat, lon) coordinates.
/// Snaps to 1e6 precision for deduplication, but stores original-precision coords.
fn get_or_create_node(
    node_map: &mut HashMap<u64, u32>,
    nodes: &mut Vec<(f64, f64)>,
    lat: f64,
    lon: f64,
) -> u32 {
    let lat_i = (lat * 1e6) as i32;
    let lon_i = (lon * 1e6) as i32;
    let key = ((lat_i as u64) << 32) | (lon_i as u32 as u64);
    *node_map.entry(key).or_insert_with(|| {
        let id = nodes.len() as u32;
        nodes.push((lat, lon));
        id
    })
}

/// Calculate haversine distance in meters between two (lat, lon) points.
fn haversine_distance_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_M * c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::optimize::read_rmp_file;

    #[test]
    fn test_is_rmp_file_valid() {
        let data = b"RMP1\x00\x00\x00\x00\x00\x00\x00\x00extra";
        assert!(is_rmp_file(data));
    }

    #[test]
    fn test_is_rmp_file_invalid() {
        let data = b"GARBAGE_data_here";
        assert!(!is_rmp_file(data));
    }

    #[test]
    fn test_compile_roundtrip() {
        // Create a simple GeoJSON FeatureCollection with 2 LineString features
        let geojson = serde_json::json!({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-73.985, 40.748], [-73.944, 40.678]]
                    },
                    "properties": {"class": "residential"}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-73.944, 40.678], [-74.006, 40.7128]]
                    },
                    "properties": {"class": "primary", "oneway": "yes"}
                }
            ]
        });

        let input_path = "/tmp/v2rmp_test_compile_input.geojson";
        let output_path = "/tmp/v2rmp_test_compile_output.rmp";

        // Write input GeoJSON
        std::fs::write(input_path, serde_json::to_string_pretty(&geojson).unwrap()).unwrap();

        let req = CompileRequest {
            input_geojson: input_path.to_string(),
            output_rmp: output_path.to_string(),
            compress: false,
            road_classes: vec![],
        };

        let result = run_compile(&req).unwrap();

        // Verify compile result
        assert_eq!(result.node_count, 3); // 3 unique nodes (middle one shared)
        assert_eq!(result.edge_count, 2); // 2 edges

        // Read the output file and verify magic bytes
        let output_data = std::fs::read(output_path).unwrap();
        assert!(output_data.len() >= 4);
        assert_eq!(&output_data[..4], b"RMP1");

        // Parse with read_rmp_file
        let (nodes, edges) = read_rmp_file(&output_data).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);

        // Verify edge data
        assert_eq!(edges[0].from, 0);
        assert_eq!(edges[0].to, 1);
        assert_eq!(edges[1].from, 1);
        assert_eq!(edges[1].to, 2);
        // Second edge has oneway=yes
        assert_eq!(edges[1].oneway, 1);

        // Clean up
        let _ = std::fs::remove_file(input_path);
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_haversine_distance_m() {
        // NYC to LA: ~3,935 km
        let dist = haversine_distance_m(40.7128, -74.0060, 34.0522, -118.2437);
        assert!((dist - 3_935_000.0).abs() < 10_000.0);
    }

    #[test]
    fn test_run_compile_non_existent_file() {
        let req = CompileRequest {
            input_geojson: "/tmp/non_existent_file.geojson".to_string(),
            output_rmp: "/tmp/output.rmp".to_string(),
            compress: false,
            road_classes: vec![],
        };

        let result = run_compile(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to open input GeoJSON"));
    }

    #[test]
    fn test_run_compile_invalid_geojson() {
        let input_path = "/tmp/v2rmp_test_invalid.geojson";
        std::fs::write(input_path, "invalid json").unwrap();

        let req = CompileRequest {
            input_geojson: input_path.to_string(),
            output_rmp: "/tmp/output.rmp".to_string(),
            compress: false,
            road_classes: vec![],
        };

        let result = run_compile(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse GeoJSON FeatureCollection"));

        let _ = std::fs::remove_file(input_path);
    }
}
