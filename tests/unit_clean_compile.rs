// Unit tests for clean + compile using synthetic GeoJSON data.
// Run with: cargo test --test unit_clean_compile --features extract

fn synthetic_city_grid_fc(num_blocks: usize, spacing_m: f64) -> geojson::FeatureCollection {
    let m_per_deg_lat = 111_320.0;
    let m_per_deg_lon = 111_320.0 * 45.0f64.to_radians().cos();

    let mut features = vec![];

    for i in 0..=num_blocks {
        let lat = 45.0 + (i as f64 * spacing_m) / m_per_deg_lat;
        for j in 0..num_blocks {
            let lon0 = (j as f64 * spacing_m) / m_per_deg_lon;
            let lon1 = ((j + 1) as f64 * spacing_m) / m_per_deg_lon;
            let mut props = serde_json::Map::new();
            props.insert(
                "highway".into(),
                serde_json::Value::String("residential".into()),
            );
            props.insert("oneway".into(), serde_json::Value::String("no".into()));
            features.push(geojson::Feature {
                bbox: None,
                geometry: Some(geojson::Geometry::new(geojson::Value::LineString(vec![
                    vec![lon0, lat],
                    vec![lon1, lat],
                ]))),
                id: None,
                properties: Some(props),
                foreign_members: None,
            });
        }
    }

    for j in 0..=num_blocks {
        let lon = (j as f64 * spacing_m) / m_per_deg_lon;
        for i in 0..num_blocks {
            let lat0 = 45.0 + (i as f64 * spacing_m) / m_per_deg_lat;
            let lat1 = 45.0 + ((i + 1) as f64 * spacing_m) / m_per_deg_lat;
            let mut props = serde_json::Map::new();
            props.insert(
                "highway".into(),
                serde_json::Value::String("tertiary".into()),
            );
            props.insert("oneway".into(), serde_json::Value::String("no".into()));
            features.push(geojson::Feature {
                bbox: None,
                geometry: Some(geojson::Geometry::new(geojson::Value::LineString(vec![
                    vec![lon, lat0],
                    vec![lon, lat1],
                ]))),
                id: None,
                properties: Some(props),
                foreign_members: None,
            });
        }
    }

    geojson::FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    }
}

#[test]
fn test_clean_city_grid() {
    let raw = synthetic_city_grid_fc(3, 200.0);
    let raw_count = raw.features.len();
    assert_eq!(raw_count, 24, "3x3 grid should have 24 segments");

    let options = v2rmp::core::clean::CleanOptions::default();
    let (cleaned, stats, _warnings) = v2rmp::core::clean::clean_geojson(&raw, &options).unwrap();

    assert_eq!(stats.input_features, raw_count);
    assert!(
        cleaned.features.len() <= raw_count,
        "cleaned should have <= features than raw"
    );
    assert!(
        stats.output_features > 0,
        "cleaned should produce at least one feature"
    );
}

#[test]
fn test_compile_city_grid() {
    use std::fs;
    use v2rmp::core::compile::{run_compile, CompileRequest};

    let raw = synthetic_city_grid_fc(2, 200.0);
    let geojson_path = std::path::PathBuf::from("mcp_test_data/test_compile_grid.geojson");
    let rmp_path = std::path::PathBuf::from("mcp_test_data/test_compile_grid.rmp");
    fs::create_dir_all("mcp_test_data").unwrap();

    let geojson_str = serde_json::to_string(&raw).unwrap();
    fs::write(&geojson_path, &geojson_str).unwrap();

    let req = CompileRequest {
        input_geojson: geojson_path.to_string_lossy().to_string(),
        output_rmp: rmp_path.to_string_lossy().to_string(),
        compress: false,
        road_classes: vec![],
        clean_options: None,
        prune_disconnected: false,
        prune_spurs: false,
    };

    let result = run_compile(&req).unwrap();
    assert!(result.node_count > 0, "compiled should have nodes");
    assert!(result.edge_count > 0, "compiled should have edges");
    assert!(
        result.output_size_bytes > 0,
        "compiled should produce non-empty file"
    );

    let file_data = fs::read(&rmp_path).unwrap();
    let (nodes, edges) = v2rmp::core::optimize::read_rmp_file(&file_data).unwrap();
    assert_eq!(
        nodes.len(),
        result.node_count,
        "read_rmp node count mismatch"
    );
    assert_eq!(
        edges.len(),
        result.edge_count,
        "read_rmp edge count mismatch"
    );

    let (min_lat, max_lat, _min_lon, max_lon) = nodes.iter().fold(
        (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
        |(mn_lat, mx_lat, mn_lon, mx_lon), n| {
            (
                mn_lat.min(n.lat),
                mx_lat.max(n.lat),
                mn_lon.min(n.lon),
                mx_lon.max(n.lon),
            )
        },
    );
    assert!(
        min_lat > 44.9 && max_lat < 45.3,
        "bbox lat should be near 45.0"
    );
    assert!(max_lon > 0.0, "bbox lon should be > 0");
}
