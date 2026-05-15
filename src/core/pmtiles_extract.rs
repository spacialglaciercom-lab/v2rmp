use anyhow::{Context, Result};
use geo_types::{Coord, Geometry, LineString, MultiLineString};
use geojson::{Feature, FeatureCollection, Value as GeoJsonValue};
use mvt_reader::Reader as MvtReader;
use pmtiles::{AsyncPmTilesReader, MmapBackend, NoCache, TileCoord};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct PmtilesExtractRequest {
    /// Path to local .pmtiles file
    pub pmtiles_path: String,
    /// West bounding box coordinate
    pub min_lon: f64,
    /// South bounding box coordinate
    pub min_lat: f64,
    /// East bounding box coordinate
    pub max_lon: f64,
    /// North bounding box coordinate
    pub max_lat: f64,
    /// Output GeoJSON path
    pub output_path: String,
    /// Zoom level to extract at (default: max zoom of archive, capped at 14)
    pub zoom: Option<u8>,
    /// Optional layer name filter (extract all layers if None)
    pub layer_name: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PmtilesExtractResult {
    pub features: usize,
    pub tiles_fetched: usize,
    pub output_path: String,
}

/// Convert lon/lat to tile coordinates at a given zoom level.
fn lonlat_to_tile(lon: f64, lat: f64, zoom: u8) -> (u32, u32) {
    let n = 2u32.pow(zoom as u32) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor() as u32;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor() as u32;
    let x = x.min(n as u32 - 1);
    let y = y.min(n as u32 - 1);
    (x, y)
}

/// Convert MVT tile-local coordinates to WGS84 lon/lat.
fn tile_to_wgs84(tile_x: u32, tile_y: u32, zoom: u8, extent: u32, local_x: f64, local_y: f64) -> (f64, f64) {
    let n = 2u32.pow(zoom as u32) as f64;
    let lon_min = tile_x as f64 / n * 360.0 - 180.0;
    let lon_max = (tile_x + 1) as f64 / n * 360.0 - 180.0;
    let lat_max_rad = (std::f64::consts::PI * (1.0 - 2.0 * tile_y as f64 / n)).sinh().atan();
    let lat_max = lat_max_rad.to_degrees();
    let lat_min_rad = (std::f64::consts::PI * (1.0 - 2.0 * (tile_y + 1) as f64 / n)).sinh().atan();
    let lat_min = lat_min_rad.to_degrees();

    let lon = lon_min + (local_x / extent as f64) * (lon_max - lon_min);
    let lat = lat_max + (local_y / extent as f64) * (lat_min - lat_max); // Y is inverted

    (lon, lat)
}

/// Check if any coordinate of a geometry falls within the bbox.
fn geometry_intersects_bbox(geom: &Geometry<f64>, min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> bool {
    match geom {
        Geometry::LineString(ls) => {
            for c in &ls.0 {
                if c.x >= min_lon && c.x <= max_lon && c.y >= min_lat && c.y <= max_lat {
                    return true;
                }
            }
            false
        }
        Geometry::MultiLineString(mls) => {
            for ls in &mls.0 {
                for c in &ls.0 {
                    if c.x >= min_lon && c.x <= max_lon && c.y >= min_lat && c.y <= max_lat {
                        return true;
                    }
                }
            }
            false
        }
        Geometry::Point(p) => {
            p.x() >= min_lon && p.x() <= max_lon && p.y() >= min_lat && p.y() <= max_lat
        }
        _ => false,
    }
}

/// Convert an MVT feature's geometry to WGS84 GeoJSON geometry.
fn convert_geometry(geom: &Geometry<f64>, tile_x: u32, tile_y: u32, zoom: u8, extent: u32) -> Option<geojson::Geometry> {
    match geom {
        Geometry::LineString(ls) => {
            let coords: Vec<Vec<f64>> = ls.0.iter().map(|c| {
                let (lon, lat) = tile_to_wgs84(tile_x, tile_y, zoom, extent, c.x, c.y);
                vec![lon, lat]
            }).collect();
            Some(geojson::Geometry::new(GeoJsonValue::LineString(coords)))
        }
        Geometry::MultiLineString(mls) => {
            let coords: Vec<Vec<Vec<f64>>> = mls.0.iter().map(|ls| {
                ls.0.iter().map(|c| {
                    let (lon, lat) = tile_to_wgs84(tile_x, tile_y, zoom, extent, c.x, c.y);
                    vec![lon, lat]
                }).collect()
            }).collect();
            Some(geojson::Geometry::new(GeoJsonValue::MultiLineString(coords)))
        }
        Geometry::Point(p) => {
            let (lon, lat) = tile_to_wgs84(tile_x, tile_y, zoom, extent, p.x(), p.y());
            Some(geojson::Geometry::new(GeoJsonValue::Point(vec![lon, lat])))
        }
        _ => None,
    }
}

/// Convert geojson::Geometry back to geo_types::Geometry<f64> for bbox checking.
fn geojson_geom_to_geo_types(geom: &geojson::Geometry) -> Geometry<f64> {
    match &geom.value {
        GeoJsonValue::LineString(coords) => {
            let c: Vec<Coord<f64>> = coords.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
            Geometry::LineString(LineString(c))
        }
        GeoJsonValue::MultiLineString(coords) => {
            let lines: Vec<LineString<f64>> = coords.iter().map(|ring| {
                let c: Vec<Coord<f64>> = ring.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
                LineString(c)
            }).collect();
            Geometry::MultiLineString(MultiLineString(lines))
        }
        GeoJsonValue::Point(coords) => {
            Geometry::Point(geo_types::Point::new(coords[0], coords[1]))
        }
        _ => Geometry::Point(geo_types::Point::new(0.0, 0.0)),
    }
}

/// Convert geo_types Geometry<f32> to Geometry<f64> (MVT reader returns f32 by default).
fn convert_geom_to_f64(geom: &Geometry<f32>) -> Geometry<f64> {
    match geom {
        Geometry::LineString(ls) => {
            let coords: Vec<Coord<f64>> = ls.0.iter().map(|c| Coord {
                x: c.x as f64,
                y: c.y as f64,
            }).collect();
            Geometry::LineString(LineString(coords))
        }
        Geometry::MultiLineString(mls) => {
            let lines: Vec<LineString<f64>> = mls.0.iter().map(|ls| {
                let coords: Vec<Coord<f64>> = ls.0.iter().map(|c| Coord {
                    x: c.x as f64,
                    y: c.y as f64,
                }).collect();
                LineString(coords)
            }).collect();
            Geometry::MultiLineString(MultiLineString(lines))
        }
        Geometry::Point(p) => {
            Geometry::Point(geo_types::Point::new(p.x() as f64, p.y() as f64))
        }
        _ => Geometry::Point(geo_types::Point::new(0.0, 0.0)), // fallback
    }
}

/// Extract features from a PMTiles file within a bounding box.
pub async fn run_pmtiles_extract(req: &PmtilesExtractRequest) -> Result<PmtilesExtractResult> {
    let pmtiles_path = Path::new(&req.pmtiles_path);
    if !pmtiles_path.exists() {
        anyhow::bail!("PMTiles file not found: {}", req.pmtiles_path);
    }
    if req.min_lat >= req.max_lat || req.min_lon >= req.max_lon {
        anyhow::bail!("Invalid bbox: min must be less than max");
    }

    // Open the PMTiles archive
    let reader = AsyncPmTilesReader::<MmapBackend, NoCache>::new_with_path(&req.pmtiles_path)
        .await
        .context("Failed to open PMTiles file")?;

    let header = reader.get_header();
    let zoom = req.zoom.unwrap_or_else(|| header.max_zoom.min(14));
    if zoom < header.min_zoom || zoom > header.max_zoom {
        anyhow::bail!(
            "Zoom {} out of range [{}..{}]",
            zoom,
            header.min_zoom,
            header.max_zoom
        );
    }

    tracing::info!(
        "Extracting PMTiles bbox [{:.4},{:.4},{:.4},{:.4}] at zoom {} from {}",
        req.min_lon, req.min_lat, req.max_lon, req.max_lat,
        zoom, req.pmtiles_path
    );

    // Compute tile range
    let (x1, y1) = lonlat_to_tile(req.min_lon, req.max_lat, zoom); // top-left
    let (x2, y2) = lonlat_to_tile(req.max_lon, req.min_lat, zoom); // bottom-right

    tracing::info!("Tile range: z={}, x=[{}..{}], y=[{}..{}]", zoom, x1, x2, y1, y2);

    let mut all_features: Vec<Feature> = Vec::new();
    let mut tiles_fetched: usize = 0;

    for tx in x1..=x2 {
        for ty in y1..=y2 {
            let coord = TileCoord::new(zoom, tx, ty)
                .context("Invalid tile coordinate")?;

            let tile_data = match reader.get_tile_decompressed(coord).await {
                Ok(Some(data)) => data,
                Ok(None) => continue, // no tile at this coord
                Err(e) => {
                    tracing::warn!("Failed to get tile {}/{}/{}: {}", zoom, tx, ty, e);
                    continue;
                }
            };

            tiles_fetched += 1;

            // Decode MVT
            let mvt = match MvtReader::new(tile_data.to_vec()) {
                Ok(mvt) => mvt,
                Err(e) => {
                    tracing::warn!("Failed to decode MVT tile {}/{}/{}: {}", zoom, tx, ty, e);
                    continue;
                }
            };

            let layer_names = match mvt.get_layer_names() {
                Ok(names) => names,
                Err(_) => continue,
            };

            for (layer_idx, layer_name) in layer_names.iter().enumerate() {
                // Filter by layer name if specified
                if let Some(ref filter) = req.layer_name {
                    if layer_name != filter {
                        continue;
                    }
                }

                let features = match mvt.get_features(layer_idx) {
                    Ok(f) => f,
                    Err(_) => continue,
                };

                // Default extent for MVT is 4096
                let layer_meta = match mvt.get_layer_metadata() {
                    Ok(m) => m.into_iter().nth(layer_idx),
                    Err(_) => None,
                };
                let extent = layer_meta.map(|m| m.extent).unwrap_or(4096);

                for mvt_feat in features {
                    // Convert to GeoJSON geometry (also converts to WGS84)
                    let geom_f64 = convert_geom_to_f64(&mvt_feat.geometry);
                    let geojson_geom = match convert_geometry(&geom_f64, tx, ty, zoom, extent) {
                        Some(g) => g,
                        None => continue,
                    };

                    // Extract WGS84 coords and filter by bbox
                    let wgs84_geom = geojson_geom_to_geo_types(&geojson_geom);
                    if !geometry_intersects_bbox(
                        &wgs84_geom,
                        req.min_lon, req.min_lat,
                        req.max_lon, req.max_lat,
                    ) {
                        continue;
                    }

                    // Build properties
                    let mut props = serde_json::Map::new();
                    if let Some(ref mvt_props) = mvt_feat.properties {
                        for (key, val) in mvt_props {
                            let json_val = match val {
                                mvt_reader::feature::Value::String(s) => serde_json::Value::String(s.clone()),
                                mvt_reader::feature::Value::Float(f) => serde_json::json!(f),
                                mvt_reader::feature::Value::Double(d) => serde_json::json!(d),
                                mvt_reader::feature::Value::Int(i) => serde_json::json!(i),
                                mvt_reader::feature::Value::UInt(u) => serde_json::json!(u),
                                mvt_reader::feature::Value::SInt(s) => serde_json::json!(s),
                                mvt_reader::feature::Value::Bool(b) => serde_json::json!(b),
                                mvt_reader::feature::Value::Null => serde_json::Value::Null,
                            };
                            props.insert(key.clone(), json_val);
                        }
                    }
                    props.insert("_layer".to_string(), serde_json::Value::String(layer_name.clone()));
                    props.insert("_zoom".to_string(), serde_json::json!(zoom));
                    props.insert("_tile_x".to_string(), serde_json::json!(tx));
                    props.insert("_tile_y".to_string(), serde_json::json!(ty));

                    let mut feature = Feature::default();
                    feature.geometry = Some(geojson_geom);
                    feature.properties = Some(props);
                    if let Some(id) = mvt_feat.id {
                        feature.id = Some(geojson::feature::Id::Number(serde_json::Number::from(id)));
                    }

                    all_features.push(feature);
                }
            }
        }
    }

    let feature_count = all_features.len();

    // Write GeoJSON
    let fc = FeatureCollection {
        bbox: Some(vec![req.min_lon, req.min_lat, req.max_lon, req.max_lat]),
        features: all_features,
        foreign_members: None,
    };

    let geojson_str = serde_json::to_string(&fc)
        .context("Failed to serialize GeoJSON")?;

    std::fs::write(&req.output_path, geojson_str)
        .context("Failed to write output GeoJSON")?;

    tracing::info!(
        "Extracted {} features from {} tiles → {}",
        feature_count, tiles_fetched, req.output_path
    );

    Ok(PmtilesExtractResult {
        features: feature_count,
        tiles_fetched,
        output_path: req.output_path.clone(),
    })
}
