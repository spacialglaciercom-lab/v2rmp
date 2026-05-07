#![allow(dead_code)]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_macros,
    clippy::all
)]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_macros,
    clippy::all
)]
//! Overture Maps S3 extractor
//!
//! Queries Overture Maps transportation segments from S3 Parquet files,
//! filters by bounding box and road class, and builds a road network graph.

use anyhow::{Context, Result};
use arrow::array::{Array, BinaryArray, BooleanArray, StringArray, StructArray};
use futures_util::stream::{self, StreamExt};
use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, PointTrait,
};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::sync::Arc;
use wkb::reader::read_wkb;

/// Overture Maps S3 configuration
pub const OVERTURE_S3_BUCKET: &str = "overturemaps-us-west-2";
pub const OVERTURE_S3_REGION: &str = "us-west-2";
pub const OVERTURE_RELEASE: &str = "2026-04-15.0";

/// Maximum number of retries for individual file downloads
/// Delay between retries for file downloads (milliseconds)
pub fn segment_path() -> String {
    format!(
        "release/{}/theme=transportation/type=segment/",
        OVERTURE_RELEASE
    )
}



/// Bounding box for spatial filtering
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

impl BBox {
    /// Check if a point is within the bounding box
    #[allow(dead_code)]
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lon >= self.min_lon && lon <= self.max_lon && lat >= self.min_lat && lat <= self.max_lat
    }

    /// Check if a bbox intersects this bbox
    pub fn intersects(&self, other: &BBox) -> bool {
        !(self.max_lon < other.min_lon
            || self.min_lon > other.max_lon
            || self.max_lat < other.min_lat
            || self.min_lat > other.max_lat)
    }

    /// Calculate area in degrees squared (approximate)
    #[allow(dead_code)]
    pub fn area(&self) -> f64 {
        (self.max_lon - self.min_lon) * (self.max_lat - self.min_lat)
    }
}

/// Overture segment extracted from S3
#[derive(Debug, Clone)]
pub struct OvertureSegment {
    pub id: String,
    pub name: Option<String>,
    pub class: Option<String>,
    pub subtype: Option<String>,
    #[allow(dead_code)]
    pub subclass: Option<String>,
    pub surface: Option<String>,
    pub geometry: Geometry,
    pub oneway: Option<String>,
    pub junction: Option<String>,
    pub osm_id: Option<String>,
}

/// Geometry representation
#[derive(Debug, Clone)]
pub enum Geometry {
    LineString(Vec<(f64, f64)>), // lon, lat
    Point(f64, f64),             // lon, lat
}

impl Geometry {
    /// Get bounding box of the geometry
    pub fn bbox(&self) -> Option<BBox> {
        match self {
            Geometry::LineString(coords) => {
                if coords.is_empty() {
                    return None;
                }
                let mut min_lon = f64::MAX;
                let mut max_lon = f64::MIN;
                let mut min_lat = f64::MAX;
                let mut max_lat = f64::MIN;
                for (lon, lat) in coords {
                    min_lon = min_lon.min(*lon);
                    max_lon = max_lon.max(*lon);
                    min_lat = min_lat.min(*lat);
                    max_lat = max_lat.max(*lat);
                }
                Some(BBox {
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                })
            }
            Geometry::Point(lon, lat) => Some(BBox {
                min_lon: *lon,
                min_lat: *lat,
                max_lon: *lon,
                max_lat: *lat,
            }),
        }
    }
}

/// Overture S3 extractor
pub struct OvertureExtractor {
    store: Arc<dyn ObjectStore>,
}

impl OvertureExtractor {
    /// Create a new extractor connected to Overture Maps S3
    pub fn new() -> Result<Self> {
        let client_options =
            object_store::ClientOptions::new().with_timeout(std::time::Duration::from_secs(300));

        let store = AmazonS3Builder::new()
            .with_bucket_name(OVERTURE_S3_BUCKET)
            .with_region(OVERTURE_S3_REGION)
            .with_skip_signature(true)
            .with_client_options(client_options)
            .build()
            .context(
                "Failed to create S3 client for Overture Maps. \
                      Check network connectivity and DNS resolution for \
                      s3.us-west-2.amazonaws.com",
            )?;

        Ok(Self {
            store: Arc::new(store),
        })
    }

    /// Create a new extractor with custom credentials (for testing)
    #[allow(dead_code)]
    pub fn with_credentials(
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
    ) -> Result<Self> {
        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(OVERTURE_S3_BUCKET)
            .with_region(OVERTURE_S3_REGION)
            .with_allow_http(true);

        if let (Some(akid), Some(sak)) = (access_key_id, secret_access_key) {
            builder = builder.with_access_key_id(akid).with_secret_access_key(sak);
        }

        let store = builder.build()?;

        Ok(Self {
            store: Arc::new(store),
        })
    }

    /// Extract segments within a bounding box
    pub async fn extract_bbox(&self, bbox: &BBox) -> Result<Vec<OvertureSegment>> {
        // List all parquet files in the segment directory
        let prefix = Path::from(segment_path());
        let mut files = Vec::new();

        let mut stream = self.store.list(Some(&prefix));

        while let Some(item) = stream.next().await {
            let meta = item.context(
                "Failed to list S3 objects. \
                                      This usually indicates a network connectivity issue \
                                      or DNS resolution failure for s3.us-west-2.amazonaws.com",
            )?;
            if meta.location.to_string().ends_with(".parquet") {
                files.push(meta.location);
            }
        }

        tracing::info!(
            "Found {} parquet files in Overture segment directory",
            files.len()
        );

        let mut segments = Vec::new();

        // Process files in parallel with reduced concurrency to avoid S3 rate limiting
        let file_results = stream::iter(files)
            .map(|path| {
                let store = Arc::clone(&self.store);
                let bbox = *bbox;
                async move {
                    tracing::info!("Processing file: {}", path);
                    Self::extract_from_file(store.as_ref(), &path, &bbox).await
                }
            })
            .buffer_unordered(4) // Reduced from 10 to avoid overwhelming S3
            .collect::<Vec<_>>()
            .await;

        for result in file_results {
            match result {
                Ok(mut file_segments) => segments.append(&mut file_segments),
                Err(e) => {
                    tracing::warn!("Failed to extract from file: {}", e);
                }
            }
        }

        tracing::info!("Extracted {} total segments from S3", segments.len());
        Ok(segments)
    }

    /// Extract segments from a single file with retry logic
    #[allow(dead_code)]
    #[allow(dead_code)]
    #[allow(dead_code)]
    #[allow(dead_code)]
    #[allow(dead_code)]
    /// Check if an error is retryable (network/timeout related)
    #[allow(dead_code)]
    #[allow(dead_code)]
    #[allow(dead_code)]
    #[allow(dead_code)]
    #[allow(dead_code)]

    /// Convert a WKB geometry (via geo_traits GeometryTrait) into our Geometry enum.
    ///
    /// Returns `None` for unsupported geometry types (Polygon, MultiPolygon, etc.)
    fn convert_wkb_geometry(geom: &impl GeometryTrait<T = f64>) -> Option<Geometry> {
        match geom.as_type() {
            GeometryType::Point(pt) => {
                let coord = pt.coord()?;
                Some(Geometry::Point(coord.x(), coord.y()))
            }
            GeometryType::LineString(ls) => {
                let coords: Vec<(f64, f64)> = (0..ls.num_coords())
                    .map(|i| {
                        let c = unsafe { ls.coord_unchecked(i) };
                        (c.x(), c.y())
                    })
                    .collect();
                Some(Geometry::LineString(coords))
            }
            GeometryType::MultiLineString(mls) => {
                // Take first LineString from MultiLineString
                if mls.num_line_strings() == 0 {
                    return None;
                }
                let ls = unsafe { mls.line_string_unchecked(0) };
                let coords: Vec<(f64, f64)> = (0..ls.num_coords())
                    .map(|i| {
                        let c = unsafe { ls.coord_unchecked(i) };
                        (c.x(), c.y())
                    })
                    .collect();
                Some(Geometry::LineString(coords))
            }
            GeometryType::MultiPoint(mpts) => {
                if mpts.num_points() == 0 {
                    return None;
                }
                let pt = unsafe { mpts.point_unchecked(0) };
                let coord = pt.coord()?;
                Some(Geometry::Point(coord.x(), coord.y()))
            }
            _ => None,
        }
    }

    /// Extract segments from a single parquet file
    async fn extract_from_file(
        store: &dyn ObjectStore,
        path: &Path,
        bbox: &BBox,
    ) -> Result<Vec<OvertureSegment>> {
        // Download the file
        let bytes = store
            .get(path)
            .await
            .with_context(|| format!("Failed to initiate download from S3 for {}", path))?
            .bytes()
            .await
            .with_context(|| format!("Failed to download file content from S3 for {}", path))?;

        // Create parquet reader
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .with_context(|| format!("Failed to create Parquet reader for {}", path))?
            .with_batch_size(1024)
            .build()
            .with_context(|| format!("Failed to build Parquet reader for {}", path))?;

        let mut segments = Vec::new();

        for batch in reader {
            let batch = batch?;
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }

            // Get columns by name
            let id_col = batch.column_by_name("id");
            let names_col = batch.column_by_name("names");
            let class_col = batch.column_by_name("class");
            let road_class_col = batch.column_by_name("road_class");
            let subtype_col = batch.column_by_name("subtype");
            let subclass_col = batch.column_by_name("subclass");
            let surface_col = batch.column_by_name("surface");
            let geometry_col = batch.column_by_name("geometry");
            let oneway_col = batch.column_by_name("oneway");
            let directed_col = batch.column_by_name("directed");
            let junction_col = batch.column_by_name("junction");
            let osm_id_col = batch.column_by_name("osm_id");
            let sources_col = batch.column_by_name("sources");

            // Cast columns to expected Arrow types
            let id_arr = id_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let class_arr = class_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let road_class_arr =
                road_class_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let subtype_arr = subtype_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let subclass_arr = subclass_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let surface_arr = surface_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let geometry_arr = geometry_col.and_then(|c| c.as_any().downcast_ref::<BinaryArray>());
            let oneway_arr = oneway_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let directed_str_arr =
                directed_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let directed_bool_arr =
                directed_col.and_then(|c| c.as_any().downcast_ref::<BooleanArray>());
            let junction_arr = junction_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let osm_id_arr = osm_id_col.and_then(|c| c.as_any().downcast_ref::<StringArray>());

            // Extract nested names.primary or names.primary.value
            let name_values = names_col.and_then(|c| {
                let struct_arr = c.as_any().downcast_ref::<StructArray>()?;
                let primary = struct_arr.column_by_name("primary")?;
                // Try names.primary.value first (doubly nested)
                if let Some(primary_struct) = primary.as_any().downcast_ref::<StructArray>() {
                    if let Some(value_col) = primary_struct.column_by_name("value") {
                        return Some(value_col.clone());
                    }
                }
                // Try names.primary directly as StringArray
                Some(primary.clone())
            });
            let name_arr = name_values
                .as_ref()
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            // Extract nested sources (for osm_id fallback)
            let source_values = sources_col.and_then(|c| {
                let struct_arr = c.as_any().downcast_ref::<StructArray>()?;
                if let Some(dataset_col) = struct_arr.column_by_name("dataset") {
                    return Some(dataset_col.clone());
                }
                None
            });
            let source_arr = source_values
                .as_ref()
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            for row_idx in 0..num_rows {
                // Extract id (required)
                let Some(id_arr) = id_arr else { continue };
                if id_arr.is_null(row_idx) {
                    continue;
                }
                let id = id_arr.value(row_idx).to_string();

                // Extract name (optional, nested)
                let name = name_arr.and_then(|arr| {
                    if arr.is_null(row_idx) {
                        None
                    } else {
                        Some(arr.value(row_idx).to_string())
                    }
                });

                // Extract class (try "class" then "road_class")
                let class = class_arr
                    .and_then(|arr| {
                        if arr.is_null(row_idx) {
                            None
                        } else {
                            Some(arr.value(row_idx).to_string())
                        }
                    })
                    .or_else(|| {
                        road_class_arr.and_then(|arr| {
                            if arr.is_null(row_idx) {
                                None
                            } else {
                                Some(arr.value(row_idx).to_string())
                            }
                        })
                    });

                // Extract subtype, subclass, surface (all optional)
                let subtype = subtype_arr.and_then(|arr| {
                    if arr.is_null(row_idx) {
                        None
                    } else {
                        Some(arr.value(row_idx).to_string())
                    }
                });
                let subclass = subclass_arr.and_then(|arr| {
                    if arr.is_null(row_idx) {
                        None
                    } else {
                        Some(arr.value(row_idx).to_string())
                    }
                });
                let surface = surface_arr.and_then(|arr| {
                    if arr.is_null(row_idx) {
                        None
                    } else {
                        Some(arr.value(row_idx).to_string())
                    }
                });

                // Extract oneway (try "oneway" as string, then "directed" as string or bool)
                let oneway = oneway_arr
                    .and_then(|arr| {
                        if arr.is_null(row_idx) {
                            None
                        } else {
                            Some(arr.value(row_idx).to_string())
                        }
                    })
                    .or_else(|| {
                        directed_str_arr.and_then(|arr| {
                            if arr.is_null(row_idx) {
                                None
                            } else {
                                Some(arr.value(row_idx).to_string())
                            }
                        })
                    })
                    .or_else(|| {
                        directed_bool_arr.and_then(|arr| {
                            if arr.is_null(row_idx) {
                                None
                            } else {
                                Some(arr.value(row_idx).to_string())
                            }
                        })
                    });

                let junction = junction_arr.and_then(|arr| {
                    if arr.is_null(row_idx) {
                        None
                    } else {
                        Some(arr.value(row_idx).to_string())
                    }
                });

                // Extract osm_id (try column directly, then nested sources)
                let osm_id = osm_id_arr
                    .and_then(|arr| {
                        if arr.is_null(row_idx) {
                            None
                        } else {
                            Some(arr.value(row_idx).to_string())
                        }
                    })
                    .or_else(|| {
                        source_arr.and_then(|arr| {
                            if arr.is_null(row_idx) {
                                None
                            } else {
                                Some(arr.value(row_idx).to_string())
                            }
                        })
                    });

                // Decode WKB geometry
                let Some(geom_arr) = geometry_arr else {
                    continue;
                };
                if geom_arr.is_null(row_idx) {
                    continue;
                }
                let wkb_bytes = geom_arr.value(row_idx);

                let wkb_geom = match read_wkb(wkb_bytes) {
                    Ok(g) => g,
                    Err(e) => {
                        tracing::debug!("WKB decode error for id {}: {}", id, e);
                        continue;
                    }
                };

                let Some(geometry) = Self::convert_wkb_geometry(&wkb_geom) else {
                    tracing::debug!("Unsupported geometry type for id {}", id);
                    continue;
                };

                // Filter by bounding box: skip segments whose geometry bbox doesn't intersect
                if let Some(geom_bbox) = geometry.bbox() {
                    if !bbox.intersects(&geom_bbox) {
                        continue;
                    }
                }

                segments.push(OvertureSegment {
                    id,
                    name,
                    class,
                    subtype,
                    subclass,
                    surface,
                    geometry,
                    oneway,
                    junction,
                    osm_id,
                });
            }
        }

        Ok(segments)
    }
}

impl Default for OvertureExtractor {
    fn default() -> Self {
        Self::new().expect("Failed to create Overture extractor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_contains() {
        let bbox = BBox {
            min_lon: -122.5,
            min_lat: 37.7,
            max_lon: -122.4,
            max_lat: 37.8,
        };
        assert!(bbox.contains(-122.45, 37.75));
        assert!(!bbox.contains(-122.6, 37.75));
    }

    #[test]
    fn test_bbox_intersects() {
        let bbox1 = BBox {
            min_lon: -122.5,
            min_lat: 37.7,
            max_lon: -122.4,
            max_lat: 37.8,
        };
        let bbox2 = BBox {
            min_lon: -122.45,
            min_lat: 37.75,
            max_lon: -122.35,
            max_lat: 37.85,
        };
        assert!(bbox1.intersects(&bbox2));

        let bbox3 = BBox {
            min_lon: -122.6,
            min_lat: 37.7,
            max_lon: -122.55,
            max_lat: 37.75,
        };
        assert!(!bbox1.intersects(&bbox3));
    }

    #[test]
    fn test_geometry_bbox() {
        let geom = Geometry::LineString(vec![(-122.5, 37.7), (-122.4, 37.8), (-122.45, 37.75)]);
        let bbox = geom.bbox().unwrap();
        assert_eq!(bbox.min_lon, -122.5);
        assert_eq!(bbox.max_lon, -122.4);
        assert_eq!(bbox.min_lat, 37.7);
        assert_eq!(bbox.max_lat, 37.8);
    }
}
