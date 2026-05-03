//! Overture Maps S3 extractor

use anyhow::Result;
use futures_util::stream::{self, StreamExt};
use geo_traits::{CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::sync::Arc;

/// Overture Maps S3 configuration
pub const OVERTURE_S3_BUCKET: &str = "overturemaps-us-west-2";
pub const OVERTURE_S3_REGION: &str = "us-west-2";
pub const OVERTURE_RELEASE: &str = "2026-04-15.0";

pub fn segment_path() -> String {
    format!(
        "release/{}/theme=transportation/type=segment/",
        OVERTURE_RELEASE
    )
}

/// Supported road classes for extraction
pub const ROAD_CLASSES: &[&str] = &[
    "residential",
    "tertiary",
    "secondary",
    "primary",
    "trunk",
    "motorway",
    "unclassified",
    "living_street",
    "service",
    "secondary_link",
    "primary_link",
    "trunk_link",
    "motorway_link",
];

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
        let store = AmazonS3Builder::new()
            .with_bucket_name(OVERTURE_S3_BUCKET)
            .with_region(OVERTURE_S3_REGION)
            .with_allow_http(true)
            .build()?;

        Ok(Self {
            store: Arc::new(store),
        })
    }

    /// Create a new extractor with custom credentials (for testing)
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

    /// Extract segments from S3 for the given bounding box.
    pub async fn extract_bbox(&self, bbox: &BBox) -> Result<Vec<OvertureSegment>> {
        let prefix = segment_path();
        let path = Path::from(prefix.as_str());

        let list_stream = self.store.list(Some(&path));
        let objects: Vec<_> = list_stream.collect().await;
        let objects: Vec<_> = objects.into_iter().filter_map(|r| r.ok()).collect();

        tracing::info!("Found {} S3 objects to process", objects.len());

        let segments = stream::iter(objects)
            .map(|meta| async move {
                let location = meta.location;
                self.process_parquet_file(&location, bbox).await
            })
            .buffer_unordered(10)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        Ok(segments)
    }

    /// Process a single Parquet file from S3.
    async fn process_parquet_file(
        &self,
        location: &Path,
        bbox: &BBox,
    ) -> Result<Vec<OvertureSegment>> {
        let data = self.store.get(location).await?.bytes().await?;

        let reader = ParquetRecordBatchReaderBuilder::try_new(data)?.build()?;

        let mut segments = Vec::new();

        for batch in reader {
            let batch = match batch {
                Ok(b) => b,
                Err(e) => {
                    tracing::debug!("Error reading batch: {}", e);
                    continue;
                }
            };

            use arrow::array::{Array, BinaryArray, BooleanArray, StringArray};

            let id_arr = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let road_class_arr = batch
                .column_by_name("road_class")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let class_arr = batch
                .column_by_name("class")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let subtype_arr = batch
                .column_by_name("subtype")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let subclass_arr = batch
                .column_by_name("subclass")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let surface_arr = batch
                .column_by_name("surface")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let oneway_arr = batch
                .column_by_name("oneway")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let directed_str_arr = batch
                .column_by_name("directed")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let directed_bool_arr = batch
                .column_by_name("directed")
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>());

            let junction_arr = batch
                .column_by_name("junction")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let osm_id_arr = batch
                .column_by_name("osm_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let source_arr = batch
                .column_by_name("source")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            let geometry_arr = batch
                .column_by_name("geometry")
                .and_then(|c| c.as_any().downcast_ref::<BinaryArray>());

            for row_idx in 0..batch.num_rows() {
                let id = match id_arr {
                    Some(arr) => {
                        if arr.is_null(row_idx) {
                            continue;
                        }
                        arr.value(row_idx).to_string()
                    }
                    None => continue,
                };

                let name = None;
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

                let Some(geom_arr) = geometry_arr else {
                    continue;
                };
                if geom_arr.is_null(row_idx) {
                    continue;
                }
                let wkb_bytes = geom_arr.value(row_idx);

                let wkb_geom = match wkb::reader::read_wkb(wkb_bytes) {
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

    /// Convert a WKB geometry to our Geometry type
    fn convert_wkb_geometry<G: GeometryTrait<T = f64>>(geom: &G) -> Option<Geometry> {
        match geom.as_type() {
            GeometryType::LineString(ls) => {
                let coords: Vec<(f64, f64)> = ls.coords().map(|c| (c.x(), c.y())).collect();
                Some(Geometry::LineString(coords))
            }
            GeometryType::MultiLineString(mls) => {
                let coords: Vec<(f64, f64)> = mls
                    .line_strings()
                    .flat_map(|ls| ls.coords().map(|c| (c.x(), c.y())).collect::<Vec<_>>())
                    .collect();
                Some(Geometry::LineString(coords))
            }
            _ => None,
        }
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
