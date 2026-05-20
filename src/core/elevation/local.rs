use anyhow::{Context, Result};
use gdal::Dataset;
use std::path::Path;

use super::types::*;

/// Local DEM file reader using GDAL.
///
/// Opens a GeoTIFF (or any GDAL-supported raster) and provides
/// elevation sampling at arbitrary WGS84 coordinates.
pub struct LocalDem {
    dataset: Dataset,
    /// GeoTransform: [origin_x, pixel_width, 0, origin_y, 0, -pixel_height]
    geo_transform: [f64; 6],
    /// Raster size (width, height) in pixels.
    size: (usize, usize),
    /// Band index (0-based internally, 1-based in GDAL).
    band_index: usize,
    /// NoData value for the band.
    nodata: Option<f64>,
}

impl LocalDem {
    /// Open a DEM file (GeoTIFF, etc.) and prepare for elevation queries.
    pub fn open(path: &Path) -> Result<Self> {
        let dataset = Dataset::open(path)
            .with_context(|| format!("Failed to open DEM file: {}", path.display()))?;

        let geo_transform = dataset
            .geo_transform()
            .context("DEM file has no geo transform")?;
        let size = dataset.raster_size();

        let rasterband = dataset
            .rasterband(1)
            .context("DEM file has no raster band")?;
        let nodata = rasterband.no_data_value();

        Ok(Self {
            dataset,
            geo_transform,
            size,
            band_index: 1,
            nodata,
        })
    }

    /// Return the bounding box of the DEM in WGS84 (min_lon, min_lat, max_lon, max_lat).
    pub fn bbox(&self) -> BBox {
        let gt = &self.geo_transform;
        let (w, h) = self.size;

        let x_0 = gt[0];
        let y_0 = gt[3];
        let x_1 = gt[0] + w as f64 * gt[1];
        let y_1 = gt[3] + h as f64 * gt[5];

        BBox {
            min_lon: x_0.min(x_1),
            min_lat: y_0.min(y_1),
            max_lon: x_0.max(x_1),
            max_lat: y_0.max(y_1),
        }
    }

    // -----------------------------------------------------------------------
    // Core sampling
    // -----------------------------------------------------------------------

    /// Convert WGS84 (lon, lat) to pixel (col, row) fractional coordinates.
    fn latlon_to_pixel(&self, lon: f64, lat: f64) -> (f64, f64) {
        let gt = &self.geo_transform;
        // inverse affine: col = (x - origin_x) / pixel_width
        //                 row = (y - origin_y) / pixel_height
        let col = (lon - gt[0]) / gt[1];
        let row = (lat - gt[3]) / gt[5];
        (col, row)
    }

    /// Check if pixel coordinates are within raster bounds.
    fn in_bounds(&self, col: f64, row: f64) -> bool {
        col >= 0.0 && col < self.size.0 as f64 && row >= 0.0 && row < self.size.1 as f64
    }

    /// Read a single pixel value from the raster band.
    fn read_pixel(&self, col: isize, row: isize) -> Result<Option<f64>> {
        let band = self.dataset.rasterband(self.band_index)?;

        let mut buf = [0.0f64];
        band.read_into_slice((col, row), (1, 1), (1, 1), &mut buf, None)?;

        let val = buf[0];
        if let Some(nd) = self.nodata {
            if (val - nd).abs() < 1e-6 {
                return Ok(None);
            }
        }
        Ok(Some(val))
    }

    /// Get elevation at a single point using bilinear interpolation.
    pub fn get_elevation(&self, lon: f64, lat: f64) -> Result<Option<f64>> {
        let (col_f, row_f) = self.latlon_to_pixel(lon, lat);

        if !self.in_bounds(col_f, row_f) {
            return Ok(None);
        }

        let col0 = col_f.floor() as isize;
        let row0 = row_f.floor() as isize;
        let dx = col_f - col0 as f64;
        let dy = row_f - row0 as f64;

        let (w, h) = self.size;
        let col1 = ((col0 + 1) as usize).min(w - 1) as isize;
        let row1 = ((row0 + 1) as usize).min(h - 1) as isize;

        // Read 4 surrounding pixels
        let v00 = self.read_pixel(col0, row0)?;
        let v10 = self.read_pixel(col1, row0)?;
        let v01 = self.read_pixel(col0, row1)?;
        let v11 = self.read_pixel(col1, row1)?;

        // If any corner is nodata, fall back to nearest-neighbor
        match (v00, v10, v01, v11) {
            (Some(p00), Some(p10), Some(p01), Some(p11)) => {
                let val = p00 * (1.0 - dx) * (1.0 - dy)
                    + p10 * dx * (1.0 - dy)
                    + p01 * (1.0 - dx) * dy
                    + p11 * dx * dy;
                Ok(Some(val))
            }
            _ => {
                // Nearest-neighbor fallback
                let nc = if dx < 0.5 { col0 } else { col0 + 1 };
                let nr = if dy < 0.5 { row0 } else { row0 + 1 };
                self.read_pixel(nc.min(w as isize - 1), nr.min(h as isize - 1))
            }
        }
    }

    /// Get elevations at multiple points.
    pub fn get_elevations(&self, points: &[(f64, f64)]) -> Result<Vec<Option<f64>>> {
        points
            .iter()
            .map(|(lon, lat)| self.get_elevation(*lon, *lat))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Route profile
    // -----------------------------------------------------------------------

    /// Sample elevation along a route at a fixed interval (in meters).
    ///
    /// `route` is `[(lon, lat), ...]`.
    /// `sample_interval_m` is distance between samples.
    pub fn route_profile(
        &self,
        route: &[(f64, f64)],
        sample_interval_m: f64,
    ) -> Result<ElevationProfile> {
        if route.len() < 2 {
            return Ok(ElevationProfile::default());
        }

        // Build cumulative distance array
        let mut cum_dist = vec![0.0f64];
        for i in 1..route.len() {
            let (lon1, lat1) = route[i - 1];
            let (lon2, lat2) = route[i];
            let d = crate::core::haversine_m(lat1, lon1, lat2, lon2);
            cum_dist.push(cum_dist[i - 1] + d);
        }

        let total_length = *cum_dist.last().unwrap();
        if total_length < 1.0 {
            // Route is too short
            let elev = self.get_elevation(route[0].0, route[0].1)?;
            return Ok(ElevationProfile {
                points: vec![RouteElevationPoint {
                    distance_m: 0.0,
                    elevation_m: elev,
                    point: Point {
                        lon: route[0].0,
                        lat: route[0].1,
                    },
                }],
                total_ascent: 0.0,
                total_descent: 0.0,
                max_elevation: elev.unwrap_or(0.0),
                min_elevation: elev.unwrap_or(0.0),
                avg_elevation: elev.unwrap_or(0.0),
                distance_km: total_length / 1000.0,
            });
        }

        let num_samples = (total_length / sample_interval_m).ceil() as usize;
        let mut points = Vec::with_capacity(num_samples + 1);
        let mut total_ascent = 0.0;
        let mut total_descent = 0.0;

        for i in 0..=num_samples {
            let target_dist = i as f64 * sample_interval_m;
            let target_dist = target_dist.min(total_length);

            // Find which segment this falls on
            let seg_idx = cum_dist
                .iter()
                .position(|&d| d >= target_dist)
                .unwrap_or(cum_dist.len() - 1)
                .max(1);
            let seg_idx = seg_idx - 1;

            let seg_len = cum_dist[seg_idx + 1] - cum_dist[seg_idx];
            let t = if seg_len > 0.0 {
                (target_dist - cum_dist[seg_idx]) / seg_len
            } else {
                0.0
            };

            let (lon1, lat1) = route[seg_idx];
            let (lon2, lat2) = route[seg_idx + 1];
            let lon = lon1 + t * (lon2 - lon1);
            let lat = lat1 + t * (lat2 - lat1);

            let elevation = self.get_elevation(lon, lat)?;

            if i > 0 {
                if let (Some(prev_e), Some(curr_e)) = (
                    points
                        .last()
                        .and_then(|p: &RouteElevationPoint| p.elevation_m),
                    elevation,
                ) {
                    let diff = curr_e - prev_e;
                    if diff > 0.0 {
                        total_ascent += diff;
                    } else {
                        total_descent += diff.abs();
                    }
                }
            }

            points.push(RouteElevationPoint {
                distance_m: target_dist,
                elevation_m: elevation,
                point: Point { lon, lat },
            });
        }

        // Compute stats from valid elevation values
        let elevations: Vec<f64> = points.iter().filter_map(|p| p.elevation_m).collect();
        let max_elevation = elevations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_elevation = elevations.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg_elevation = if elevations.is_empty() {
            0.0
        } else {
            elevations.iter().sum::<f64>() / elevations.len() as f64
        };

        Ok(ElevationProfile {
            points,
            total_ascent,
            total_descent,
            max_elevation,
            min_elevation,
            avg_elevation,
            distance_km: total_length / 1000.0,
        })
    }

    // -----------------------------------------------------------------------
    // Area statistics
    // -----------------------------------------------------------------------

    /// Sample elevation statistics within a bounding box at a grid resolution.
    ///
    /// `grid_step` is the pixel skip (e.g., 1 = every pixel, 10 = every 10th pixel).
    pub fn bbox_stats(&self, bbox: BBox, grid_step: usize) -> Result<GeofenceStats> {
        let (col_start_f, row_start_f) = self.latlon_to_pixel(bbox.min_lon, bbox.max_lat);
        let (col_end_f, row_end_f) = self.latlon_to_pixel(bbox.max_lon, bbox.min_lat);

        let col_start = col_start_f.floor().max(0.0) as usize;
        let row_start = row_start_f.floor().max(0.0) as usize;
        let col_end = (col_end_f.ceil() as usize).min(self.size.0);
        let row_end = (row_end_f.ceil() as usize).min(self.size.1);

        let step = grid_step.max(1);
        let mut min_elev = f64::INFINITY;
        let mut max_elev = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut count = 0i64;
        let mut total_pixels = 0i64;

        for row in (row_start..row_end).step_by(step) {
            for col in (col_start..col_end).step_by(step) {
                total_pixels += 1;
                if let Some(val) = self.read_pixel(col as isize, row as isize)? {
                    min_elev = min_elev.min(val);
                    max_elev = max_elev.max(val);
                    sum += val;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok(GeofenceStats {
                min_elevation: 0.0,
                max_elevation: 0.0,
                avg_elevation: 0.0,
                coverage_percent: 0.0,
                pixel_count: 0,
            });
        }

        Ok(GeofenceStats {
            min_elevation: min_elev,
            max_elevation: max_elev,
            avg_elevation: sum / count as f64,
            coverage_percent: (count as f64 / total_pixels as f64) * 100.0,
            pixel_count: count,
        })
    }

    /// Get DEM info (size, bbox, nodata) as a JSON-serializable struct.
    pub fn info(&self) -> DemInfo {
        let bbox = self.bbox();
        DemInfo {
            width: self.size.0,
            height: self.size.1,
            bbox,
            nodata: self.nodata,
            pixel_size_x: self.geo_transform[1],
            pixel_size_y: self.geo_transform[5].abs(),
        }
    }
}

/// DEM file metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DemInfo {
    pub width: usize,
    pub height: usize,
    pub bbox: BBox,
    pub nodata: Option<f64>,
    pub pixel_size_x: f64,
    pub pixel_size_y: f64,
}
