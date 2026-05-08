use serde::{Deserialize, Serialize};

/// WGS84 coordinate point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub lon: f64,
    pub lat: f64,
}

/// A single point in a route elevation profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteElevationPoint {
    pub distance_m: f64,
    pub elevation_m: Option<f64>,
    pub point: Point,
}

/// Complete elevation profile for a route.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElevationProfile {
    pub points: Vec<RouteElevationPoint>,
    pub total_ascent: f64,
    pub total_descent: f64,
    pub max_elevation: f64,
    pub min_elevation: f64,
    pub avg_elevation: f64,
    pub distance_km: f64,
}

impl Default for ElevationProfile {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            total_ascent: 0.0,
            total_descent: 0.0,
            max_elevation: 0.0,
            min_elevation: 0.0,
            avg_elevation: 0.0,
            distance_km: 0.0,
        }
    }
}

/// Elevation statistics within a geofence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeofenceStats {
    pub min_elevation: f64,
    pub max_elevation: f64,
    pub avg_elevation: f64,
    pub coverage_percent: f64,
    pub pixel_count: i64,
}

pub use crate::core::geo_types::BBox;
/// Fuel consumption analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuelConsumption {
    pub total_fuel_l: f64,
    pub avg_consumption_l_per_km: f64,
    pub elevation_penalty_l: f64,
    pub elevation_benefit_l: f64,
}
