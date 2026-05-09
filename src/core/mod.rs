pub mod clean;
pub mod compile;
#[cfg(feature = "extract")]
pub mod elevation;
#[cfg(feature = "ml")]
pub mod embed;
#[cfg(feature = "extract")]
pub mod extract;
pub mod geo_types;
pub mod optimize;
#[cfg(feature = "extract")]
pub mod osm;
#[cfg(feature = "extract")]
pub mod overture;
pub mod vrp;

/// Haversine distance in meters between two WGS-84 points (lat, lon order).
pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    haversine_m_rad(
        lat1.to_radians(),
        lon1.to_radians(),
        lat2.to_radians(),
        lon2.to_radians(),
    )
}

/// Haversine distance in meters between two points already in radians.
pub fn haversine_m_rad(lat1_r: f64, lon1_r: f64, lat2_r: f64, lon2_r: f64) -> f64 {
    const R: f64 = 6_371_000.0;
    let dlat = lat2_r - lat1_r;
    let dlon = lon2_r - lon1_r;
    let a = (dlat / 2.0).sin().powi(2) + lat1_r.cos() * lat2_r.cos() * (dlon / 2.0).sin().powi(2);
    R * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
}
