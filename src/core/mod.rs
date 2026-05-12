pub mod clean;
pub mod compile;
#[cfg(feature = "extract")]
pub mod elevation;
#[cfg(feature = "extract")]
pub mod extract;
pub mod geo_types;
pub mod optimize;
#[cfg(feature = "ml")]
pub mod ml;
pub mod ml_legacy;
pub mod nlp;
#[cfg(feature = "extract")]
pub mod osm;
#[cfg(feature = "extract")]
pub mod overture;
pub mod vrp;
#[cfg(feature = "ml")]
pub mod embed;

/// Haversine distance in meters between two WGS-84 points (lat, lon order).
pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    R * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
}
