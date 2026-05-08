use serde::{Deserialize, Serialize};

/// Canonical bounding box for spatial queries.
///
/// This is the single source of truth — used by osm, overture, elevation,
/// extract, and everywhere else that needs lat/lon bounding boxes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BBox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

impl BBox {
    /// Check if a point is within the bounding box.
    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        lon >= self.min_lon && lon <= self.max_lon && lat >= self.min_lat && lat <= self.max_lat
    }

    /// Check if another bbox intersects this one.
    pub fn intersects(&self, other: &BBox) -> bool {
        !(self.max_lon < other.min_lon
            || self.min_lon > other.max_lon
            || self.max_lat < other.min_lat
            || self.min_lat > other.max_lat)
    }

    /// Approximate area in degrees squared.
    #[allow(dead_code)]
    pub fn area(&self) -> f64 {
        (self.max_lon - self.min_lon) * (self.max_lat - self.min_lat)
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
}
