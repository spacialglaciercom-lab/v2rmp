use anyhow::Result;
use geo::algorithm::bool_ops::BooleanOps;
use geo::algorithm::orient::{Direction, Orient};
use geo::algorithm::remove_repeated_points::RemoveRepeatedPoints;
use geo::algorithm::simplify::Simplify;
use geo::{LineString, Polygon};
use geojson::{Feature, Geometry, Value};
use serde_json::Map;

use super::super::clean::{CleanOptions, CleanStats};

/// Repair and validate GeoJSON features.
/// Returns repaired LineString features and separately collected Point features.
pub fn repair_features(
    features: &[Feature],
    options: &CleanOptions,
    stats: &mut CleanStats,
) -> Result<(Vec<Feature>, Vec<Feature>)> {
    let mut repaired = Vec::new();
    let mut point_features: Vec<Feature> = Vec::new();

    for feature in features {
        let geom = match &feature.geometry {
            Some(g) => g,
            None => {
                stats.invalid_dropped += 1;
                continue;
            }
        };

        let geom_type = match &geom.value {
            Value::LineString(_) => "LineString",
            Value::MultiLineString(_) => "MultiLineString",
            Value::Polygon(_) if options.include_polygons => "Polygon",
            Value::MultiPolygon(_) if options.include_polygons => "MultiPolygon",
            Value::Point(_) if options.include_points => "Point",
            _ => {
                stats.invalid_dropped += 1;
                continue;
            }
        };

        match geom_type {
            "LineString" => {
                if let Some(line) = extract_linestring(&geom.value) {
                    if let Some(valid_line) = make_valid_and_validate_linestring(line, options) {
                        repaired.push(create_feature(valid_line, feature.properties.clone()));
                    } else {
                        stats.invalid_dropped += 1;
                    }
                } else {
                    stats.invalid_dropped += 1;
                }
            }
            "MultiLineString" => {
                if let Value::MultiLineString(ref coords) = geom.value {
                    for line_coords in coords {
                        if let Some(line) = coords_to_linestring(line_coords) {
                            if let Some(valid_line) =
                                make_valid_and_validate_linestring(line, options)
                            {
                                repaired
                                    .push(create_feature(valid_line, feature.properties.clone()));
                            }
                        }
                    }
                }
            }
            "Polygon" => {
                if let Some(poly) = extract_polygon(&geom.value) {
                    let lines = make_valid_polygon_to_linestrings(poly, options);
                    for line in lines {
                        if let Some(valid_line) = make_valid_and_validate_linestring(line, options)
                        {
                            repaired.push(create_feature(valid_line, feature.properties.clone()));
                        }
                    }
                }
            }
            "MultiPolygon" => {
                if let Value::MultiPolygon(ref coords) = geom.value {
                    for poly_coords in coords {
                        if let Some(poly) = polygon_coords_to_polygon(poly_coords) {
                            let lines = make_valid_polygon_to_linestrings(poly, options);
                            for line in lines {
                                if let Some(valid_line) =
                                    make_valid_and_validate_linestring(line, options)
                                {
                                    repaired.push(create_feature(
                                        valid_line,
                                        feature.properties.clone(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            "Point" => {
                // Points are passed through unchanged, collected separately
                point_features.push(feature.clone());
            }
            _ => {}
        }
    }

    Ok((repaired, point_features))
}

fn extract_linestring(value: &Value) -> Option<LineString<f64>> {
    if let Value::LineString(ref coords) = value {
        coords_to_linestring(coords)
    } else {
        None
    }
}

fn extract_polygon(value: &Value) -> Option<Polygon<f64>> {
    if let Value::Polygon(ref coords) = value {
        polygon_coords_to_polygon(coords)
    } else {
        None
    }
}

fn coords_to_linestring(coords: &[Vec<f64>]) -> Option<LineString<f64>> {
    if coords.len() < 2 {
        return None;
    }

    let points: Vec<_> = coords
        .iter()
        .filter(|c| c.len() >= 2)
        .map(|c| (c[0], c[1]))
        .collect();

    if points.len() < 2 {
        return None;
    }

    Some(LineString::from(points))
}

fn polygon_coords_to_polygon(coords: &[Vec<Vec<f64>>]) -> Option<Polygon<f64>> {
    if coords.is_empty() {
        return None;
    }

    let exterior = coords_to_linestring(&coords[0])?;
    let interiors: Vec<_> = coords[1..]
        .iter()
        .filter_map(|ring| coords_to_linestring(ring))
        .collect();

    Some(Polygon::new(exterior, interiors))
}

/// Make-valid for LineStrings: remove repeated points, ensure >= 2 coords.
fn make_valid_linestring(mut line: LineString<f64>) -> Option<LineString<f64>> {
    line = line.remove_repeated_points();
    if line.0.len() < 2 {
        return None;
    }
    Some(line)
}

/// Make-valid for Polygons: fix ring orientation and self-intersections.
/// Uses `BooleanOps::union` with an empty polygon to repair self-intersections
/// (the geo crate docs note that union with an empty geom removes degeneracies).
/// Then orients rings to standard winding order and extracts LineStrings.
fn make_valid_polygon_to_linestrings(
    poly: Polygon<f64>,
    options: &CleanOptions,
) -> Vec<LineString<f64>> {
    if !options.make_valid {
        return polygon_to_linestrings(&poly);
    }

    // Remove repeated points first
    let poly = poly.remove_repeated_points();

    // Check if polygon has enough points for a valid exterior ring
    if poly.exterior().0.len() < 4 {
        // A valid polygon ring needs at least 4 points (3 unique + closing).
        // Try to treat it as a degenerate line.
        let coords: Vec<_> = poly.exterior().0.iter().map(|c| (c.x, c.y)).collect();
        if coords.len() >= 2 {
            let mut deduped = coords;
            deduped.dedup();
            if deduped.len() >= 2 {
                return vec![LineString::from(deduped)];
            }
        }
        return vec![];
    }

    // Fix self-intersections via union with empty polygon.
    // This handles bowtie/self-intersecting polygons and may split
    // them into a MultiPolygon.
    let empty = Polygon::new(LineString::new(vec![]), vec![]);
    let repaired = poly.union(&empty);

    // Convert each polygon in the result to linestrings
    let mut all_lines = Vec::new();
    for repaired_poly in &repaired.0 {
        // Orient to standard winding (CCW exterior, CW interior)
        let oriented = repaired_poly.orient(Direction::Default);
        all_lines.extend(polygon_to_linestrings(&oriented));
    }

    all_lines
}

/// Validate and optionally simplify a LineString.
fn validate_linestring(
    mut line: LineString<f64>,
    options: &CleanOptions,
) -> Option<LineString<f64>> {
    // Remove duplicate consecutive points
    let mut coords: Vec<_> = line.0.to_vec();
    coords.dedup();

    if coords.len() < 2 {
        return None;
    }

    line = LineString::from(coords);

    // Simplify if requested
    if options.simplify_tolerance_m > 0.0 {
        let tolerance_deg = options.simplify_tolerance_m / 111_320.0;
        line = line.simplify(&tolerance_deg);
    }

    if line.0.len() < 2 {
        return None;
    }

    Some(line)
}

/// Combined make-valid + validate for a LineString.
fn make_valid_and_validate_linestring(
    line: LineString<f64>,
    options: &CleanOptions,
) -> Option<LineString<f64>> {
    let line = if options.make_valid {
        make_valid_linestring(line)?
    } else {
        line
    };
    validate_linestring(line, options)
}

fn polygon_to_linestrings(poly: &Polygon<f64>) -> Vec<LineString<f64>> {
    let mut lines = Vec::new();

    // Exterior ring
    let exterior = poly.exterior();
    if !exterior.0.is_empty() {
        let mut coords = exterior.0.to_vec();
        // Remove closing point if present
        if coords.len() >= 2 && coords[0] == coords[coords.len() - 1] {
            coords.pop();
        }
        if coords.len() >= 2 {
            lines.push(LineString::from(coords));
        }
    }

    // Interior rings
    for interior in poly.interiors() {
        if !interior.0.is_empty() {
            let mut coords = interior.0.to_vec();
            if coords.len() >= 2 && coords[0] == coords[coords.len() - 1] {
                coords.pop();
            }
            if coords.len() >= 2 {
                lines.push(LineString::from(coords));
            }
        }
    }

    lines
}

fn create_feature(
    line: LineString<f64>,
    properties: Option<Map<String, serde_json::Value>>,
) -> Feature {
    let coords: Vec<Vec<f64>> = line.0.iter().map(|coord| vec![coord.x, coord.y]).collect();

    Feature {
        bbox: None,
        geometry: Some(Geometry {
            bbox: None,
            value: Value::LineString(coords),
            foreign_members: None,
        }),
        id: None,
        properties,
        foreign_members: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::clean::{CleanOptions, CleanStats};
    use geojson::{Feature, Geometry, Value};

    fn default_options() -> CleanOptions {
        CleanOptions::default()
    }

    fn make_linestring_feature(coords: Vec<Vec<f64>>) -> Feature {
        Feature {
            bbox: None,
            geometry: Some(Geometry {
                bbox: None,
                value: Value::LineString(coords),
                foreign_members: None,
            }),
            id: None,
            properties: None,
            foreign_members: None,
        }
    }

    fn make_point_feature(lon: f64, lat: f64) -> Feature {
        Feature {
            bbox: None,
            geometry: Some(Geometry {
                bbox: None,
                value: Value::Point(vec![lon, lat]),
                foreign_members: None,
            }),
            id: None,
            properties: None,
            foreign_members: None,
        }
    }

    #[test]
    fn test_repair_valid_linestring() {
        let feature = make_linestring_feature(vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]]);
        let mut stats = CleanStats::default();
        let (repaired, _) = repair_features(&[feature], &default_options(), &mut stats).unwrap();
        assert_eq!(repaired.len(), 1);
    }

    #[test]
    fn test_repair_short_linestring_dropped() {
        let feature = make_linestring_feature(vec![vec![0.0, 0.0]]);
        let mut stats = CleanStats::default();
        let (repaired, _) = repair_features(&[feature], &default_options(), &mut stats).unwrap();
        assert_eq!(repaired.len(), 0);
        assert_eq!(stats.invalid_dropped, 1);
    }

    #[test]
    fn test_repair_duplicate_points_removed() {
        let feature = make_linestring_feature(vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0],
        ]);
        let mut stats = CleanStats::default();
        let (repaired, _) = repair_features(&[feature], &default_options(), &mut stats).unwrap();
        assert_eq!(repaired.len(), 1);
    }

    #[test]
    fn test_repair_point_passthrough() {
        let mut opts = default_options();
        opts.include_points = true;
        let feature = make_point_feature(1.0, 2.0);
        let mut stats = CleanStats::default();
        let (repaired, points) = repair_features(&[feature], &opts, &mut stats).unwrap();
        assert_eq!(repaired.len(), 0);
        assert_eq!(points.len(), 1);
    }

    #[test]
    fn test_repair_no_geometry_dropped() {
        let feature = Feature {
            bbox: None,
            geometry: None,
            id: None,
            properties: None,
            foreign_members: None,
        };
        let mut stats = CleanStats::default();
        let (repaired, _) = repair_features(&[feature], &default_options(), &mut stats).unwrap();
        assert_eq!(repaired.len(), 0);
        assert_eq!(stats.invalid_dropped, 1);
    }

    #[test]
    fn test_repair_multilinestring() {
        let feature = Feature {
            bbox: None,
            geometry: Some(Geometry {
                bbox: None,
                value: Value::MultiLineString(vec![
                    vec![vec![0.0, 0.0], vec![1.0, 0.0]],
                    vec![vec![2.0, 0.0], vec![3.0, 0.0]],
                ]),
                foreign_members: None,
            }),
            id: None,
            properties: None,
            foreign_members: None,
        };
        let mut stats = CleanStats::default();
        let (repaired, _) = repair_features(&[feature], &default_options(), &mut stats).unwrap();
        assert_eq!(repaired.len(), 2);
    }

    #[test]
    fn test_make_valid_disabled() {
        let mut opts = default_options();
        opts.make_valid = false;
        // Even with make_valid=false, duplicate consecutive points
        // should still be removed by validate_linestring's dedup step
        let feature = make_linestring_feature(vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]]);
        let mut stats = CleanStats::default();
        let (repaired, _) = repair_features(&[feature], &opts, &mut stats).unwrap();
        assert_eq!(repaired.len(), 1);
    }
}
