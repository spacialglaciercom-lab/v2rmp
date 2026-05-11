//! Natural Language Query Parser for VRP routing.
//!
//! Converts free-text routing requests into structured VRP JSON configs.
//!
//! Example:
//!   "Route 50 packages with 5 vans starting at 45.5,-73.6 by 5pm"
//!   → {"num_stops":50, "num_vehicles":5, "depot":[45.5,-73.6], "deadline":"17:00"}
//!
//! Research basis: "From Words to Routes" (2403.10795, 2024)
//!
//! Design: Hybrid approach
//!   1. Regex-based entity extraction (numbers, coordinates, times)
//!   2. Intent classification via keyword matching
//!   3. Template-based JSON generation with slot filling
//!   4. (Future) Small fine-tuned LLM for complex/ambiguous queries

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parsed VRP configuration from natural language.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedRoutingQuery {
    /// Variant: CVRP, CVRPTW, mTSP, CPP, etc.
    pub variant: String,
    /// Number of stops / packages / customers.
    pub num_stops: Option<u32>,
    /// Number of vehicles / vans / drivers.
    pub num_vehicles: Option<u32>,
    /// Depot coordinates [lat, lon].
    pub depot: Option<(f64, f64)>,
    /// Time window deadline (HH:MM or ISO).
    pub deadline: Option<String>,
    /// Vehicle capacity.
    pub capacity: Option<f64>,
    /// Average speed km/h.
    pub avg_speed_kmh: Option<f64>,
    /// Optimization objective.
    pub objective: Option<String>,
    /// Source data bbox (if extracting network).
    pub bbox: Option<BoundingBox>,
    /// Raw extracted entities for debugging.
    pub entities: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

/// Parse a natural language routing query.
pub fn parse_query(query: &str) -> ParsedRoutingQuery {
    let lower = query.to_lowercase();
    let mut entities = HashMap::new();
    let mut result = ParsedRoutingQuery::default();

    // ── Intent classification ─────────────────────────────────────────
    if lower.contains("sweep") || lower.contains("street") || lower.contains("road") {
        result.variant = "cpp".to_string();
    } else if lower.contains("time window") || lower.contains("by ") || lower.contains("deadline") {
        result.variant = "cvrptw".to_string();
    } else if lower.contains("package") || lower.contains("delivery") || lower.contains("customer") {
        result.variant = "cvrp".to_string();
    } else if lower.contains("multiple depot") || lower.contains("multi-depot") {
        result.variant = "mdvrp".to_string();
    } else {
        result.variant = "cvrp".to_string();
    }

    // ── Entity extraction: numbers ──────────────────────────────────
    // "50 packages", "5 vans", "100 customers", etc.
    let num_regex = regex::Regex::new(r"(\d+)\s*(package|stop|customer|van|vehicle|driver|truck|route)").unwrap();
    for cap in num_regex.captures_iter(&lower) {
        let num: u32 = cap[1].parse().unwrap_or(0);
        let noun = &cap[2];
        entities.insert(noun.to_string(), num.to_string());
        match noun {
            "package" | "stop" | "customer" => result.num_stops = Some(num),
            "van" | "vehicle" | "driver" | "truck" | "route" => result.num_vehicles = Some(num),
            _ => {}
        }
    }

    // ── Entity extraction: coordinates ────────────────────────────────
    // "45.5, -73.6" or "lat 45.5 lon -73.6"
    let coord_regex = regex::Regex::new(
        r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)"
    ).unwrap();
    if let Some(cap) = coord_regex.captures(&lower) {
        let lat: f64 = cap[1].parse().unwrap_or(0.0);
        let lon: f64 = cap[2].parse().unwrap_or(0.0);
        if lat.abs() <= 90.0 && lon.abs() <= 180.0 {
            result.depot = Some((lat, lon));
            entities.insert("depot".to_string(), format!("{}, {}", lat, lon));
        }
    }

    // ── Entity extraction: time ─────────────────────────────────────
    // "by 5pm", "before 17:00", "deadline 5:30 PM"
    let time_regex = regex::Regex::new(
        r"(?:by|before|deadline|until)\s*(\d{1,2}):?(\d{2})?\s*(am|pm)?"
    ).unwrap();
    if let Some(cap) = time_regex.captures(&lower) {
        let hour: u32 = cap[1].parse().unwrap_or(0);
        let minute: u32 = cap.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
        let ampm = cap.get(3).map(|m| m.as_str()).unwrap_or("");
        let mut h = hour;
        if ampm == "pm" && h < 12 { h += 12; }
        if ampm == "am" && h == 12 { h = 0; }
        let time_str = format!("{:02}:{:02}", h, minute);
        result.deadline = Some(time_str.clone());
        entities.insert("deadline".to_string(), time_str);
    }

    // ── Entity extraction: capacity ─────────────────────────────────
    // "capacity 100", "up to 50 kg"
    let cap_regex = regex::Regex::new(r"capacity\s*(\d+)").unwrap();
    if let Some(cap) = cap_regex.captures(&lower) {
        let c: f64 = cap[1].parse().unwrap_or(0.0);
        result.capacity = Some(c);
        entities.insert("capacity".to_string(), c.to_string());
    }

    // ── Entity extraction: speed ────────────────────────────────────
    let speed_regex = regex::Regex::new(r"(\d+)\s*km/h").unwrap();
    if let Some(cap) = speed_regex.captures(&lower) {
        let s: f64 = cap[1].parse().unwrap_or(0.0);
        result.avg_speed_kmh = Some(s);
        entities.insert("speed".to_string(), s.to_string());
    }

    // ── Objective extraction ────────────────────────────────────────
    if lower.contains("shortest") || lower.contains("min distance") || lower.contains("fastest route") {
        result.objective = Some("min_distance".to_string());
    } else if lower.contains("min time") || lower.contains("quickest") {
        result.objective = Some("min_time".to_string());
    } else if lower.contains("balance") || lower.contains("even") || lower.contains("fair") {
        result.objective = Some("balance_load".to_string());
    } else if lower.contains("min vehicle") || lower.contains("fewest") {
        result.objective = Some("min_vehicles".to_string());
    }

    // ── Bounding box extraction (loose) ─────────────────────────────
    // "area around 45.5,-73.6" or "in Montreal"
    if let Some((lat, lon)) = result.depot {
        let delta = 0.05; // ~5km
        result.bbox = Some(BoundingBox {
            min_lat: lat - delta,
            max_lat: lat + delta,
            min_lon: lon - delta,
            max_lon: lon + delta,
        });
    }

    result.entities = entities;
    result
}

/// Convert a parsed query to a VRP solver JSON payload.
pub fn to_vrp_json(parsed: &ParsedRoutingQuery) -> serde_json::Value {
    let mut obj = serde_json::json!({
        "variant": parsed.variant,
    });

    if let Some(n) = parsed.num_stops {
        obj["num_stops"] = serde_json::json!(n);
    }
    if let Some(v) = parsed.num_vehicles {
        obj["num_vehicles"] = serde_json::json!(v);
    }
    if let Some((lat, lon)) = parsed.depot {
        obj["depot"] = serde_json::json!({"lat": lat, "lon": lon});
    }
    if let Some(ref d) = parsed.deadline {
        obj["deadline"] = serde_json::json!(d);
    }
    if let Some(c) = parsed.capacity {
        obj["capacity"] = serde_json::json!(c);
    }
    if let Some(s) = parsed.avg_speed_kmh {
        obj["avg_speed_kmh"] = serde_json::json!(s);
    }
    if let Some(ref o) = parsed.objective {
        obj["objective"] = serde_json::json!(o);
    }
    if let Some(ref b) = parsed.bbox {
        obj["bbox"] = serde_json::json!({
            "min_lon": b.min_lon,
            "min_lat": b.min_lat,
            "max_lon": b.max_lon,
            "max_lat": b.max_lat,
        });
    }

    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_packages_vans() {
        let q = "Route 50 packages with 5 vans starting at 45.5,-73.6 by 5pm";
        let p = parse_query(q);
        assert_eq!(p.variant, "cvrptw"); // time window detected
        assert_eq!(p.num_stops, Some(50));
        assert_eq!(p.num_vehicles, Some(5));
        assert_eq!(p.depot, Some((45.5, -73.6)));
        assert_eq!(p.deadline, Some("17:00".to_string()));
    }

    #[test]
    fn test_parse_sweep_streets() {
        let q = "Sweep all streets in area 45.5,-73.6 with 2 trucks";
        let p = parse_query(q);
        assert_eq!(p.variant, "cpp");
        assert_eq!(p.depot, Some((45.5, -73.6)));
        assert_eq!(p.num_vehicles, Some(2));
    }

    #[test]
    fn test_parse_time_window() {
        let q = "Deliver to 30 customers with 3 vans by 16:30, min time";
        let p = parse_query(q);
        assert_eq!(p.variant, "cvrptw");
        assert_eq!(p.num_stops, Some(30));
        assert_eq!(p.num_vehicles, Some(3));
        assert_eq!(p.deadline, Some("16:30".to_string()));
        assert_eq!(p.objective, Some("min_time".to_string()));
    }

    #[test]
    fn test_to_vrp_json() {
        let p = parse_query("Route 10 packages with 2 vans at 40.7,-74.0");
        let json = to_vrp_json(&p);
        assert_eq!(json["num_stops"], 10);
        assert_eq!(json["num_vehicles"], 2);
    }
}
