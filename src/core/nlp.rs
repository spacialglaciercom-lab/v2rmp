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

#[cfg(feature = "ml")]
use candle_core::{Device, Tensor, DType};
#[cfg(feature = "ml")]
use candle_nn::VarBuilder;
#[cfg(feature = "ml")]
use candle_transformers::models::qwen2::{ModelForCausalLM, Config};
#[cfg(feature = "ml")]
use candle_transformers::generation::LogitsProcessor;
#[cfg(feature = "ml")]
use hf_hub::{api::sync::Api, Repo, RepoType};
#[cfg(feature = "ml")]
use tokenizers::Tokenizer;
#[cfg(feature = "ml")]
use anyhow::{Context, Result};

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
    /// City name for coordinate generation (e.g., "montreal", "toronto").
    pub city: Option<String>,
    /// Number of random coordinates to generate (if no explicit stops).
    pub generate_stops: Option<u32>,
    /// Whether this is a drone VRP task.
    pub is_drone: bool,
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
    // Check for drone VRP first (highest priority)
    if lower.contains("drone") || lower.contains("uav") || lower.contains("drone vrp") {
        result.variant = "drone_vrp".to_string();
        result.is_drone = true;
        // Drone-specific defaults
        result.avg_speed_kmh = Some(36.0); // 10 m/s = 36 km/h typical drone speed
    } else if lower.contains("sweep") || lower.contains("street") || lower.contains("road") {
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
    let num_regex = regex::Regex::new(r"(\d+)\s*(package|stop|customer|van|vehicle|driver|truck|route|random|coordinates?|points?|locations?)").unwrap();
    for cap in num_regex.captures_iter(&lower) {
        let num: u32 = cap[1].parse().unwrap_or(0);
        let noun = &cap[2];
        entities.insert(noun.to_string(), num.to_string());
        match noun {
            "package" | "stop" | "customer" | "random" | "coordinates" | "coordinate" | "points" | "point" | "locations" | "location" => {
                result.num_stops = Some(num);
                result.generate_stops = Some(num);
            },
            "van" | "vehicle" | "driver" | "truck" | "route" => result.num_vehicles = Some(num),
            _ => {}
        }
    }

    // ── City extraction for coordinate generation ────────────────────
    // Check for known city names
    let cities = ["montreal", "toronto", "vancouver", "calgary", "ottawa", "edmonton", "winnipeg", "quebec", "halifax"];
    for city in &cities {
        if lower.contains(city) {
            result.city = Some(city.to_string());
            entities.insert("city".to_string(), city.to_string());
            break;
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
    } else if let Some(city) = &result.city {
        // Set default depot and bbox for known cities
        match city.as_str() {
            "montreal" => {
                result.depot = Some((45.5017, -73.5673));
                result.bbox = Some(BoundingBox {
                    min_lat: 45.4,
                    max_lat: 45.8,
                    min_lon: -73.9,
                    max_lon: -73.5,
                });
            },
            "toronto" => {
                result.depot = Some((43.6511, -79.3470));
                result.bbox = Some(BoundingBox {
                    min_lat: 43.58,
                    max_lat: 43.85,
                    min_lon: -79.6,
                    max_lon: -79.1,
                });
            },
            "vancouver" => {
                result.depot = Some((49.2827, -123.1207));
                result.bbox = Some(BoundingBox {
                    min_lat: 49.15,
                    max_lat: 49.4,
                    min_lon: -123.3,
                    max_lon: -122.9,
                });
            },
            "calgary" => {
                result.depot = Some((51.0447, -114.0719));
                result.bbox = Some(BoundingBox {
                    min_lat: 50.85,
                    max_lat: 51.25,
                    min_lon: -114.3,
                    max_lon: -113.8,
                });
            },
            "ottawa" => {
                result.depot = Some((45.4215, -75.6972));
                result.bbox = Some(BoundingBox {
                    min_lat: 45.25,
                    max_lat: 45.6,
                    min_lon: -75.9,
                    max_lon: -75.4,
                });
            },
            "edmonton" => {
                result.depot = Some((53.5444, -113.4909));
                result.bbox = Some(BoundingBox {
                    min_lat: 53.4,
                    max_lat: 53.7,
                    min_lon: -113.7,
                    max_lon: -113.2,
                });
            },
            "winnipeg" => {
                result.depot = Some((49.8951, -97.1384));
                result.bbox = Some(BoundingBox {
                    min_lat: 49.75,
                    max_lat: 50.05,
                    min_lon: -97.4,
                    max_lon: -96.8,
                });
            },
            "quebec" => {
                result.depot = Some((46.8139, -71.2080));
                result.bbox = Some(BoundingBox {
                    min_lat: 46.7,
                    max_lat: 46.95,
                    min_lon: -71.45,
                    max_lon: -70.95,
                });
            },
            "halifax" => {
                result.depot = Some((44.6488, -63.5752));
                result.bbox = Some(BoundingBox {
                    min_lat: 44.55,
                    max_lat: 44.75,
                    min_lon: -63.7,
                    max_lon: -63.4,
                });
            },
            _ => {},
        }
    }

    // Default to 10 random stops if no explicit stops but generate_stops is set
    if result.generate_stops.is_none() && result.num_stops.is_none() {
        result.generate_stops = Some(10);
        result.num_stops = Some(10);
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
    if let Some(ref city) = parsed.city {
        obj["city"] = serde_json::json!(city);
    }
    if let Some(g) = parsed.generate_stops {
        obj["generate_stops"] = serde_json::json!(g);
    }
    if parsed.is_drone {
        obj["is_drone"] = serde_json::json!(true);
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

#[cfg(feature = "ml")]
pub struct QwenNLParser {
    model: ModelForCausalLM,
    tokenizer: Tokenizer,
    device: Device,
}

#[cfg(feature = "ml")]
impl QwenNLParser {
    /// Loads the Qwen2.5-0.5B-Instruct model from Hugging Face hub (cached locally).
    /// Uses 0.5B by default as 1.5B is too heavy for CPU-only MCP tools.
    pub fn new() -> Result<Self> {
        let device = crate::core::ml::best_device()?;
        let api = Api::new().context("Failed to create HF API client")?;
        let repo = api.repo(Repo::with_revision(
            "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        tracing::info!("Using device: {:?}", device);
        tracing::info!("Fetching Qwen2.5-0.5B tokenizer and config...");
        let tokenizer_path = repo.get("tokenizer.json")?;
        let config_path = repo.get("config.json")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            
        let config: Config = serde_json::from_reader(std::fs::File::open(config_path)?)?;

        // Download safetensors.
        tracing::info!("Fetching Qwen2.5-0.5B safetensors...");
        let model_path = repo.get("model.safetensors")?;

        // Use F16 on GPU, F32 on CPU for best compatibility/performance
        let dtype = if device.is_cuda() || device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)? };
        
        tracing::info!("Loading Qwen2.5 model into Candle ({:?})...", dtype);
        let model = ModelForCausalLM::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Translates a natural language query into a VRP JSON string using the LLM.
    pub fn parse_llm(&mut self, query: &str) -> Result<String> {
        let system_prompt = "You are an expert route optimization assistant. Convert the user's natural language routing query into a valid JSON object describing the Vehicle Routing Problem (VRP) configuration. Extract: 'num_stops', 'num_vehicles', 'depot' (as {\"lat\": .., \"lon\": ..}), 'deadline' (HH:MM), 'capacity', 'variant' (e.g., 'cvrp', 'cvrptw'). ONLY output valid JSON and nothing else.";
        
        let prompt = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", system_prompt, query);

        let tokens = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let mut tokens = tokens.get_ids().to_vec();

        let mut logits_processor = LogitsProcessor::new(1337, None, None);
        let mut output_text = String::new();

        let mut pos = 0;
        let max_tokens = 256;
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        for index in 0..max_tokens {
            if start_time.elapsed() > timeout {
                tracing::warn!("LLM inference timed out after 30s");
                break;
            }

            let context_size = if index == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;
            
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            pos += context_size;

            if let Some(text) = self.tokenizer.decode(&[next_token], true).ok() {
                output_text.push_str(&text);
                if output_text.contains("<|im_end|>") {
                    break;
                }
            }
        }

        let clean_json = output_text.replace("<|im_end|>", "").trim().to_string();
        Ok(clean_json)
    }
}

/// City boundaries for random coordinate generation.
pub struct CityBounds {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
    pub depot_lat: f64,
    pub depot_lon: f64,
}

/// Get boundaries for known cities.
pub fn get_city_bounds(city: &str) -> Option<CityBounds> {
    match city.to_lowercase().as_str() {
        "montreal" => Some(CityBounds {
            min_lat: 45.4,
            max_lat: 45.8,
            min_lon: -73.9,
            max_lon: -73.5,
            depot_lat: 45.5017,
            depot_lon: -73.5673,
        }),
        "toronto" => Some(CityBounds {
            min_lat: 43.58,
            max_lat: 43.85,
            min_lon: -79.6,
            max_lon: -79.1,
            depot_lat: 43.6511,
            depot_lon: -79.3470,
        }),
        "vancouver" => Some(CityBounds {
            min_lat: 49.15,
            max_lat: 49.4,
            min_lon: -123.3,
            max_lon: -122.9,
            depot_lat: 49.2827,
            depot_lon: -123.1207,
        }),
        "calgary" => Some(CityBounds {
            min_lat: 50.85,
            max_lat: 51.25,
            min_lon: -114.3,
            max_lon: -113.8,
            depot_lat: 51.0447,
            depot_lon: -114.0719,
        }),
        "ottawa" => Some(CityBounds {
            min_lat: 45.25,
            max_lat: 45.6,
            min_lon: -75.9,
            max_lon: -75.4,
            depot_lat: 45.4215,
            depot_lon: -75.6972,
        }),
        "edmonton" => Some(CityBounds {
            min_lat: 53.4,
            max_lat: 53.7,
            min_lon: -113.7,
            max_lon: -113.2,
            depot_lat: 53.5444,
            depot_lon: -113.4909,
        }),
        "winnipeg" => Some(CityBounds {
            min_lat: 49.75,
            max_lat: 50.05,
            min_lon: -97.4,
            max_lon: -96.8,
            depot_lat: 49.8951,
            depot_lon: -97.1384,
        }),
        "quebec" | "quebec city" => Some(CityBounds {
            min_lat: 46.7,
            max_lat: 46.95,
            min_lon: -71.45,
            max_lon: -70.95,
            depot_lat: 46.8139,
            depot_lon: -71.2080,
        }),
        "halifax" => Some(CityBounds {
            min_lat: 44.55,
            max_lat: 44.75,
            min_lon: -63.7,
            max_lon: -63.4,
            depot_lat: 44.6488,
            depot_lon: -63.5752,
        }),
        _ => None,
    }
}

/// Generate random coordinates within a city's boundaries.
/// Returns (depot, stops) where depot is the city center and stops are random points.
pub fn generate_city_coordinates(city: &str, num_stops: u32) -> Vec<(f64, f64)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    if let Some(bounds) = get_city_bounds(city) {
        let mut coordinates = Vec::new();
        
        // Add depot as first coordinate
        coordinates.push((bounds.depot_lat, bounds.depot_lon));
        
        // Generate random stops
        for _ in 0..num_stops {
            let lat = rng.gen_range(bounds.min_lat..bounds.max_lat);
            let lon = rng.gen_range(bounds.min_lon..bounds.max_lon);
            coordinates.push((lat, lon));
        }
        
        coordinates
    } else {
        // Fallback: generate around Montreal if city unknown
        let mut coordinates = vec![(45.5017, -73.5673)];
        let mut rng = rand::thread_rng();
        for _ in 0..num_stops {
            let lat = rng.gen_range(45.4..45.8);
            let lon = rng.gen_range(-73.9..-73.5);
            coordinates.push((lat, lon));
        }
        coordinates
    }
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
    fn test_parse_drone_montreal() {
        let q = "run vrp on random coordinates in montreal using drone vrp";
        let p = parse_query(q);
        assert_eq!(p.variant, "drone_vrp");
        assert!(p.is_drone);
        assert_eq!(p.city, Some("montreal".to_string()));
        assert_eq!(p.depot, Some((45.5017, -73.5673)));
        assert!(p.generate_stops.is_some());
    }

    #[test]
    fn test_to_vrp_json() {
        let p = parse_query("Route 10 packages with 2 vans at 40.7,-74.0");
        let json = to_vrp_json(&p);
        assert_eq!(json["num_stops"], 10);
        assert_eq!(json["num_vehicles"], 2);
    }

    #[test]
    fn test_city_bounds() {
        let bounds = get_city_bounds("montreal").unwrap();
        assert!((45.4..45.8).contains(&bounds.depot_lat));
        assert!((-73.9..-73.5).contains(&bounds.depot_lon));
    }

    #[test]
    fn test_generate_coordinates() {
        let coords = generate_city_coordinates("montreal", 5);
        assert_eq!(coords.len(), 6); // 1 depot + 5 stops
    }
}
