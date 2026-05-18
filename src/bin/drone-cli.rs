use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use v2rmp::core::drone::solver::DroneSolver;
use v2rmp::core::drone::{DroneModel, DroneSpec, DroneVrpInstance};

#[derive(Parser)]
#[command(name = "rmpca-drone")]
#[command(about = "Drone VRP Solver and Energy Calculator", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve a multi-trip Drone VRP
    Solve {
        /// Drone model (FlyCart30 or Wing)
        #[arg(long, default_value = "FlyCart30")]
        drone: String,

        /// Path to GeoJSON file with customers (point features with 'demand' property)
        #[arg(long)]
        customers: String,

        /// Depot latitude
        #[arg(long)]
        depot_lat: f64,

        /// Depot longitude
        #[arg(long)]
        depot_lon: f64,

        /// Wind speed in m/s
        #[arg(long, default_value_t = 0.0)]
        wind: f64,

        /// Output path for the solution GeoJSON
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Calculate energy for a single flight leg
    Energy {
        /// Drone model
        #[arg(long, default_value = "FlyCart30")]
        drone: String,

        /// Distance in meters
        #[arg(short, long)]
        distance: f64,

        /// Payload in kg
        #[arg(short, long, default_value_t = 0.0)]
        payload: f64,

        /// Wind speed in m/s
        #[arg(short, long, default_value_t = 0.0)]
        wind: f64,
    },

    /// Display drone specifications
    Specs {
        /// Drone model
        #[arg(long)]
        drone: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Solve {
            drone,
            customers,
            depot_lat,
            depot_lon,
            wind,
            output,
        } => {
            let model = match drone.to_lowercase().as_str() {
                "flycart30" | "fc30" => DroneModel::FlyCart30,
                "wing" => DroneModel::Wing,
                _ => anyhow::bail!("Unknown drone model: {}", drone),
            };

            // Load customers from GeoJSON
            let geojson_str =
                fs::read_to_string(&customers).context("Failed to read customers file")?;
            let geojson = geojson_str.parse::<geojson::GeoJson>()?;

            let mut customer_coords = Vec::new();
            let mut demands = Vec::new();

            if let geojson::GeoJson::FeatureCollection(collection) = geojson {
                for feature in collection.features {
                    if let Some(ref geometry) = feature.geometry {
                        if let geojson::Value::Point(ref coords) = geometry.value {
                            customer_coords.push([coords[1], coords[0]]); // lat, lon
                            let demand = feature
                                .property("demand")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0);
                            demands.push(demand);
                        }
                    }
                }
            }

            let instance = DroneVrpInstance {
                drone_model: model.clone(),
                depot: [depot_lat, depot_lon],
                customers: customer_coords,
                demands_kg: demands,
                wind_speed_ms: wind,
                no_fly_zones: vec![],
            };

            println!(
                "Solving VRP for {} customers using {:?}...",
                instance.customers.len(),
                model
            );
            let result = DroneSolver::solve(&instance)?;

            println!("Solution found:");
            println!("  Routes: {}", result.routes.len());
            println!("  Energy used: {:.2} Wh", result.energy_used_wh);
            if !result.violations.is_empty() {
                println!("  Violations: {:?}", result.violations);
            }

            if let Some(out_path) = output {
                let json = serde_json::to_string_pretty(&result)?;
                fs::write(out_path, json)?;
                println!("Result saved to {}", result.routes.len());
            }
        }
        Commands::Energy {
            drone,
            distance,
            payload,
            wind,
        } => {
            let spec = match drone.to_lowercase().as_str() {
                "flycart30" | "fc30" => DroneSpec::flycart30(),
                "wing" => DroneSpec::wing(),
                _ => anyhow::bail!("Unknown drone model: {}", drone),
            };
            let energy = spec.calculate_energy_wh(distance, payload, wind);
            println!("Energy consumption: {:.4} Wh", energy);
        }
        Commands::Specs { drone } => {
            let models = if let Some(d) = drone {
                vec![match d.to_lowercase().as_str() {
                    "flycart30" | "fc30" => DroneSpec::flycart30(),
                    "wing" => DroneSpec::wing(),
                    _ => anyhow::bail!("Unknown drone model: {}", d),
                }]
            } else {
                vec![DroneSpec::flycart30(), DroneSpec::wing()]
            };

            for s in models {
                println!("--- {:?} ---", s.model);
                println!("Mass: {} kg", s.mass_kg);
                println!("Max Payload: {} kg", s.max_payload_kg);
                println!("Battery: {} Wh", s.battery_capacity_wh);
                println!("Cruise Speed: {} m/s", s.cruise_speed_ms);
                println!("Power (empty): {} W", s.power_no_load_w);
                println!("Power (full): {} W", s.power_max_load_w);
                println!();
            }
        }
    }

    Ok(())
}
