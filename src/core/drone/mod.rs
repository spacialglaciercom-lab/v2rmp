pub mod solver;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DroneModel {
    FlyCart30,
    Wing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneSpec {
    pub model: DroneModel,
    pub mass_kg: f64,
    pub max_payload_kg: f64,
    pub battery_capacity_wh: f64,
    pub cruise_speed_ms: f64,
    pub power_no_load_w: f64,  // P_p in Dorling et al.
    pub power_max_load_w: f64, // P_l in Dorling et al.
}

impl DroneSpec {
    pub fn flycart30() -> Self {
        Self {
            model: DroneModel::FlyCart30,
            mass_kg: 29.9,
            max_payload_kg: 30.0,
            battery_capacity_wh: 2000.0, // Dual battery system
            cruise_speed_ms: 15.0,
            power_no_load_w: 500.0,
            power_max_load_w: 1200.0,
        }
    }

    pub fn wing() -> Self {
        Self {
            model: DroneModel::Wing,
            mass_kg: 4.8,
            max_payload_kg: 1.2,
            battery_capacity_wh: 150.0,
            cruise_speed_ms: 30.0,
            power_no_load_w: 80.0,
            power_max_load_w: 150.0,
        }
    }

    /// Physics-based energy model (Dorling et al. 2017)
    /// Power = (P_p + P_l * (payload / max_payload))
    /// Energy (Wh) = (Power * Distance / Speed) / 3600
    pub fn calculate_energy_wh(&self, distance_m: f64, payload_kg: f64, wind_ms: f64) -> f64 {
        let payload_ratio = (payload_kg / self.max_payload_kg).clamp(0.0, 1.0);
        let base_power =
            self.power_no_load_w + (self.power_max_load_w - self.power_no_load_w) * payload_ratio;

        // Simplified wind impact: power scales with (relative_speed / cruise_speed)^3
        // For simplicity here, we adjust the effective speed
        let ground_speed = (self.cruise_speed_ms - wind_ms).max(1.0);
        let time_h = (distance_m / ground_speed) / 3600.0;

        base_power * time_h
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoFlyZone {
    pub id: String,
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneVrpInstance {
    pub drone_model: DroneModel,
    pub battery_capacity_wh: Option<f64>,
    pub depot: [f64; 2],
    pub customers: Vec<[f64; 2]>,
    pub demands_kg: Vec<f64>,
    pub wind_speed_ms: f64,
    pub no_fly_zones: Vec<NoFlyZone>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneVrpResult {
    pub routes: Vec<Vec<usize>>,
    pub energy_used_wh: f64,
    pub algorithm: String,
    pub violations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drone_specs() {
        let fc = DroneSpec::flycart30();
        let wing = DroneSpec::wing();
        assert_eq!(fc.model, DroneModel::FlyCart30);
        assert_eq!(wing.model, DroneModel::Wing);
    }

    #[test]
    fn test_energy_model_empty() {
        let fc = DroneSpec::flycart30();
        let energy = fc.calculate_energy_wh(15000.0, 0.0, 0.0); // 15km at 15m/s = 1000s = ~0.27h
                                                                // 500W * 0.27h = 138.8 Wh
        assert!(energy > 130.0 && energy < 145.0);
    }

    #[test]
    fn test_energy_model_full_load() {
        let fc = DroneSpec::flycart30();
        let energy = fc.calculate_energy_wh(15000.0, 30.0, 0.0);
        // 1200W * 0.27h = 333.3 Wh
        assert!(energy > 330.0 && energy < 340.0);
    }

    #[test]
    fn test_wind_impact() {
        let fc = DroneSpec::flycart30();
        let energy_no_wind = fc.calculate_energy_wh(10000.0, 0.0, 0.0);
        let energy_headwind = fc.calculate_energy_wh(10000.0, 0.0, 5.0);
        assert!(energy_headwind > energy_no_wind);
    }

    #[test]
    fn test_wing_energy() {
        let wing = DroneSpec::wing();
        let energy = wing.calculate_energy_wh(30000.0, 0.0, 0.0); // 30km at 30m/s = 1000s
                                                                  // 80W * 0.27h = 22.2 Wh
        assert!(energy > 20.0 && energy < 25.0);
    }

    #[test]
    fn test_payload_limit() {
        let fc = DroneSpec::flycart30();
        let energy_full = fc.calculate_energy_wh(1000.0, 30.0, 0.0);
        let energy_over = fc.calculate_energy_wh(1000.0, 100.0, 0.0); // Should cap at max_load
        assert_eq!(energy_full, energy_over);
    }
}
