use super::types::*;

/// Fuel consumption calculator accounting for elevation changes.
pub struct FuelCalculator;

impl FuelCalculator {
    /// Calculate fuel consumption factor based on an elevation profile.
    ///
    /// This is a simplified model:
    /// - Base fuel consumption on flat terrain (`base_consumption` L/km)
    /// - +15% per 1% uphill grade
    /// - -5% per 1% downhill grade (capped at -20%)
    pub fn calculate(profile: &ElevationProfile, base_consumption: f64) -> FuelConsumption {
        let mut total_fuel = 0.0;
        let mut elevation_penalty = 0.0;
        let mut elevation_benefit = 0.0;

        for i in 1..profile.points.len() {
            let prev = &profile.points[i - 1];
            let curr = &profile.points[i];

            let prev_elev = match prev.elevation_m {
                Some(e) => e,
                None => continue,
            };
            let curr_elev = match curr.elevation_m {
                Some(e) => e,
                None => continue,
            };

            let distance_km = (curr.distance_m - prev.distance_m) / 1000.0;
            let elevation_change = curr_elev - prev_elev;
            let grade = if distance_km > 0.0 {
                elevation_change / (distance_km * 1000.0)
            } else {
                0.0
            };

            let base_fuel = base_consumption * distance_km;

            let elevation_factor = if grade > 0.0 {
                // Uphill penalty: +15% per 1% grade
                let factor = 0.15 * grade * 100.0;
                elevation_penalty += base_fuel * factor;
                factor
            } else {
                // Downhill benefit: -5% per 1% grade, capped at -20%
                let factor = (0.05 * grade * 100.0).max(-0.20);
                elevation_benefit += (base_fuel * factor).abs();
                factor
            };

            total_fuel += base_fuel * (1.0 + elevation_factor);
        }

        let avg_consumption = if profile.distance_km > 0.0 {
            total_fuel / profile.distance_km
        } else {
            base_consumption
        };

        FuelConsumption {
            total_fuel_l: total_fuel,
            avg_consumption_l_per_km: avg_consumption,
            elevation_penalty_l: elevation_penalty,
            elevation_benefit_l: elevation_benefit,
        }
    }
}
