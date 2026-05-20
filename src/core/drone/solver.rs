use crate::core::drone::{DroneModel, DroneSpec, DroneVrpInstance, DroneVrpResult};
use anyhow::Result;

pub struct DroneSolver;

impl DroneSolver {
    pub fn solve(instance: &DroneVrpInstance) -> Result<DroneVrpResult> {
        let mut spec = match instance.drone_model {
            DroneModel::FlyCart30 => DroneSpec::flycart30(),
            DroneModel::Wing => DroneSpec::wing(),
        };

        if let Some(cap) = instance.battery_capacity_wh {
            spec.battery_capacity_wh = cap;
        }

        let mut routes = Vec::new();
        let mut unvisited: Vec<usize> = (0..instance.customers.len()).collect();
        let mut total_energy = 0.0;
        let mut violations = Vec::new();

        while !unvisited.is_empty() {
            let mut current_route = Vec::new();
            let mut current_loc = instance.depot;
            let mut current_battery = spec.battery_capacity_wh;
            let mut current_payload = unvisited
                .iter()
                .map(|&i| instance.demands_kg[i])
                .sum::<f64>();

            while !unvisited.is_empty() {
                let nearest_idx = Self::find_nearest(current_loc, &unvisited, &instance.customers);
                let customer_idx = unvisited[nearest_idx];
                let customer_loc = instance.customers[customer_idx];

                // Check if customer is in a no-fly zone
                let mut in_no_fly_zone = false;
                for nfz in &instance.no_fly_zones {
                    let dist_to_nfz =
                        Self::haversine_m(customer_loc, [nfz.center_lat, nfz.center_lon]);
                    if dist_to_nfz <= nfz.radius_m {
                        violations.push(format!(
                            "Customer {} is inside no-fly zone {}",
                            customer_idx, nfz.id
                        ));
                        in_no_fly_zone = true;
                        break;
                    }
                }

                if in_no_fly_zone {
                    unvisited.remove(nearest_idx);
                    continue;
                }

                let dist = Self::haversine_m(current_loc, customer_loc);
                let energy_to_customer =
                    spec.calculate_energy_wh(dist, current_payload, instance.wind_speed_ms);

                // Return to home energy check
                let dist_to_depot =
                    Self::haversine_m(instance.customers[customer_idx], instance.depot);
                let energy_to_depot = spec.calculate_energy_wh(
                    dist_to_depot,
                    current_payload - instance.demands_kg[customer_idx],
                    instance.wind_speed_ms,
                );

                if current_battery >= (energy_to_customer + energy_to_depot) {
                    current_battery -= energy_to_customer;
                    total_energy += energy_to_customer;
                    current_payload -= instance.demands_kg[customer_idx];
                    current_loc = instance.customers[customer_idx];
                    current_route.push(customer_idx);
                    unvisited.remove(nearest_idx);
                } else {
                    // Cannot reach next customer and return safely, go back to depot
                    let dist_to_depot = Self::haversine_m(current_loc, instance.depot);
                    let energy_to_depot = spec.calculate_energy_wh(
                        dist_to_depot,
                        current_payload,
                        instance.wind_speed_ms,
                    );
                    total_energy += energy_to_depot;
                    break;
                }
            }

            if !current_route.is_empty() {
                routes.push(current_route);
            } else if !unvisited.is_empty() {
                violations.push(format!(
                    "Customer {} unreachable with battery limits",
                    unvisited[0]
                ));
                unvisited.remove(0);
            }
        }

        Ok(DroneVrpResult {
            routes,
            energy_used_wh: total_energy,
            algorithm: "Greedy Nearest Neighbor (Energy Constrained)".to_string(),
            violations,
        })
    }

    fn find_nearest(current: [f64; 2], unvisited: &[usize], customers: &[[f64; 2]]) -> usize {
        let mut min_dist = f64::MAX;
        let mut best_idx = 0;
        for (i, &cust_idx) in unvisited.iter().enumerate() {
            let dist = Self::haversine_m(current, customers[cust_idx]);
            if dist < min_dist {
                min_dist = dist;
                best_idx = i;
            }
        }
        best_idx
    }

    fn haversine_m(p1: [f64; 2], p2: [f64; 2]) -> f64 {
        let r = 6371000.0; // Earth radius in meters
        let phi1 = p1[0].to_radians();
        let phi2 = p2[0].to_radians();
        let d_phi = (p2[0] - p1[0]).to_radians();
        let d_lambda = (p2[1] - p1[1]).to_radians();

        let a =
            (d_phi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (d_lambda / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        r * c
    }
}
