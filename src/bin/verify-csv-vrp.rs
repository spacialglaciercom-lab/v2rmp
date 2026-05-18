use v2rmp::core::vrp::registry::solve_with;
use v2rmp::core::vrp::types::{VRPSolverInput, VrpObjective};
use v2rmp::core::vrp::utils::parse_csv_stops;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let csv_path = "coords.csv";
    println!("Loading CSV from {}...", csv_path);

    let (stops, _depot_indices) = parse_csv_stops(csv_path).map_err(|e| anyhow::anyhow!(e))?;
    println!("Loaded {} stops.", stops.len());

    let matrix = v2rmp::core::vrp::utils::build_haversine_matrix(&stops, 40.0);

    let input = VRPSolverInput {
        locations: stops,
        num_vehicles: 1,
        vehicle_capacity: 100.0,
        objective: VrpObjective::MinDistance,
        matrix: Some(matrix),
        service_time_secs: Some(30.0),
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };

    println!("Running VRP solver (clarke_wright)...");
    let output = solve_with("clarke_wright", &input)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    println!("VRP Success!");
    println!("Total distance: {} km", output.total_distance_km);
    if let Some(routes) = output.routes {
        println!("Number of routes: {}", routes.len());
        for (i, route) in routes.iter().enumerate() {
            println!("Route {}: {} stops", i + 1, route.len());
            for stop in route {
                println!("  - {}", stop.label);
            }
        }

        let output_dir = "output_routes";
        std::fs::create_dir_all(output_dir)?;
        let output_path = format!("{}/route_20_coords.gpx", output_dir);
        if v2rmp::core::optimize::write_gpx_multi(&output_path, &routes).is_ok() {
            println!(
                "\n✅ Route saved to: {}/{}",
                std::env::current_dir()?.display(),
                output_path
            );

            // Generate OsmAnd Deep Link
            // Note: This requires the GPX to be hosted on a public URL.
            // Example placeholder:
            let public_base = "https://pub-37b94eb1edd1ce06999158bebd2f0a42.r2.dev"; // convention for R2 dev
            let gpx_url = format!("{}/{}", public_base, "route_20_coords.gpx");
            let osmand_link = v2rmp::core::vrp::utils::generate_osmand_import_url(&gpx_url);

            println!("\n🌍 OsmAnd Import Link (requires upload to R2):");
            println!("{}", osmand_link);
        }
    }

    Ok(())
}
