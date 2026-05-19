/// One-shot binary: PMTiles extract → compile → CPP optimize → GPX
use anyhow::Result;
use v2rmp::core::compile::{run_compile, CompileRequest};
use v2rmp::core::optimize::{run_optimize, OnewayMode, OptimizeRequest, TurnPenalties};
use v2rmp::core::pmtiles_extract::{run_pmtiles_extract, PmtilesExtractRequest};

#[tokio::main]
async fn main() -> Result<()> {
    let pmtiles_path = "montreal-v2026-02.pmtiles";
    let geojson_path = "montreal_pmtiles_extract.geojson";
    let rmp_path = "montreal_pmtiles.rmp";
    let gpx_path = "montreal_pmtiles_route.gpx";

    // Montreal bounding box
    let min_lon = -73.73;
    let min_lat = 45.42;
    let max_lon = -73.48;
    let max_lat = 45.57;

    // Step 1: Extract roads from PMTiles
    println!("Step 1/3: Extracting roads from PMTiles...");
    let extract_req = PmtilesExtractRequest {
        pmtiles_path: pmtiles_path.to_string(),
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        output_path: geojson_path.to_string(),
        zoom: None,
        layer_name: Some("transportation".to_string()),
    };
    let extract_res = run_pmtiles_extract(&extract_req).await?;
    println!(
        "  → Extracted {} features from {} tiles",
        extract_res.features, extract_res.tiles_fetched
    );

    // Step 2: Compile GeoJSON → .rmp
    println!("Step 2/3: Compiling GeoJSON → .rmp...");
    let compile_req = CompileRequest {
        input_geojson: geojson_path.to_string(),
        output_rmp: rmp_path.to_string(),
        prune_disconnected: false,
        compress: false,
        road_classes: vec![],
        clean_options: None,
        prune_spurs: false,
    };
    let compile_res = run_compile(&compile_req)?;
    println!(
        "  → {} nodes, {} edges ({:.1} KB)",
        compile_res.node_count,
        compile_res.edge_count,
        compile_res.output_size_bytes as f64 / 1024.0
    );

    // Step 3: CPP optimization → GPX
    println!("Step 3/3: Running CPP optimization...");
    let optimize_req = OptimizeRequest {
        cache_file: rmp_path.to_string(),
        route_file: Some(gpx_path.to_string()),
        turn_penalties: TurnPenalties::default(),
        depot: None,
        oneway_mode: OnewayMode::Respect,
        mode: v2rmp::core::optimize::SolverMode::Cpp,
        cpp_engine: v2rmp::core::optimize::CppEngine::default(),
        num_vehicles: 1,
        solver_id: "default".to_string(),
        coordinates: None,
    };
    let optimize_res = run_optimize(&optimize_req).await?;
    println!("  → Distance: {:.2} km", optimize_res.total_distance_km);
    println!("  → Segments: {}", optimize_res.total_segments);
    println!("  → Deadhead: {:.2} km", optimize_res.deadhead_distance_km);
    println!("  → Efficiency: {:.1}%", optimize_res.efficiency_pct);
    println!(
        "  → Turns: L={} R={} U={} S={}",
        optimize_res.turns.left,
        optimize_res.turns.right,
        optimize_res.turns.u_turn,
        optimize_res.turns.straight
    );

    println!(
        "\nGPX written to: {}/{}",
        std::env::current_dir()?.display(),
        gpx_path
    );
    Ok(())
}
