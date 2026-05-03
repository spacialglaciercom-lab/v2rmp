use clap::Parser;
use v2rmp::core::extract::{
    run_extract, BBoxRequest, ExtractRequest, ExtractSource, RoadClass,
};

#[derive(Parser, Debug)]
#[command(name = "rmpca-extract")]
#[command(about = "Extract road networks from Overture Maps or OpenStreetMap", long_about = None)]
struct Args {
    /// Data source: "osm" or "overture"
    #[arg(short, long, default_value = "overture")]
    source: String,

    /// Bounding box: "min_lat,min_lon,max_lat,max_lon"
    #[arg(short, long)]
    bbox: String,

    /// Output GeoJSON file path
    #[arg(short, long)]
    output: String,

    /// Path to OSM PBF file (required when source is "osm")
    #[arg(short, long)]
    pbf_path: Option<String>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let bbox_parts: Vec<&str> = args.bbox.split(',').collect();
    if bbox_parts.len() != 4 {
        anyhow::bail!("Invalid bbox format. Expected: min_lat,min_lon,max_lat,max_lon");
    }
    let min_lat: f64 = bbox_parts[0].parse()?;
    let min_lon: f64 = bbox_parts[1].parse()?;
    let max_lat: f64 = bbox_parts[2].parse()?;
    let max_lon: f64 = bbox_parts[3].parse()?;
    let source = match args.source.to_lowercase().as_str() {
        "osm" => ExtractSource::Osm,
        "overture" => ExtractSource::Overture,
        _ => anyhow::bail!("Invalid source. Must be 'osm' or 'overture'"),
    };
    if source == ExtractSource::Osm && args.pbf_path.is_none() {
        anyhow::bail!("--pbf-path is required when using OSM source");
    }
    let request = ExtractRequest {
        source,
        bbox: BBoxRequest { min_lon, min_lat, max_lon, max_lat },
        road_classes: RoadClass::all_vehicle(),
        output_path: args.output.clone(),
        pbf_path: args.pbf_path,
    };
    println!("Extracting road network...");
    println!("  Source: {:?}", request.source);
    println!("  BBox: [{:.4}, {:.4}, {:.4}, {:.4}]", min_lat, min_lon, max_lat, max_lon);
    if let Some(ref pbf) = request.pbf_path {
        println!("  PBF file: {}", pbf);
    }
    println!("  Output: {}", args.output);
    println!();
    let result = run_extract(&request)?;
    println!("✓ Extraction complete!");
    println!("  Nodes: {}", result.nodes);
    println!("  Edges: {}", result.edges);
    println!("  Total distance: {:.2} km", result.total_km);
    println!("  Output: {}", result.output_path);
    Ok(())
}
