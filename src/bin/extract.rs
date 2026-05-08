#![cfg(feature = "extract")]
use anyhow::Result;
use clap::Parser;
use v2rmp::core::extract::{ExtractRequest, ExtractSource, RoadClass};

/// Extract road network data from OSM or Overture Maps
#[derive(Parser, Debug)]
#[command(name = "rmpca-extract")]
#[command(version, about, long_about = None)]
struct Args {
    /// Data source: osm or overture
    #[arg(long, default_value = "overture")]
    source: String,

    /// Bounding box: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT
    #[arg(long)]
    bbox: String,

    /// Path to local OSM PBF file (optional, falls back to Overpass API)
    #[arg(long)]
    pbf: Option<String>,

    /// Road classes to include (comma-separated)
    #[arg(long, value_delimiter = ',')]
    road_classes: Vec<String>,

    /// Output file path
    #[arg(short, long, default_value = "extract-output.geojson")]
    output: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    // Parse bounding box
    let bbox_parts: Vec<f64> = args
        .bbox
        .split(',')
        .map(|s| {
            s.parse::<f64>()
                .unwrap_or_else(|_| panic!("Invalid bbox value: {}", s))
        })
        .collect();

    if bbox_parts.len() != 4 {
        anyhow::bail!("Bounding box must have exactly 4 values: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT");
    }

    // Parse road classes
    let road_classes = if args.road_classes.is_empty() {
        RoadClass::all_vehicle()
    } else {
        args.road_classes
            .into_iter()
            .map(|s| -> Result<RoadClass> {
                match s.as_str() {
                    "residential" => Ok(RoadClass::Residential),
                    "tertiary" => Ok(RoadClass::Tertiary),
                    "secondary" => Ok(RoadClass::Secondary),
                    "primary" => Ok(RoadClass::Primary),
                    "trunk" => Ok(RoadClass::Trunk),
                    "motorway" => Ok(RoadClass::Motorway),
                    "unclassified" => Ok(RoadClass::Unclassified),
                    "living_street" => Ok(RoadClass::LivingStreet),
                    "service" => Ok(RoadClass::Service),
                    "secondary_link" => Ok(RoadClass::SecondaryLink),
                    "primary_link" => Ok(RoadClass::PrimaryLink),
                    "trunk_link" => Ok(RoadClass::TrunkLink),
                    "motorway_link" => Ok(RoadClass::MotorwayLink),
                    _ => anyhow::bail!("Unknown road class: {}", s),
                }
            })
            .collect::<Result<Vec<RoadClass>>>()?
    };

    // Parse source
    let source = match args.source.as_str() {
        "osm" => ExtractSource::Osm,
        "overture" => ExtractSource::Overture,
        _ => anyhow::bail!("Unknown source: {}", args.source),
    };

    tracing::info!("Starting extraction...");
    tracing::info!("Source: {:?}", source);
    tracing::info!(
        "Bounding box: [{:.4}, {:.4}, {:.4}, {:.4}]",
        bbox_parts[0],
        bbox_parts[1],
        bbox_parts[2],
        bbox_parts[3]
    );
    tracing::info!("Road classes: {:?}", road_classes);

    let request = ExtractRequest {
        source,
        bbox: v2rmp::core::extract::BBoxRequest {
            min_lon: bbox_parts[0],
            min_lat: bbox_parts[1],
            max_lon: bbox_parts[2],
            max_lat: bbox_parts[3],
        },
        road_classes,
        output_path: args.output,
        pbf_path: args.pbf,
    };

    let result = v2rmp::core::extract::run_extract(&request).await?;

    tracing::info!("Extraction complete!");
    tracing::info!("Nodes: {}", result.nodes);
    tracing::info!("Edges: {}", result.edges);
    tracing::info!("Total road length: {:.2} km", result.total_km);
    tracing::info!("Output: {}", result.output_path);

    Ok(())
}
