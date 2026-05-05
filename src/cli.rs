use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;

use crate::core::clean::{clean_geojson, CleanOptions};
use crate::core::compile::CompileRequest;
use crate::core::extract::{BBoxRequest, ExtractRequest, ExtractSource, RoadClass};
use crate::core::optimize::{OnewayMode, OptimizeRequest, TurnPenalties};

/// rmpca - Route optimization and data extraction
#[derive(Parser)]
#[command(name = "rmpca", version = "0.3.8")]
#[command(about = "Route optimization and data extraction")]
struct Cli {
    /// Output results as JSON (for machine / agent consumption)
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract road network data from Overture Maps
    Extract(ExtractArgs),
    /// Compile GeoJSON into binary .rmp format
    Compile(CompileArgs),
    /// Clean a GeoJSON road network
    Clean(CleanArgs),
    /// Optimize a route (Chinese Postman Problem)
    Optimize(OptimizeArgs),
    /// Solve Vehicle Routing Problem (VRP) for multiple agents
    Vrp(VrpArgs),
    /// Run the full pipeline: extract -> clean -> compile -> optimize
    Pipeline(PipelineArgs),
}

// ── VRP ───────────────────────────────────────────────────────────────

#[derive(clap::ValueEnum, Clone, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum VrpAlgorithm {
    /// Simple greedy nearest-neighbor approach
    Greedy,
    /// Savings algorithm (Clarke-Wright)
    Savings,
    /// Metaheuristic: Local search / Tabu search
    LocalSearch,
    /// Metaheuristic: Simulated Annealing
    SimulatedAnnealing,
}

#[derive(clap::Args)]
struct VrpArgs {
    /// Input .rmp cache file
    #[arg(short, long)]
    input: String,

    /// Output directory for agent GPX routes
    #[arg(short, long, default_value = "routes/")]
    output_dir: String,

    /// Number of vehicles/agents
    #[arg(short, long, default_value_t = 1)]
    vehicles: usize,

    /// Algorithm to use for VRP
    #[arg(short, long, value_enum, default_value_t = VrpAlgorithm::Greedy)]
    algo: VrpAlgorithm,

    /// Vehicle capacity (if applicable)
    #[arg(long)]
    capacity: Option<f64>,

    /// Depot coordinates: LAT,LON (can be specified multiple times)
    #[arg(long, action = clap::ArgAction::Append)]
    depot: Vec<String>,

    /// Path to a JSON/CSV file containing specific waypoints to visit
    #[arg(long)]
    waypoints: Option<String>,
}

// ── Extract ───────────────────────────────────────────────────────────

#[derive(clap::Args)]
struct ExtractArgs {
    /// Bounding box: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT
    #[arg(long, allow_hyphen_values = true)]
    bbox: String,

    /// Data source: overture (default)
    #[arg(long, default_value = "overture")]
    source: String,

    /// Road classes to include (comma-separated, default: all vehicle)
    #[arg(long, value_delimiter = ',')]
    road_classes: Vec<String>,

    /// Output GeoJSON file path
    #[arg(short, long, default_value = "extract-output.geojson")]
    output: String,
}

// ── Compile ───────────────────────────────────────────────────────────

#[derive(clap::Args)]
struct CompileArgs {
    /// Input GeoJSON file
    #[arg(short, long)]
    input: String,

    /// Output .rmp file
    #[arg(short, long)]
    output: String,

    /// Run cleaning before compilation (with default clean options)
    #[arg(long)]
    clean: bool,
}

// ── Clean ─────────────────────────────────────────────────────────────

#[derive(clap::Args)]
struct CleanArgs {
    /// Input GeoJSON file
    #[arg(short, long)]
    input: String,

    /// Output cleaned GeoJSON file
    #[arg(short, long)]
    output: String,

    /// Minimum edge length in meters (default: 0.1)
    #[arg(long)]
    min_length_m: Option<f64>,

    /// Node snapping distance in meters (default: 1.0)
    #[arg(long)]
    node_snap_m: Option<f64>,

    /// Keep only the N largest connected components (default: 1)
    #[arg(long)]
    max_components: Option<usize>,

    /// Geometry simplification tolerance in meters (default: 0.0)
    #[arg(long)]
    simplify_tolerance_m: Option<f64>,

    /// Skip edge deduplication
    #[arg(long)]
    no_dedupe: bool,

    /// Skip isolated node removal
    #[arg(long)]
    no_remove_isolates: bool,
}

// ── Optimize ──────────────────────────────────────────────────────────

#[derive(clap::Args)]
struct OptimizeArgs {
    /// Input .rmp cache file
    #[arg(short, long)]
    input: String,

    /// Output route GPX file
    #[arg(short, long)]
    output: Option<String>,

    /// Depot coordinates: LAT,LON
    #[arg(long)]
    depot: Option<String>,

    /// Oneway mode: respect, ignore, reverse (default: respect)
    #[arg(long, default_value = "respect")]
    oneway: String,

    /// Left turn penalty (default: 1.0)
    #[arg(long, default_value_t = 1.0)]
    left_penalty: f64,

    /// Right turn penalty (default: 0.0)
    #[arg(long, default_value_t = 0.0)]
    right_penalty: f64,

    /// U-turn penalty (default: 5.0)
    #[arg(long, default_value_t = 5.0)]
    uturn_penalty: f64,
}

// ── Pipeline ──────────────────────────────────────────────────────────

#[derive(clap::Args)]
struct PipelineArgs {
    /// Bounding box: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT
    #[arg(long, allow_hyphen_values = true)]
    bbox: String,

    /// Data source: overture (default)
    #[arg(long, default_value = "overture")]
    source: String,

    /// Output directory for all intermediate and final files
    #[arg(short, long, default_value = ".")]
    output_dir: String,

    /// Depot coordinates: LAT,LON
    #[arg(long)]
    depot: Option<String>,
}

#[derive(Serialize)]
struct PipelineResult {
    extract: crate::core::extract::ExtractResult,
    clean: crate::core::clean::CleanStats,
    compile: crate::core::compile::CompileResult,
    optimize: crate::core::optimize::OptimizeResult,
}

// ── Parsing helpers ───────────────────────────────────────────────────

fn parse_bbox(s: &str) -> Result<(f64, f64, f64, f64)> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|v| v.parse::<f64>().context(format!("Invalid bbox value: {v}")))
        .collect::<Result<Vec<f64>>>()?;
    if parts.len() != 4 {
        anyhow::bail!("Bounding box must have exactly 4 values: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT");
    }
    Ok((parts[0], parts[1], parts[2], parts[3]))
}

fn parse_road_classes(classes: &[String]) -> Result<Vec<RoadClass>> {
    if classes.is_empty() {
        return Ok(RoadClass::all_vehicle());
    }
    classes
        .iter()
        .map(|s| match s.as_str() {
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
            other => anyhow::bail!("Unknown road class: {other}"),
        })
        .collect()
}

fn parse_source(s: &str) -> Result<ExtractSource> {
    match s {
        "osm" => Ok(ExtractSource::Osm),
        "overture" => Ok(ExtractSource::Overture),
        other => anyhow::bail!("Unknown source: {other}"),
    }
}

fn parse_depot(s: &str) -> Result<(f64, f64)> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|v| v.parse::<f64>().context(format!("Invalid depot value: {v}")))
        .collect::<Result<Vec<f64>>>()?;
    if parts.len() != 2 {
        anyhow::bail!("Depot must be LAT,LON (2 values)");
    }
    Ok((parts[0], parts[1]))
}

fn parse_oneway(s: &str) -> Result<OnewayMode> {
    match s {
        "respect" => Ok(OnewayMode::Respect),
        "ignore" => Ok(OnewayMode::Ignore),
        "reverse" => Ok(OnewayMode::Reverse),
        other => anyhow::bail!("Unknown oneway mode: {other} (respect|ignore|reverse)"),
    }
}

// ── Output helpers ────────────────────────────────────────────────────

fn output_json<T: Serialize>(value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    println!("{json}");
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .try_init();
}

// ── Command handlers ──────────────────────────────────────────────────

fn run_extract_cmd(args: ExtractArgs, json: bool) -> Result<()> {
    let (min_lon, min_lat, max_lon, max_lat) = parse_bbox(&args.bbox)?;
    let road_classes = parse_road_classes(&args.road_classes)?;
    let source = parse_source(&args.source)?;

    if !json {
        tracing::info!("Starting extraction...");
        tracing::info!("Source: {:?}", source);
        tracing::info!(
            "Bounding box: [{min_lon:.4}, {min_lat:.4}, {max_lon:.4}, {max_lat:.4}]"
        );
    }

    let req = ExtractRequest {
        source,
        bbox: BBoxRequest {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        },
        road_classes,
        output_path: args.output.clone(),
    };

    let result = crate::core::extract::run_extract(&req)?;

    if json {
        output_json(&result)?;
    } else {
        tracing::info!("Extraction complete!");
        tracing::info!("Nodes: {}", result.nodes);
        tracing::info!("Edges: {}", result.edges);
        tracing::info!("Total road length: {:.2} km", result.total_km);
        tracing::info!("Output: {}", result.output_path);
    }

    Ok(())
}

fn run_compile_cmd(args: CompileArgs, json: bool) -> Result<()> {
    let clean_options = if args.clean {
        Some(CleanOptions::default())
    } else {
        None
    };

    if !json {
        tracing::info!("Compiling {} -> {}", args.input, args.output);
    }

    let req = CompileRequest {
        input_geojson: args.input,
        output_rmp: args.output,
        compress: false,
        road_classes: vec![],
        clean_options,
    };

    let result = crate::core::compile::run_compile(&req)?;

    if json {
        output_json(&result)?;
    } else {
        tracing::info!("Compilation complete!");
        tracing::info!("Nodes: {}", result.node_count);
        tracing::info!("Edges: {}", result.edge_count);
        tracing::info!(
            "Size: {} -> {} bytes ({:.1}% ratio)",
            result.input_size_bytes,
            result.output_size_bytes,
            if result.input_size_bytes > 0 {
                (result.output_size_bytes as f64 / result.input_size_bytes as f64) * 100.0
            } else {
                0.0
            }
        );
        tracing::info!("Elapsed: {} ms", result.elapsed_ms);
    }

    Ok(())
}

fn run_clean_cmd(args: CleanArgs, json: bool) -> Result<()> {
    let mut opts = CleanOptions::default();
    if let Some(v) = args.min_length_m {
        opts.min_length_m = v;
    }
    if let Some(v) = args.node_snap_m {
        opts.node_snap_m = v;
    }
    if let Some(v) = args.max_components {
        opts.max_components = v;
    }
    if let Some(v) = args.simplify_tolerance_m {
        opts.simplify_tolerance_m = v;
    }
    if args.no_dedupe {
        opts.dedupe_edges = false;
    }
    if args.no_remove_isolates {
        opts.remove_isolates = false;
    }

    if !json {
        tracing::info!("Cleaning {} -> {}", args.input, args.output);
    }

    let input_data = std::fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read input file: {}", args.input))?;
    let geojson: geojson::FeatureCollection = serde_json::from_str(&input_data)
        .with_context(|| "Failed to parse GeoJSON FeatureCollection")?;

    let (cleaned, stats, warnings) = clean_geojson(&geojson, &opts)?;

    let output_str = serde_json::to_string_pretty(&cleaned)?;
    std::fs::write(&args.output, &output_str)
        .with_context(|| format!("Failed to write output file: {}", args.output))?;

    if json {
        #[derive(Serialize)]
        struct CleanOutput {
            stats: crate::core::clean::CleanStats,
            warnings: Vec<String>,
            output_file: String,
        }
        output_json(&CleanOutput {
            stats,
            warnings,
            output_file: args.output,
        })?;
    } else {
        tracing::info!("Cleaning complete!");
        tracing::info!("Features: {} -> {}", stats.input_features, stats.output_features);
        for w in &warnings {
            tracing::warn!("Warning: {w}");
        }
    }

    Ok(())
}

fn run_optimize_cmd(args: OptimizeArgs, json: bool) -> Result<()> {
    let depot = args.depot.as_deref().map(parse_depot).transpose()?;
    let oneway_mode = parse_oneway(&args.oneway)?;

    if !json {
        tracing::info!("Optimizing route from {}", args.input);
    }

    let req = OptimizeRequest {
        cache_file: args.input.clone(),
        route_file: args.output.clone(),
        turn_penalties: TurnPenalties {
            left: args.left_penalty,
            right: args.right_penalty,
            u_turn: args.uturn_penalty,
        },
        depot,
        oneway_mode,
    };

    let result = crate::core::optimize::run_optimize(&req)?;

    if json {
        output_json(&result)?;
    } else {
        tracing::info!("Optimization complete!");
        tracing::info!("Total distance: {:.2} km", result.total_distance_km);
        tracing::info!("Segments: {}", result.total_segments);
        tracing::info!("Deadhead: {:.2} km", result.deadhead_distance_km);
        tracing::info!("Efficiency: {:.1}%", result.efficiency_pct);
        tracing::info!(
            "Turns: {} left, {} right, {} u-turn, {} straight",
            result.turns.left,
            result.turns.right,
            result.turns.u_turn,
            result.turns.straight
        );
        tracing::info!("Elapsed: {} ms", result.elapsed_ms);
        if let Some(ref path) = args.output {
            tracing::info!("Route written to: {path}");
        }
    }

    Ok(())
}

fn run_vrp_cmd(args: VrpArgs, _json: bool) -> Result<()> {
    tracing::info!("VRP solving requested for {}", args.input);
    tracing::info!("Algorithm: {:?}", args.algo);
    tracing::info!("Vehicles: {}, Depots: {}, Waypoints: {:?}", 
        args.vehicles, 
        args.depot.len(),
        args.waypoints
    );
    
    // Parse all depots
    let _depots: Vec<(f64, f64)> = args.depot.iter()
        .map(|s| parse_depot(s))
        .collect::<Result<Vec<_>>>()?;

    // Placeholder for actual VRP logic
    tracing::warn!("VRP implementation ({:?}) is currently a wireframe.", args.algo);
    
    Ok(())
}

fn run_pipeline_cmd(args: PipelineArgs, json: bool) -> Result<()> {
    let (min_lon, min_lat, max_lon, max_lat) = parse_bbox(&args.bbox)?;
    let source = parse_source(&args.source)?;
    let depot = args.depot.as_deref().map(parse_depot).transpose()?;

    // Ensure output directory exists
    std::fs::create_dir_all(&args.output_dir)?;

    let extract_path = format!("{}/extract.geojson", args.output_dir);
    let cleaned_path = format!("{}/cleaned.geojson", args.output_dir);
    let rmp_path = format!("{}/network.rmp", args.output_dir);
    let route_path = format!("{}/route.gpx", args.output_dir);

    // Stage 1: Extract
    if !json {
        tracing::info!("=== Pipeline Stage 1: Extract ===");
    }
    let extract_req = ExtractRequest {
        source,
        bbox: BBoxRequest {
            min_lon,
            min_lat,
            max_lon,
            max_lat,
        },
        road_classes: RoadClass::all_vehicle(),
        output_path: extract_path.clone(),
    };
    let extract_result =
        crate::core::extract::run_extract(&extract_req).context("Pipeline failed at stage 'extract'")?;

    // Stage 2: Clean
    if !json {
        tracing::info!("=== Pipeline Stage 2: Clean ===");
    }
    let input_data = std::fs::read_to_string(&extract_path)
        .context("Pipeline failed reading extracted GeoJSON")?;
    let geojson: geojson::FeatureCollection = serde_json::from_str(&input_data)
        .context("Pipeline failed parsing extracted GeoJSON")?;
    let (cleaned, clean_stats, _warnings) = clean_geojson(&geojson, &CleanOptions::default())
        .context("Pipeline failed at stage 'clean'")?;
    let cleaned_str = serde_json::to_string_pretty(&cleaned)?;
    std::fs::write(&cleaned_path, &cleaned_str)
        .context("Pipeline failed writing cleaned GeoJSON")?;

    // Stage 3: Compile
    if !json {
        tracing::info!("=== Pipeline Stage 3: Compile ===");
    }
    let compile_req = CompileRequest {
        input_geojson: cleaned_path.clone(),
        output_rmp: rmp_path.clone(),
        compress: false,
        road_classes: vec![],
        clean_options: None,
    };
    let compile_result =
        crate::core::compile::run_compile(&compile_req).context("Pipeline failed at stage 'compile'")?;

    // Stage 4: Optimize
    if !json {
        tracing::info!("=== Pipeline Stage 4: Optimize ===");
    }
    let optimize_req = OptimizeRequest {
        cache_file: rmp_path.clone(),
        route_file: Some(route_path.clone()),
        turn_penalties: TurnPenalties::default(),
        depot,
        oneway_mode: OnewayMode::default(),
    };
    let optimize_result =
        crate::core::optimize::run_optimize(&optimize_req).context("Pipeline failed at stage 'optimize'")?;

    // Output
    let pipeline_result = PipelineResult {
        extract: extract_result,
        clean: clean_stats,
        compile: compile_result,
        optimize: optimize_result,
    };

    if json {
        output_json(&pipeline_result)?;
    } else {
        tracing::info!("=== Pipeline Complete ===");
        tracing::info!(
            "Extract: {} nodes, {} edges, {:.2} km",
            pipeline_result.extract.nodes,
            pipeline_result.extract.edges,
            pipeline_result.extract.total_km
        );
        tracing::info!(
            "Clean: {} -> {} features",
            pipeline_result.clean.input_features,
            pipeline_result.clean.output_features
        );
        tracing::info!(
            "Compile: {} nodes, {} edges, {} bytes",
            pipeline_result.compile.node_count,
            pipeline_result.compile.edge_count,
            pipeline_result.compile.output_size_bytes
        );
        tracing::info!(
            "Optimize: {:.2} km total, {:.1}% efficient",
            pipeline_result.optimize.total_distance_km,
            pipeline_result.optimize.efficiency_pct
        );
        tracing::info!("Files in: {}/", args.output_dir);
    }

    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    init_tracing();

    let result = match cli.command {
        Commands::Extract(args) => run_extract_cmd(args, cli.json),
        Commands::Compile(args) => run_compile_cmd(args, cli.json),
        Commands::Clean(args) => run_clean_cmd(args, cli.json),
        Commands::Optimize(args) => run_optimize_cmd(args, cli.json),
        Commands::Vrp(args) => run_vrp_cmd(args, cli.json),
        Commands::Pipeline(args) => run_pipeline_cmd(args, cli.json),
    };

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }

    Ok(())
}
