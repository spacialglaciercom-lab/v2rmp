use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

use crate::core::clean::{clean_geojson, CleanOptions};
use crate::core::compile::CompileRequest;
use crate::core::extract::{BBoxRequest, ExtractRequest, ExtractSource, RoadClass};
use crate::core::optimize::{OnewayMode, OptimizeRequest, TurnPenalties};

/// rmpca - Route optimization and data extraction
#[derive(Parser)]
#[command(name = "rmpca", version = "0.3.9")]
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
    /// List available resources (maps, routes)
    List(ListArgs),
    /// Execute a JSON-encoded task plan
    Agent(AgentArgs),
}

#[derive(clap::Args)]
struct ListArgs {
    /// Resource type to list
    #[arg(value_enum, default_value = "maps")]
    resource: ResourceType,
}

#[derive(clap::ValueEnum, Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ResourceType {
    Maps,
    Routes,
}

#[derive(clap::Args)]
struct AgentArgs {
    /// Path to JSON task file (or '-' for stdin)
    #[arg(short, long, default_value = "-")]
    task: String,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AgentTask {
    Extract(ExtractArgs),
    Compile(CompileArgs),
    Clean(CleanArgs),
    Optimize(OptimizeArgs),
    Vrp(VrpArgs),
    Pipeline(PipelineArgs),
}

// ── VRP ───────────────────────────────────────────────────────────────

#[derive(clap::ValueEnum, Clone, Debug, Serialize, Deserialize)]
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

#[derive(clap::Args, Serialize, Deserialize)]
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

#[derive(clap::Args, Serialize, Deserialize)]
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

    /// Path to local OSM PBF file (required for source=osm)
    #[arg(long)]
    pbf: Option<String>,
}

// ── Compile ───────────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
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

#[derive(clap::Args, Serialize, Deserialize)]
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

#[derive(clap::Args, Serialize, Deserialize)]
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

    /// Number of vehicles (for VRP)
    #[arg(short, long, default_value_t = 1)]
    vehicles: usize,

    /// Solver algorithm (clarke_wright, sweep, two_opt, or_opt, default)
    #[arg(short, long, default_value = "clarke_wright")]
    solver: String,
}

// ── Pipeline ──────────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
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

async fn run_extract_cmd(args: ExtractArgs, json: bool) -> Result<()> {
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

    if let Some(pbf) = args.pbf {
        // If PBF path is provided, we can pass it via env or update ExtractRequest
        // For now, let's set the env var that run_osm_extract uses
        std::env::set_var("OSM_PBF_PATH", pbf);
    }

    let result = crate::core::extract::run_extract(&req).await?;

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

async fn run_optimize_cmd(args: OptimizeArgs, json: bool) -> Result<()> {
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
        num_vehicles: args.vehicles,
        solver_id: args.solver,
    };

    let result = crate::core::optimize::run_optimize(&req).await?;

    if json {
        output_json(&result)?;
    } else {
        tracing::info!("Optimization complete!");
        tracing::info!("Total distance: {:.2} km", result.total_distance_km);
        tracing::info!("Stops/Segments: {}", result.total_segments);
        tracing::info!("Vehicles used: {}", result.num_routes);
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

async fn run_vrp_cmd(args: VrpArgs, json: bool) -> Result<()> {
    // VRP command now uses the same core as optimize but focused on multi-vehicle
    let depot = if args.depot.is_empty() {
        None
    } else {
        Some(parse_depot(&args.depot[0])?)
    };

    let solver_id = match args.algo {
        VrpAlgorithm::Greedy => "default",
        VrpAlgorithm::Savings => "clarke_wright",
        VrpAlgorithm::LocalSearch => "or_opt",
        VrpAlgorithm::SimulatedAnnealing => "sweep",
    }.to_string();

    let req = OptimizeRequest {
        cache_file: args.input.clone(),
        route_file: Some(format!("{}/route.gpx", args.output_dir)),
        turn_penalties: TurnPenalties::default(),
        depot,
        oneway_mode: OnewayMode::Respect,
        num_vehicles: args.vehicles,
        solver_id,
    };

    let result = crate::core::optimize::run_optimize(&req).await?;

    if json {
        output_json(&result)?;
    } else {
        tracing::info!("VRP Solving complete!");
        tracing::info!("Total distance: {:.2} km", result.total_distance_km);
        tracing::info!("Vehicles: {}", result.num_routes);
        tracing::info!("Elapsed: {} ms", result.elapsed_ms);
    }
    
    Ok(())
}

async fn run_pipeline_cmd(args: PipelineArgs, json: bool) -> Result<()> {
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
        crate::core::extract::run_extract(&extract_req).await.context("Pipeline failed at stage 'extract'")?;

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
        num_vehicles: 1,
        solver_id: "clarke_wright".to_string(),
    };
    let optimize_result =
        crate::core::optimize::run_optimize(&optimize_req).await.context("Pipeline failed at stage 'optimize'")?;

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
            "Optimize: {:.2} km total, {} vehicles",
            pipeline_result.optimize.total_distance_km,
            pipeline_result.optimize.num_routes
        );
        tracing::info!("Files in: {}/", args.output_dir);
    }

    Ok(())
}

async fn run_agent_cmd(args: AgentArgs, json: bool) -> Result<()> {
    let input: Box<dyn std::io::Read> = if args.task == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&args.task)?)
    };

    let task: AgentTask = serde_json::from_reader(input)
        .context("Failed to parse agent task JSON")?;

    match task {
        AgentTask::Extract(a) => run_extract_cmd(a, json).await,
        AgentTask::Compile(a) => run_compile_cmd(a, json),
        AgentTask::Clean(a) => run_clean_cmd(a, json),
        AgentTask::Optimize(a) => run_optimize_cmd(a, json).await,
        AgentTask::Vrp(a) => run_vrp_cmd(a, json).await,
        AgentTask::Pipeline(a) => run_pipeline_cmd(a, json).await,
    }
}

fn run_list_cmd(args: ListArgs, json: bool) -> Result<()> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                match args.resource {
                    ResourceType::Maps => {
                        if ext == "rmp" {
                            files.push(path.display().to_string());
                        }
                    }
                    ResourceType::Routes => {
                        if ext == "gpx" || ext == "geojson" {
                            files.push(path.display().to_string());
                        }
                    }
                }
            }
        }
    }

    if json {
        output_json(&files)?;
    } else {
        println!("Available {:?}:", args.resource);
        for f in files {
            println!("  - {f}");
        }
    }
    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    init_tracing();

    let result = match cli.command {
        Commands::Extract(args) => run_extract_cmd(args, cli.json).await,
        Commands::Compile(args) => run_compile_cmd(args, cli.json),
        Commands::Clean(args) => run_clean_cmd(args, cli.json),
        Commands::Optimize(args) => run_optimize_cmd(args, cli.json).await,
        Commands::Vrp(args) => run_vrp_cmd(args, cli.json).await,
        Commands::Pipeline(args) => run_pipeline_cmd(args, cli.json).await,
        Commands::List(args) => run_list_cmd(args, cli.json),
        Commands::Agent(args) => run_agent_cmd(args, cli.json).await,
    };

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }

    Ok(())
}
