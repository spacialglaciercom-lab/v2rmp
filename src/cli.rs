use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

use crate::core::clean::{clean_geojson, CleanOptions};
use crate::core::compile::CompileRequest;
#[cfg(feature = "extract")]
use crate::core::elevation::{FuelCalculator, LocalDem};
#[cfg(feature = "extract")]
use crate::core::extract::{BBoxRequest, ExtractRequest, ExtractResult, ExtractSource, RoadClass};
use crate::core::optimize::{CppEngine, OnewayMode, OptimizeRequest, SolverMode, TurnPenalties};

/// rmpca - Route optimization and data extraction
#[derive(Parser)]
#[command(name = "rmpca", version = env!("CARGO_PKG_VERSION"))]
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
    #[cfg(feature = "extract")]
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
    #[cfg(feature = "extract")]
    Pipeline(PipelineArgs),
    /// List available resources (maps, routes)
    List(ListArgs),
    /// Execute a JSON-encoded task plan
    Agent(AgentArgs),
    /// Start a headless, long-running JSON-RPC/STDIO server for frontend integrations
    Serve(ServeArgs),
    /// Generate embeddings for text using fastembed
    #[cfg(feature = "ml")]
    Embed(EmbedArgs),
    /// DEM elevation queries from local GeoTIFF
    #[cfg(feature = "extract")]
    Elevation(ElevationArgs),
    /// Predict the best VRP solver for a set of stops
    #[cfg(feature = "ml")]
    PredictSolver(PredictSolverArgs),
    /// Predict expected route quality before solving
    #[cfg(feature = "ml")]
    PredictQuality(PredictQualityArgs),
    /// Tune solver hyperparameters for an instance
    #[cfg(feature = "ml")]
    TuneHyperparams(TuneHyperparamsArgs),
    /// Parse a natural-language routing query into JSON
    #[cfg(feature = "ml")]
    ParseQuery(ParseQueryArgs),
    /// Generate node/edge embeddings for a road network graph
    #[cfg(feature = "ml")]
    GraphEmbed(GraphEmbedArgs),
}

#[derive(clap::Args, Serialize, Deserialize)]
struct EmbedArgs {
    /// Text to embed (can be specified multiple times)
    #[arg(short, long, action = clap::ArgAction::Append)]
    text: Vec<String>,
}

// ── Graph Embed ──────────────────────────────────────────────────────

#[cfg(feature = "ml")]
#[derive(clap::Args, Serialize, Deserialize)]
#[serde(default)]
struct GraphEmbedArgs {
    /// Input .rmp cache file
    #[arg(short, long)]
    input: String,

    /// Embedding method
    #[arg(short, long, value_enum, default_value = "node2vec")]
    method: crate::core::ml::node_embed::EmbedMethod,

    /// Embedding dimension (default: 64)
    #[arg(short, long, default_value = "64")]
    dim: usize,

    /// Walk length for node2vec (default: 30)
    #[arg(long, default_value = "30")]
    walk_length: usize,

    /// Number of walks per node for node2vec (default: 10)
    #[arg(long, default_value = "10")]
    num_walks: usize,

    /// Return parameter p for node2vec (default: 1.0)
    #[arg(long, default_value = "1.0")]
    p: f64,

    /// In-out parameter q for node2vec (default: 1.0)
    #[arg(long, default_value = "1.0")]
    q: f64,

    /// Skip-gram window size (default: 5)
    #[arg(long, default_value = "5")]
    window: usize,

    /// Number of negative samples (default: 5)
    #[arg(long, default_value = "5")]
    negative_samples: usize,

    /// Learning rate (default: 0.025)
    #[arg(long, default_value = "0.025")]
    lr: f64,

    /// Number of training epochs for LINE (default: 5)
    #[arg(long, default_value = "5")]
    epochs: usize,

    /// Include edge embeddings (derived via Hadamard product)
    #[arg(long, default_value = "false")]
    include_edges: bool,

    /// Output file path (JSON). If omitted, prints to stdout.
    #[arg(short, long)]
    output: Option<String>,
}

#[cfg(feature = "ml")]
impl Default for GraphEmbedArgs {
    fn default() -> Self {
        Self {
            input: String::new(),
            method: crate::core::ml::node_embed::EmbedMethod::Node2Vec,
            dim: 64,
            walk_length: 30,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            window: 5,
            negative_samples: 5,
            lr: 0.025,
            epochs: 5,
            include_edges: false,
        }
    }
}

// ── Elevation ─────────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
struct ElevationArgs {
    /// Path to DEM GeoTIFF file
    #[arg(short, long)]
    dem: String,

    #[command(subcommand)]
    command: ElevationCommand,
}

#[derive(Subcommand, Serialize, Deserialize)]
enum ElevationCommand {
    /// Query elevation at a single LON,LAT point
    Point(PointElevationArgs),
    /// Query elevation at multiple points from a JSON file
    Points(PointsElevationArgs),
    /// Get elevation profile along a route
    Profile(ProfileElevationArgs),
    /// Get elevation statistics for a bounding box
    Stats(StatsElevationArgs),
    /// Show DEM file metadata (size, bbox, nodata)
    Info(InfoElevationArgs),
    /// Calculate fuel consumption from a route profile
    Fuel(FuelElevationArgs),
}

#[derive(clap::Args, Serialize, Deserialize)]
struct PointElevationArgs {
    /// Longitude
    #[arg(long, allow_hyphen_values = true)]
    lon: f64,
    /// Latitude
    #[arg(long, allow_hyphen_values = true)]
    lat: f64,
}

#[derive(clap::Args, Serialize, Deserialize)]
struct PointsElevationArgs {
    /// Path to JSON file with [[lon,lat],...] array (or '-' for stdin)
    #[arg(short, long, default_value = "-")]
    input: String,
}

#[derive(clap::Args, Serialize, Deserialize)]
struct ProfileElevationArgs {
    /// Path to JSON file with [[lon,lat],...] route (or '-' for stdin)
    #[arg(short, long, default_value = "-")]
    input: String,
    /// Distance between samples in meters
    #[arg(short = 's', long, default_value_t = 10.0)]
    interval: f64,
}

#[derive(clap::Args, Serialize, Deserialize)]
struct StatsElevationArgs {
    /// Bounding box: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT
    #[arg(long, allow_hyphen_values = true)]
    bbox: String,
    /// Pixel skip for sampling (1 = every pixel, 10 = every 10th)
    #[arg(long, default_value_t = 10)]
    step: usize,
}

#[derive(clap::Args, Serialize, Deserialize)]
struct InfoElevationArgs;

#[derive(clap::Args, Serialize, Deserialize)]
struct FuelElevationArgs {
    /// Path to JSON file with [[lon,lat],...] route (or '-' for stdin)
    #[arg(short, long, default_value = "-")]
    input: String,
    /// Distance between samples in meters
    #[arg(short = 's', long, default_value_t = 10.0)]
    interval: f64,
    /// Base fuel consumption in L/km
    #[arg(long, default_value_t = 0.35)]
    base_consumption: f64,
}

// ── Predict Solver ─────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
#[cfg(feature = "ml")]
struct PredictSolverArgs {
    /// Path to JSON file with stops array (first = depot) or '-' for stdin
    #[arg(short, long, default_value = "-")]
    input: String,
    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

// ── Predict Quality ────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
#[cfg(feature = "ml")]
struct PredictQualityArgs {
    /// Path to JSON file with stops array (first = depot) or '-' for stdin
    #[arg(short, long, default_value = "-")]
    input: String,
    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

// ── Tune Hyperparams ─────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
#[cfg(feature = "ml")]
struct TuneHyperparamsArgs {
    /// Path to JSON file with stops array (first = depot) or '-' for stdin
    #[arg(short, long, default_value = "-")]
    input: String,
    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

// ── Parse Query ────────────────────────────────────────────────────────

#[cfg(feature = "ml")]
#[derive(clap::Args, Serialize, Deserialize)]
struct ParseQueryArgs {
    /// Natural language routing query
    #[arg(short, long)]
    query: String,
    /// Output results as JSON
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args)]
struct ServeArgs {}

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
    #[cfg(feature = "extract")]
    Extract(ExtractArgs),
    Compile(CompileArgs),
    Clean(CleanArgs),
    Optimize(OptimizeArgs),
    Vrp(VrpArgs),
    #[cfg(feature = "extract")]
    Pipeline(PipelineArgs),
    #[cfg(feature = "ml")]
    Embed(EmbedArgs),
    #[cfg(feature = "ml")]
    GraphEmbed(GraphEmbedArgs),
    #[cfg(feature = "extract")]
    Elevation(ElevationArgs),
}

// ── VRP ───────────────────────────────────────────────────────────────

// ── Custom Serde Deserializers for Agent Payloads ──────────────────────

fn deserialize_depot_opt<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DepotInput {
        String(String),
        Array([f64; 2]),
    }

    let opt = Option::<DepotInput>::deserialize(deserializer)?;
    Ok(opt.map(|input| match input {
        DepotInput::String(s) => s,
        DepotInput::Array([lat, lon]) => format!("{},{}", lat, lon),
    }))
}

fn deserialize_depots_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DepotInput {
        String(String),
        Array([f64; 2]),
    }

    let inputs = Vec::<DepotInput>::deserialize(deserializer).unwrap_or_default();
    Ok(inputs
        .into_iter()
        .map(|input| match input {
            DepotInput::String(s) => s,
            DepotInput::Array([lat, lon]) => format!("{},{}", lat, lon),
        })
        .collect())
}

fn deserialize_bbox<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BboxInput {
        String(String),
        Array([f64; 4]),
    }

    let input = BboxInput::deserialize(deserializer)?;
    Ok(match input {
        BboxInput::String(s) => s,
        BboxInput::Array([min_lon, min_lat, max_lon, max_lat]) => {
            format!("{},{},{},{}", min_lon, min_lat, max_lon, max_lat)
        }
    })
}

fn default_output_dir() -> String {
    "routes/".to_string()
}

fn default_vehicles() -> usize {
    1
}

fn default_vrp_algo() -> VrpAlgorithm {
    VrpAlgorithm::Greedy
}

fn default_source() -> String {
    "overture".to_string()
}

fn default_extract_output() -> String {
    "extract-output.geojson".to_string()
}

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
#[serde(rename_all = "snake_case")]
struct VrpArgs {
    /// Input .rmp cache file
    #[arg(short, long)]
    input: String,

    /// Output directory for agent GPX routes
    #[arg(short, long, default_value = "routes/")]
    #[serde(default = "default_output_dir")]
    output_dir: String,

    /// Number of vehicles/agents
    #[arg(short, long, default_value_t = 1)]
    #[serde(default = "default_vehicles", alias = "num_vehicles")]
    vehicles: usize,

    /// Algorithm to use for VRP
    #[arg(short, long, value_enum, default_value_t = VrpAlgorithm::Greedy)]
    #[serde(default = "default_vrp_algo")]
    algo: VrpAlgorithm,

    /// Vehicle capacity (if applicable)
    #[arg(long)]
    #[serde(default)]
    capacity: Option<f64>,

    /// Depot coordinates: LAT,LON (can be specified multiple times)
    #[arg(long, action = clap::ArgAction::Append)]
    #[serde(default, deserialize_with = "deserialize_depots_vec")]
    depot: Vec<String>,

    /// Path to a CSV file containing coordinates to visit (columns: lat,lon [, label, demand, type])
    #[arg(long)]
    #[serde(default)]
    coordinates: Option<String>,

    /// Generate Google Maps URL links for the routes
    #[arg(long, default_value_t = false)]
    #[serde(default)]
    google_maps: bool,

    /// Public base URL for OsmAnd GPX import links (e.g. https://pub-xxx.r2.dev)
    #[arg(long)]
    #[serde(default)]
    osmand_base_url: Option<String>,
}

// ── Extract ───────────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ExtractArgs {
    /// Bounding box: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT
    #[arg(long, allow_hyphen_values = true)]
    #[serde(deserialize_with = "deserialize_bbox")]
    bbox: String,

    /// Data source: overture (default)
    #[arg(long, default_value = "overture")]
    #[serde(default = "default_source")]
    source: String,

    /// Road classes to include (comma-separated, default: all vehicle)
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    road_classes: Vec<String>,

    /// Output GeoJSON file path
    #[arg(short, long, default_value = "extract-output.geojson")]
    #[serde(default = "default_extract_output")]
    output: String,

    /// Path to local OSM PBF file (required for source=osm)
    #[arg(long)]
    #[serde(default)]
    pbf_path: Option<String>,
}

// ── Compile ───────────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct CompileArgs {
    /// Input GeoJSON file
    #[arg(short, long)]
    input: String,

    /// Output .rmp file
    #[arg(short, long)]
    output: String,

    /// Run cleaning before compilation (with default clean options)
    #[arg(long)]
    #[serde(default)]
    clean: bool,

    /// Road classes to include in the .rmp (comma-separated allowlist; empty = include all)
    /// e.g. --road-classes residential,tertiary,secondary,unclassified
    #[arg(long, value_delimiter = ',')]
    #[serde(default)]
    road_classes: Vec<String>,

    /// Prune leaf spurs (degree-1 edges from boundary clipping)
    #[arg(long)]
    #[serde(default)]
    prune_spurs: bool,

    /// Prune disconnected subgraphs
    #[arg(long)]
    #[serde(default)]
    prune_disconnected: bool,
}

// ── Clean ─────────────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct CleanArgs {
    /// Input GeoJSON file
    #[arg(short, long)]
    input: String,

    /// Output cleaned GeoJSON file
    #[arg(short, long)]
    output: String,

    /// Minimum edge length in meters (default: 0.1)
    #[arg(long)]
    #[serde(default)]
    min_length_m: Option<f64>,

    /// Node snapping distance in meters (default: 1.0)
    #[arg(long)]
    #[serde(default)]
    node_snap_m: Option<f64>,

    /// Keep only the N largest connected components (default: 1)
    #[arg(long)]
    #[serde(default)]
    max_components: Option<usize>,

    /// Geometry simplification tolerance in meters (default: 0.0)
    #[arg(long)]
    #[serde(default)]
    simplify_tolerance_m: Option<f64>,

    /// Skip edge deduplication
    #[arg(long)]
    #[serde(default)]
    no_dedupe: bool,

    /// Skip isolated node removal
    #[arg(long)]
    #[serde(default)]
    no_remove_isolates: bool,
}

// ── Optimize ──────────────────────────────────────────────────────────

fn default_oneway() -> String {
    "respect".to_string()
}

fn default_mode() -> String {
    "cpp".to_string()
}

fn default_cpp_engine() -> String {
    "internal".to_string()
}

fn default_left_penalty() -> f64 {
    1.0
}

fn default_right_penalty() -> f64 {
    0.0
}

fn default_uturn_penalty() -> f64 {
    5.0
}

fn default_solver() -> String {
    "default".to_string()
}

fn default_pipeline_output_dir() -> String {
    ".".to_string()
}

#[derive(clap::Args, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct OptimizeArgs {
    /// Input .rmp cache file
    #[arg(short, long)]
    input: String,

    /// Output route GPX file
    #[arg(short, long)]
    #[serde(default)]
    output: Option<String>,

    /// Depot coordinates: LAT,LON
    #[arg(long)]
    #[serde(default, deserialize_with = "deserialize_depot_opt")]
    depot: Option<String>,

    /// Oneway mode: respect, ignore, reverse (default: respect)
    #[arg(long, default_value = "respect")]
    #[serde(default = "default_oneway")]
    oneway: String,

    /// Solver mode: cpp (edge coverage) or vrp (stop visits) (default: cpp)
    #[arg(short, long, default_value = "cpp")]
    #[serde(default = "default_mode")]
    mode: String,

    /// C++ engine: internal or external-rust-optimizer (default: internal)
    #[arg(long, default_value = "internal")]
    #[serde(default = "default_cpp_engine")]
    cpp_engine: String,

    /// Left turn penalty (default: 1.0)
    #[arg(long, default_value_t = 1.0)]
    #[serde(default = "default_left_penalty")]
    left_penalty: f64,

    /// Right turn penalty (default: 0.0)
    #[arg(long, default_value_t = 0.0)]
    #[serde(default = "default_right_penalty")]
    right_penalty: f64,

    /// U-turn penalty (default: 5.0)
    #[arg(long, default_value_t = 5.0)]
    #[serde(default = "default_uturn_penalty")]
    uturn_penalty: f64,

    /// Number of vehicles (for VRP mode only)
    #[arg(long, default_value_t = 1)]
    vehicles: usize,

    /// Solver algorithm for VRP mode (clarke_wright, sweep, two_opt, or_opt, default)
    #[arg(long, default_value = "default")]
    #[serde(default = "default_solver", alias = "solver_id")]
    solver: String,

    /// Generate Google Maps URL links for the routes
    #[arg(long, default_value_t = false)]
    #[serde(default)]
    google_maps: bool,

    /// Public base URL for OsmAnd GPX import links (e.g. https://pub-xxx.r2.dev)
    #[arg(long)]
    #[serde(default)]
    osmand_base_url: Option<String>,
}

// ── Pipeline ──────────────────────────────────────────────────────────

#[derive(clap::Args, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct PipelineArgs {
    /// Bounding box: MIN_LON,MIN_LAT,MAX_LON,MAX_LAT
    #[arg(long, allow_hyphen_values = true)]
    #[serde(deserialize_with = "deserialize_bbox")]
    bbox: String,

    /// Data source: overture (default)
    #[arg(long, default_value = "overture")]
    #[serde(default = "default_source")]
    source: String,

    /// Output directory for all intermediate and final files
    #[arg(short, long, default_value = ".")]
    #[serde(default = "default_pipeline_output_dir")]
    output_dir: String,

    /// Depot coordinates: LAT,LON
    #[arg(long)]
    #[serde(default, deserialize_with = "deserialize_depot_opt")]
    depot: Option<String>,

    /// Prune leaf spurs (degree-1 edges from boundary clipping)
    #[arg(long)]
    #[serde(default)]
    pub prune_spurs: bool,

    /// Prune disconnected subgraphs during compilation
    #[arg(long)]
    #[serde(default)]
    pub prune_disconnected: bool,

    /// Path to local OSM PBF file (optional)
    #[arg(long)]
    #[serde(default)]
    pub pbf_path: Option<String>,
}

#[cfg(feature = "extract")]
#[derive(Serialize)]
struct PipelineResult {
    extract: crate::core::extract::ExtractResult,
    clean: crate::core::clean::CleanStats,
    compile: crate::core::compile::CompileResult,
    optimize: crate::core::optimize::OptimizeResult,
}

// ── Parsing helpers ───────────────────────────────────────────────────

#[cfg(feature = "extract")]
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

#[cfg(feature = "extract")]
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

#[cfg(feature = "extract")]
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
        .map(|v| {
            v.parse::<f64>()
                .context(format!("Invalid depot value: {v}"))
        })
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

fn parse_cpp_engine(s: &str) -> Result<CppEngine> {
    match s.to_lowercase().as_str() {
        "internal" | "" => Ok(CppEngine::Internal),
        "external-rust-optimizer" | "external" | "rust-optimizer" => Ok(CppEngine::ExternalRustOptimizer),
        other => anyhow::bail!("Unknown C++ engine: {other} (internal|external-rust-optimizer)"),
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

#[cfg(feature = "extract")]
async fn run_extract_cmd(args: ExtractArgs, json: bool) -> Result<()> {
    let (min_lon, min_lat, max_lon, max_lat) = parse_bbox(&args.bbox)?;
    let road_classes = parse_road_classes(&args.road_classes)?;
    let source = parse_source(&args.source)?;

    if !json {
        tracing::info!("Starting extraction...");
        tracing::info!("Source: {:?}", source);
        tracing::info!("Bounding box: [{min_lon:.4}, {min_lat:.4}, {max_lon:.4}, {max_lat:.4}]");
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
        pbf_path: args.pbf_path.clone(),
        database_url: None,
        table_name: None,
        r2_bucket: None,
        r2_access_key_id: None,
        r2_secret_access_key: None,
        r2_endpoint: None,
    };


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
        road_classes: args.road_classes,
        clean_options,
        prune_disconnected: args.prune_disconnected,
        prune_spurs: args.prune_spurs,
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
        tracing::info!(
            "Features: {} -> {}",
            stats.input_features,
            stats.output_features
        );
        for w in &warnings {
            tracing::warn!("Warning: {w}");
        }
    }

    Ok(())
}

async fn run_optimize_cmd(args: OptimizeArgs, json: bool) -> Result<()> {
    let depot = args.depot.as_deref().map(parse_depot).transpose()?;
    let oneway_mode = parse_oneway(&args.oneway)?;
    let mode = match args.mode.to_lowercase().as_str() {
        "vrp" => SolverMode::Vrp,
        _ => SolverMode::Cpp,
    };
    let cpp_engine = parse_cpp_engine(&args.cpp_engine)?;

    if !json {
        tracing::info!(
            "Optimizing route from {} (mode: {}, engine: {})",
            args.input,
            if mode == SolverMode::Vrp {
                "VRP"
            } else {
                "CPP"
            },
            match cpp_engine {
                CppEngine::Internal => "internal",
                CppEngine::ExternalRustOptimizer => "external-rust-optimizer",
            }
        );
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
        mode,
        cpp_engine,
        num_vehicles: args.vehicles,
        solver_id: args.solver,
        coordinates: None,
    };

    let result = crate::core::optimize::run_optimize(&req).await?;

    if json {
        output_json(&result)?;
    } else {
        tracing::info!("Optimization complete!");
        if result.is_partial {
            tracing::warn!("WARNING: Route is PARTIAL (Disconnected Components).");
            tracing::warn!("Found {} unreachable edges from start node.", result.unreachable_edges);
        }
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

        if args.google_maps {
            if result.routes.is_empty() {
                tracing::warn!("No routes available for Google Maps URL");
            } else {
                for (i, route) in result.routes.iter().enumerate() {
                    // For long routes (CPP), we might need to sample points
                    // Google Maps limit is ~20 points for /dir/
                    let max_points = 20;
                    let sampled_points = if route.len() > max_points {
                        let step = route.len() / max_points;
                        route
                            .iter()
                            .step_by(step)
                            .take(max_points)
                            .collect::<Vec<_>>()
                    } else {
                        route.iter().collect::<Vec<_>>()
                    };

                    let mut url = "https://www.google.com/maps/dir/".to_string();
                    for stop in sampled_points {
                        url.push_str(&format!("{:.6},{:.6}/", stop.lat, stop.lon));
                    }
                    if result.routes.len() > 1 {
                        tracing::info!("Route {} Google Maps (Sampled): {}", i + 1, url);
                    } else {
                        tracing::info!("Google Maps (Sampled): {}", url);
                    }
                }
            }
        }

        if let Some(ref base_url) = args.osmand_base_url {
            if let Some(ref path) = args.output {
                let filename = std::path::Path::new(path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("route.gpx");
                let gpx_url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
                let osmand_link = crate::core::vrp::utils::generate_osmand_import_url(&gpx_url);
                tracing::info!("OsmAnd Link: {}", osmand_link);
            }
        }
    }

    Ok(())
}

impl VrpAlgorithm {
    fn to_solver_id(&self) -> &'static str {
        match self {
            VrpAlgorithm::Greedy => "default",
            VrpAlgorithm::Savings => "clarke_wright",
            VrpAlgorithm::LocalSearch => "two_opt",
            VrpAlgorithm::SimulatedAnnealing => "or_opt",
        }
    }
}

async fn run_vrp_cmd(args: VrpArgs, _json: bool) -> Result<()> {
    tracing::info!("VRP solving requested for {}", args.input);
    tracing::info!("Algorithm: {:?}", args.algo);
    tracing::info!(
        "Vehicles: {}, Depots: {}, Coordinates CSV: {:?}",
        args.vehicles,
        args.depot.len(),
        args.coordinates
    );

    // Parse all depots
    let depots: Vec<(f64, f64)> = args
        .depot
        .iter()
        .map(|s| parse_depot(s))
        .collect::<Result<Vec<_>>>()?;

    if depots.is_empty() {
        anyhow::bail!("At least one depot must be specified via --depot");
    }

    let mut stops = Vec::new();

    // 1. Add depot
    stops.push(VRPSolverStop {
        lat: depots[0].0,
        lon: depots[0].1,
        label: "Depot".into(),
        demand: Some(0.0),
        arrival_time: None,
    });

    // 2. Load stops from coordinates CSV
    if let Some(csv_path) = args.coordinates {
        let (csv_stops, _depot_indices) = crate::core::vrp::utils::parse_csv_stops(&csv_path)
            .map_err(|e| anyhow::anyhow!("CSV parse error: {}", e))?;
        stops.extend(csv_stops);
    } else {
        anyhow::bail!("--coordinates CSV file is required for VRP to define the delivery stops");
    }

    let solver_id = args.algo.to_solver_id();
    let capacity = args.capacity.unwrap_or(100.0);

    let matrix = crate::core::vrp::utils::build_haversine_matrix(&stops, 40.0);

    let vrp_input = VRPSolverInput {
        locations: stops,
        num_vehicles: args.vehicles,
        vehicle_capacity: capacity,
        objective: VrpObjective::MinDistance,
        matrix: Some(matrix),
        service_time_secs: Some(30.0),
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };

    let output = crate::core::vrp::registry::solve_with(solver_id, &vrp_input)
        .await
        .map_err(|e| anyhow::anyhow!("VRP Solver error: {}", e))?;

    std::fs::create_dir_all(&args.output_dir)?;

    if let Some(routes) = output.routes {
        for (i, route) in routes.iter().enumerate() {
            let path = format!("{}/vehicle_{}.gpx", args.output_dir, i + 1);
            crate::core::optimize::write_gpx_multi(&path, std::slice::from_ref(route))?;
            tracing::info!("Wrote route to {}", path);

            if args.google_maps {
                // Google Maps URL limit is ~20 points for /dir/
                // Format: https://www.google.com/maps/dir/lat1,lon1/lat2,lon2/...
                let chunks = route.chunks(20);
                for (chunk_idx, chunk) in chunks.enumerate() {
                    let mut url = "https://www.google.com/maps/dir/".to_string();
                    for stop in chunk {
                        url.push_str(&format!("{:.6},{:.6}/", stop.lat, stop.lon));
                    }
                    if route.chunks(20).count() > 1 {
                        tracing::info!(
                            "Vehicle {} Google Maps (Part {}): {}",
                            i + 1,
                            chunk_idx + 1,
                            url
                        );
                    } else {
                        tracing::info!("Vehicle {} Google Maps: {}", i + 1, url);
                    }
                }
            }

            if let Some(ref base_url) = args.osmand_base_url {
                let filename = format!("vehicle_{}.gpx", i + 1);
                let gpx_url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
                let osmand_link = crate::core::vrp::utils::generate_osmand_import_url(&gpx_url);
                tracing::info!("Vehicle {} OsmAnd Link: {}", i + 1, osmand_link);
            }
        }
    } else {
        tracing::warn!("No routes produced by VRP solver.");
    }

    Ok(())
}

#[cfg(feature = "extract")]
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
        pbf_path: args.pbf_path.clone(),
        database_url: None,
        table_name: None,
        r2_bucket: None,
        r2_access_key_id: None,
        r2_secret_access_key: None,
        r2_endpoint: None,
    };

    let extract_result = crate::core::extract::run_extract(&extract_req)
        .await
        .context("Pipeline failed at stage 'extract'")?;

    // Stage 2: Clean
    if !json {
        tracing::info!("=== Pipeline Stage 2: Clean ===");
    }
    let input_data = std::fs::read_to_string(&extract_path)
        .context("Pipeline failed reading extracted GeoJSON")?;
    let geojson: geojson::FeatureCollection =
        serde_json::from_str(&input_data).context("Pipeline failed parsing extracted GeoJSON")?;
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
        prune_disconnected: args.prune_disconnected,
        prune_spurs: args.prune_spurs,
    };
    let compile_result = crate::core::compile::run_compile(&compile_req)
        .context("Pipeline failed at stage 'compile'")?;

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
        mode: SolverMode::Cpp,
        cpp_engine: CppEngine::Internal,
        num_vehicles: 1,
        solver_id: "clarke_wright".to_string(),
        coordinates: None,
    };
    let optimize_result = crate::core::optimize::run_optimize(&optimize_req)
        .await
        .context("Pipeline failed at stage 'optimize'")?;

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

    let task: AgentTask =
        serde_json::from_reader(input).context("Failed to parse agent task JSON")?;

    match task {
        #[cfg(feature = "extract")]
        AgentTask::Extract(a) => run_extract_cmd(a, json).await,
        AgentTask::Compile(a) => run_compile_cmd(a, json),
        AgentTask::Clean(a) => run_clean_cmd(a, json),
        AgentTask::Optimize(a) => run_optimize_cmd(a, json).await,
        AgentTask::Vrp(a) => run_vrp_cmd(a, json).await,
        #[cfg(feature = "extract")]
        AgentTask::Pipeline(a) => run_pipeline_cmd(a, json).await,
        #[cfg(feature = "ml")]
        AgentTask::Embed(a) => run_embed_cmd(a, json).await,
        #[cfg(feature = "ml")]
        AgentTask::GraphEmbed(a) => run_graph_embed_cmd(a, json),
        #[cfg(feature = "extract")]
        AgentTask::Elevation(a) => run_elevation_cmd(a, json),
    }
}
async fn run_serve_cmd(_args: ServeArgs) -> Result<()> {
    use std::io::BufRead;

    #[derive(Serialize)]
    struct ErrorResponse {
        error: String,
    }

    tracing::info!("Headless engine started. Listening on stdin for JSON-RPC/STDIO tasks...");

    let stdin = std::io::stdin();
    for line_result in stdin.lock().lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Failed to read stdin: {}", e);
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<AgentTask>(&line) {
            Ok(task) => {
                let res = match task {
                    #[cfg(feature = "extract")]
                    AgentTask::Extract(a) => run_extract_cmd(a, true).await,
                    AgentTask::Compile(a) => run_compile_cmd(a, true),
                    AgentTask::Clean(a) => run_clean_cmd(a, true),
                    AgentTask::Optimize(a) => run_optimize_cmd(a, true).await,
                    AgentTask::Vrp(a) => run_vrp_cmd(a, true).await,
                    #[cfg(feature = "extract")]
                    AgentTask::Pipeline(a) => run_pipeline_cmd(a, true).await,
                    #[cfg(feature = "ml")]
                    AgentTask::Embed(a) => run_embed_cmd(a, true).await,
                    #[cfg(feature = "ml")]
                    AgentTask::GraphEmbed(a) => run_graph_embed_cmd(a, true),
                    #[cfg(feature = "extract")]
                    AgentTask::Elevation(a) => run_elevation_cmd(a, true),
                };
                if let Err(e) = res {
                    let _ = output_json(&ErrorResponse {
                        error: format!("Task failed: {}", e),
                    });
                }
            }
            Err(e) => {
                let _ = output_json(&ErrorResponse {
                    error: format!("Failed to parse agent task JSON: {}", e),
                });
            }
        }
    }

    Ok(())
}

#[cfg(feature = "ml")]
async fn run_embed_cmd(args: EmbedArgs, json: bool) -> Result<()> {
    if !json {
        tracing::info!("Generating embeddings for {} texts...", args.text.len());
    }

    let embeddings = crate::core::embed::run_embed(args.text)?;

    if json {
        output_json(&embeddings)?;
    } else {
        for (i, emb) in embeddings.iter().enumerate() {
            tracing::info!(
                "Embedding {}: dimension {}, first few values: {:?}",
                i,
                emb.len(),
                &emb[..5.min(emb.len())]
            );
        }
    }

    Ok(())
}

#[cfg(feature = "ml")]
fn run_graph_embed_cmd(args: GraphEmbedArgs, _json: bool) -> Result<()> {
    use crate::core::ml::node_embed::{EmbedConfig, EmbedMethod, embed_graph};
    use crate::core::optimize::read_rmp_file;

    let file_data = std::fs::read(&args.input)?;
    let (nodes, edges) = read_rmp_file(&file_data)
        .map_err(|e| anyhow::anyhow!("Failed to parse .rmp: {}", e))?;

    let method = match args.method {
        crate::core::ml::node_embed::EmbedMethod::Node2Vec => EmbedMethod::Node2Vec,
        crate::core::ml::node_embed::EmbedMethod::Line => EmbedMethod::Line,
        crate::core::ml::node_embed::EmbedMethod::FastRp => EmbedMethod::FastRp,
        crate::core::ml::node_embed::EmbedMethod::Spatial => EmbedMethod::Spatial,
    };

    let config = EmbedConfig {
        method,
        dimensions: args.dim,
        walk_length: args.walk_length,
        num_walks: args.num_walks,
        p: args.p,
        q: args.q,
        window: args.window,
        negative_samples: args.negative_samples,
        lr: args.lr,
        epochs: args.epochs,
        threads: 1,
        include_edges: args.include_edges,
    };

    println!("Running {} embedding...", args.input);
    let output = embed_graph(&nodes, &edges, &config)
        .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?;

    let output_json = serde_json::to_string_pretty(&output)?;
    if let Some(out_path) = args.output {
        std::fs::write(&out_path, output_json)?;
        println!("Wrote embeddings to {}", out_path);
    } else {
        println!("{}", output_json);
    }

    Ok(())
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
        eprintln!("Available {:?}:", args.resource);
        for f in files {
            eprintln!("  - {f}");
        }
    }
    Ok(())
}

// ── Elevation handler ─────────────────────────────────────────────────

#[cfg(feature = "extract")]
fn run_elevation_cmd(args: ElevationArgs, json: bool) -> Result<()> {
    let dem_path = std::path::Path::new(&args.dem);
    if !json {
        tracing::info!("Opening DEM: {}", dem_path.display());
    }
    let dem = LocalDem::open(dem_path)?;

    match args.command {
        ElevationCommand::Point(p) => {
            let result = dem.get_elevation(p.lon, p.lat)?;
            if json {
                #[derive(Serialize)]
                struct Out {
                    lon: f64,
                    lat: f64,
                    elevation: Option<f64>,
                    has_coverage: bool,
                }
                output_json(&Out {
                    lon: p.lon,
                    lat: p.lat,
                    elevation: result,
                    has_coverage: result.is_some(),
                })?;
            } else {
                match result {
                    Some(e) => println!("Elevation at ({}, {}): {:.2} m", p.lon, p.lat, e),
                    None => println!("No coverage at ({}, {})", p.lon, p.lat),
                }
            }
        }

        ElevationCommand::Points(p) => {
            let points: Vec<(f64, f64)> = read_json_input(&p.input)?;
            let elevations = dem.get_elevations(&points)?;
            if json {
                #[derive(Serialize)]
                struct Out {
                    points: Vec<(f64, f64)>,
                    elevations: Vec<Option<f64>>,
                }
                output_json(&Out { points, elevations })?;
            } else {
                for (i, (pt, elev)) in points.iter().zip(elevations.iter()).enumerate() {
                    match elev {
                        Some(e) => println!("  [{}] ({}, {}): {:.2} m", i, pt.0, pt.1, e),
                        None => println!("  [{}] ({}, {}): no coverage", i, pt.0, pt.1),
                    }
                }
            }
        }

        ElevationCommand::Profile(p) => {
            let route: Vec<(f64, f64)> = read_json_input(&p.input)?;
            let profile = dem.route_profile(&route, p.interval)?;
            if json {
                output_json(&profile)?;
            } else {
                println!("Route Elevation Profile:");
                println!("  Distance: {:.2} km", profile.distance_km);
                println!(
                    "  Elevation: {:.1} - {:.1} m (avg {:.1} m)",
                    profile.min_elevation, profile.max_elevation, profile.avg_elevation
                );
                println!(
                    "  Ascent: {:.1} m, Descent: {:.1} m",
                    profile.total_ascent, profile.total_descent
                );
                println!("  Sample points: {}", profile.points.len());
            }
        }

        ElevationCommand::Stats(s) => {
            let (min_lon, min_lat, max_lon, max_lat) = parse_bbox(&s.bbox)?;
            let bbox = crate::core::elevation::BBox {
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            };
            let stats = dem.bbox_stats(bbox, s.step)?;
            if json {
                output_json(&stats)?;
            } else {
                println!("Elevation Stats:");
                println!(
                    "  Range: {:.1} - {:.1} m (avg {:.1} m)",
                    stats.min_elevation, stats.max_elevation, stats.avg_elevation
                );
                println!("  Coverage: {:.1}%", stats.coverage_percent);
                println!("  Valid pixels: {}", stats.pixel_count);
            }
        }

        ElevationCommand::Info(_) => {
            let info = dem.info();
            if json {
                output_json(&info)?;
            } else {
                println!("DEM Info:");
                println!("  Size: {} x {} pixels", info.width, info.height);
                println!(
                    "  BBox: [{:.4}, {:.4}, {:.4}, {:.4}]",
                    info.bbox.min_lon, info.bbox.min_lat, info.bbox.max_lon, info.bbox.max_lat
                );
                println!(
                    "  Pixel size: {:.6} x {:.6}",
                    info.pixel_size_x, info.pixel_size_y
                );
                println!("  NoData: {:?}", info.nodata);
            }
        }

        ElevationCommand::Fuel(f) => {
            let route: Vec<(f64, f64)> = read_json_input(&f.input)?;
            let profile = dem.route_profile(&route, f.interval)?;
            let fuel = FuelCalculator::calculate(&profile, f.base_consumption);
            if json {
                output_json(&fuel)?;
            } else {
                println!("Fuel Consumption:");
                println!("  Total: {:.2} L", fuel.total_fuel_l);
                println!("  Avg: {:.3} L/km", fuel.avg_consumption_l_per_km);
                println!("  Elevation penalty: {:.2} L", fuel.elevation_penalty_l);
                println!("  Elevation benefit: {:.2} L", fuel.elevation_benefit_l);
            }
        }
    }

    Ok(())
}

// ── New ML command handlers ────────────────────────────────────────────

#[cfg(feature = "ml")]
use crate::core::ml::automl::predict_hyperparams;
#[cfg(feature = "ml")]
use crate::core::ml::features::InstanceFeatures;
#[cfg(feature = "ml")]
use crate::core::ml::quality_predictor::predict_quality;
#[cfg(feature = "ml")]
use crate::core::ml::selector::{default_model_path, predict_solver};
#[cfg(feature = "ml")]
use crate::core::nlp::{parse_query, to_vrp_json};
use crate::core::vrp::types::{VRPSolverInput, VRPSolverStop, VrpObjective};

#[cfg(feature = "ml")]
fn parse_stops_json(path: &str) -> Result<Vec<VRPSolverStop>> {
    let stops: Vec<VRPSolverStop> = if path == "-" {
        serde_json::from_reader(std::io::stdin())?
    } else {
        let f = std::fs::File::open(path)?;
        serde_json::from_reader(f)?
    };
    Ok(stops)
}

#[cfg(feature = "ml")]
async fn run_predict_solver_cmd(args: PredictSolverArgs) -> Result<()> {
    let stops = parse_stops_json(&args.input)?;
    let input = VRPSolverInput {
        locations: stops,
        num_vehicles: 1,
        vehicle_capacity: 100.0,
        objective: VrpObjective::MinDistance,
        matrix: None,
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };
    let pred = predict_solver(&input, Some(&default_model_path()))?;
    if args.json {
        output_json(&pred)?;
    } else {
        println!(
            "Recommended solver: {} (confidence: {:.2}%)",
            pred.recommended,
            pred.confidence * 100.0
        );
        if let Some((ref id, score)) = pred.runner_up {
            println!("Runner-up: {} ({:.2}%)", id, score as f64 * 100.0);
        }
        for (id, score) in &pred.all_scores {
            println!("  {:20} {:.2}%", id, score * 100.0);
        }
    }
    Ok(())
}

#[cfg(feature = "ml")]
async fn run_predict_quality_cmd(args: PredictQualityArgs) -> Result<()> {
    let stops = parse_stops_json(&args.input)?;
    let input = VRPSolverInput {
        locations: stops,
        num_vehicles: 1,
        vehicle_capacity: 100.0,
        objective: VrpObjective::MinDistance,
        matrix: None,
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };
    let features = InstanceFeatures::from_input(&input);
    let pred = predict_quality(&features);
    if args.json {
        output_json(&pred)?;
    } else {
        println!("Predicted gap to optimal: {:.1}%", pred.predicted_gap_pct);
        println!(
            "Predicted tour length:    {:.1} km",
            pred.predicted_tour_length_km
        );
        println!("Confidence:               {:.2}", pred.confidence);
    }
    Ok(())
}

#[cfg(feature = "ml")]
async fn run_tune_hyperparams_cmd(args: TuneHyperparamsArgs) -> Result<()> {
    let stops = parse_stops_json(&args.input)?;
    let input = VRPSolverInput {
        locations: stops,
        num_vehicles: 1,
        vehicle_capacity: 100.0,
        objective: VrpObjective::MinDistance,
        matrix: None,
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };
    let features = InstanceFeatures::from_input(&input);
    let params = predict_hyperparams(&features);
    if args.json {
        output_json(&params)?;
    } else {
        println!("Suggested hyperparameters:");
        println!("  max_iterations:      {}", params.max_iterations);
        println!("  temperature:         {:.1}", params.temperature);
        println!("  tabu_tenure:         {}", params.tabu_tenure);
        println!("  cooling_rate:        {:.4}", params.cooling_rate);
        println!("  neighbourhood_radius: {}", params.neighbourhood_radius);
    }
    Ok(())
}

#[cfg(feature = "ml")]
fn run_parse_query_cmd(args: ParseQueryArgs) -> Result<()> {
    let parsed = parse_query(&args.query);
    let json = to_vrp_json(&parsed);
    if args.json {
        output_json(&json)?;
    } else {
        println!("Variant: {}", parsed.variant);
        println!("Config: {}", serde_json::to_string_pretty(&json)?);
        if !parsed.entities.is_empty() {
            println!("Entities:");
            for (k, v) in &parsed.entities {
                println!("  {} = {}", k, v);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "extract")]
fn read_json_input<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    let input: Box<dyn std::io::Read> = if path == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(path)?)
    };
    Ok(serde_json::from_reader(input)?)
}

// ── Entry point ───────────────────────────────────────────────────────

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    init_tracing();

    let result = match cli.command {
        #[cfg(feature = "extract")]
        Commands::Extract(args) => run_extract_cmd(args, cli.json).await,
        Commands::Compile(args) => run_compile_cmd(args, cli.json),
        Commands::Clean(args) => run_clean_cmd(args, cli.json),
        Commands::Optimize(args) => run_optimize_cmd(args, cli.json).await,
        Commands::Vrp(args) => run_vrp_cmd(args, cli.json).await,
        #[cfg(feature = "extract")]
        Commands::Pipeline(args) => run_pipeline_cmd(args, cli.json).await,
        Commands::List(args) => run_list_cmd(args, cli.json),
        Commands::Agent(args) => run_agent_cmd(args, cli.json).await,
        Commands::Serve(args) => run_serve_cmd(args).await,
        #[cfg(feature = "ml")]
        Commands::Embed(args) => run_embed_cmd(args, cli.json).await,
        #[cfg(feature = "extract")]
        Commands::Elevation(args) => run_elevation_cmd(args, cli.json),
        #[cfg(feature = "ml")]
        Commands::PredictSolver(args) => run_predict_solver_cmd(args).await,
        #[cfg(feature = "ml")]
        Commands::PredictQuality(args) => run_predict_quality_cmd(args).await,
        #[cfg(feature = "ml")]
        Commands::TuneHyperparams(args) => run_tune_hyperparams_cmd(args).await,
        #[cfg(feature = "ml")]
        Commands::ParseQuery(args) => run_parse_query_cmd(args),
        #[cfg(feature = "ml")]
        Commands::GraphEmbed(args) => run_graph_embed_cmd(args, cli.json),
    };

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bbox_valid() {
        let result = parse_bbox("-74.0,40.0,-73.0,41.0").unwrap();
        assert_eq!(result, (-74.0, 40.0, -73.0, 41.0));

        let result = parse_bbox("0,0,0,0").unwrap();
        assert_eq!(result, (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_parse_bbox_invalid_count() {
        assert!(parse_bbox("-74.0,40.0,-73.0").is_err());
        assert!(parse_bbox("-74.0,40.0,-73.0,41.0,42.0").is_err());
        assert!(parse_bbox("").is_err());
    }

    #[test]
    fn test_parse_bbox_invalid_format() {
        let err = parse_bbox("-74.0,abc,-73.0,41.0").unwrap_err();
        assert!(err.to_string().contains("Invalid bbox value"));
    }
}
