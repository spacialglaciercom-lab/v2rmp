use std::path::PathBuf;
use std::time::Instant;

use crate::core::clean::CleanOptions;
use crate::core::optimize::TurnPenalties;

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum View {
    Home,
    Extract,
    Compile,
    Clean,
    Optimize,
    Vrp,
    BrowseMaps,
    BrowseRoutes,
    FileBrowser,
    Help,
}

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: Option<u64>,
    pub modified: Option<std::time::SystemTime>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileFilter {
    GeoJson,
    Rmp,
    All,
}

impl FileFilter {
    pub fn matches(&self, path: &std::path::Path) -> bool {
        let ext = path.extension().and_then(|e| e.to_str());
        match self {
            FileFilter::GeoJson => matches!(ext, Some("geojson") | Some("json")),
            FileFilter::Rmp => matches!(ext, Some("rmp")),
            FileFilter::All => true,
        }
    }

    pub fn description(&self) -> &str {
        match self {
            FileFilter::GeoJson => "*.geojson, *.json",
            FileFilter::Rmp => "*.rmp",
            FileFilter::All => "*.*",
        }
    }
}

pub struct FileBrowser {
    pub current_path: PathBuf,
    pub entries: Vec<FileEntry>,
    pub selection: usize,
    pub filter: FileFilter,
    pub target_field: InputField,
    pub show_hidden: bool,
}

impl FileBrowser {
    pub fn new(target_field: InputField) -> Self {
        let current_path = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            current_path,
            entries: Vec::new(),
            selection: 0,
            filter: match target_field {
                InputField::InputFile | InputField::CleanInputFile => FileFilter::GeoJson,
                InputField::CacheFile => FileFilter::Rmp,
                _ => FileFilter::All,
            },
            target_field,
            show_hidden: false,
        }
    }

    pub fn refresh_entries(&mut self) -> anyhow::Result<()> {
        self.entries.clear();

        // Add parent directory entry if not at root
        if let Some(parent) = self.current_path.parent() {
            self.entries.push(FileEntry {
                name: "..".to_string(),
                path: parent.to_path_buf(),
                is_dir: true,
                size: None,
                modified: None,
            });
        }

        // Read directory entries
        let read_dir = std::fs::read_dir(&self.current_path)?;
        let mut entries: Vec<FileEntry> = Vec::new();

        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            // Skip hidden files unless show_hidden is true
            if !self.show_hidden && name.starts_with('.') {
                continue;
            }

            let metadata = entry.metadata().ok();
            let is_dir = metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false);
            let size = metadata
                .as_ref()
                .and_then(|m| if !is_dir { Some(m.len()) } else { None });
            let modified = metadata.as_ref().and_then(|m| m.modified().ok());

            // Include directories and files matching filter
            if is_dir || self.filter.matches(&path) {
                entries.push(FileEntry {
                    name,
                    path,
                    is_dir,
                    size,
                    modified,
                });
            }
        }

        // Sort: directories first, then by name
        entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        });

        self.entries.extend(entries);

        // Reset selection if out of bounds
        if self.selection >= self.entries.len() && !self.entries.is_empty() {
            self.selection = 0;
        }

        Ok(())
    }

    pub fn navigate_up(&mut self) {
        if !self.entries.is_empty() {
            self.selection = self.selection.saturating_sub(1);
        }
    }

    pub fn navigate_down(&mut self) {
        if !self.entries.is_empty() {
            self.selection = (self.selection + 1).min(self.entries.len() - 1);
        }
    }

    pub fn select_entry(&mut self) -> Option<PathBuf> {
        if self.entries.is_empty() {
            return None;
        }

        let entry = &self.entries[self.selection];

        if entry.is_dir {
            // Navigate into directory
            self.current_path = entry.path.clone();
            self.selection = 0;
            let _ = self.refresh_entries();
            None
        } else {
            // Return selected file path
            Some(entry.path.clone())
        }
    }

    pub fn go_parent(&mut self) {
        if let Some(parent) = self.current_path.parent() {
            self.current_path = parent.to_path_buf();
            self.selection = 0;
            let _ = self.refresh_entries();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataSource {
    Osm,
    Overture,
}

impl std::fmt::Display for DataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSource::Osm => write!(f, "OpenStreetMap (OSM)"),
            DataSource::Overture => write!(f, "Overture Maps"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub min_lon: f64,
    pub min_lat: f64,
    pub max_lon: f64,
    pub max_lat: f64,
}

impl std::fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.2},{:.2},{:.2},{:.2}",
            self.min_lat, self.min_lon, self.max_lat, self.max_lon
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Ready,
    Running { progress: u8, message: String },
    Done(String),
    Error(String),
}

impl Status {
    pub fn color(&self) -> ratatui::style::Color {
        match self {
            Status::Ready => ratatui::style::Color::DarkGray,
            Status::Running { .. } => ratatui::style::Color::Magenta,
            Status::Done(_) => ratatui::style::Color::Green,
            Status::Error(_) => ratatui::style::Color::Red,
        }
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Ready => write!(f, "Ready"),
            Status::Running { progress, message } => {
                write!(f, "{}% - {}", progress, message)
            }
            Status::Done(msg) => write!(f, "{}", msg),
            Status::Error(msg) => write!(f, "Error: {}", msg),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogLevel {
    Info,
    Success,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Success => write!(f, "SUCCESS"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

#[allow(dead_code)]
pub struct InputMode {
    pub active: bool,
    pub field: InputField,
    pub buffer: String,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum InputField {
    BoundingBox,
    InputFile,
    OutputFile,
    CacheFile,
    RouteFile,
    LeftTurnPenalty,
    RightTurnPenalty,
    UTurnPenalty,
    DepotCoordinates,
    NumVehicles,
    SolverId,
    CleanInputFile,
    CleanOutputFile,
    VrpInputFile,
    VrpOutputDir,
    VrpWaypointsFile,
    VrpAlgorithm,
    VrpCapacity,
    VrpDepot,
}

pub struct App {
    pub running: bool,
    pub current_view: View,
    pub workflow_selection: usize,
    pub log_entries: Vec<LogEntry>,
    pub log_scroll: usize,

    // Extract state
    pub data_source: DataSource,
    pub bounding_box: Option<BoundingBox>,
    pub extract_status: Status,

    // Compile state
    pub input_file: Option<String>,
    pub output_file: Option<String>,
    pub compile_status: Status,

    // Clean state
    pub clean_options: CleanOptions,
    pub clean_input_file: Option<String>,
    pub clean_output_file: Option<String>,
    pub clean_status: Status,
    pub clean_selection: usize,

    // Optimize state
    pub cache_file: Option<String>,
    pub route_file: Option<String>,
    pub turn_penalties: TurnPenalties,
    pub depot_coords: Option<(f64, f64)>,
    pub num_vehicles: usize,
    pub solver_id: String,
    pub optimize_status: Status,

    // VRP state
    pub vrp_input_file: Option<String>,
    pub vrp_output_dir: String,
    pub vrp_vehicles: usize,
    pub vrp_algo: String,
    pub vrp_capacity: Option<f64>,
    pub vrp_depots: Vec<String>,
    pub vrp_waypoints_file: Option<String>,
    pub vrp_status: Status,

    // Browse state
    pub cached_maps: Vec<String>,
    pub saved_routes: Vec<String>,
    pub browse_selection: usize,

    // File browser state
    pub file_browser: Option<FileBrowser>,

    // Input mode
    pub input_mode: InputMode,

    // Timing
    #[allow(dead_code)]
    pub start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            running: true,
            current_view: View::Home,
            workflow_selection: 0,
            log_entries: Vec::new(),
            log_scroll: 0,

            data_source: DataSource::Osm,
            bounding_box: None,
            extract_status: Status::Ready,

            input_file: None,
            output_file: None,
            compile_status: Status::Ready,

            clean_options: CleanOptions::default(),
            clean_input_file: None,
            clean_output_file: None,
            clean_status: Status::Ready,
            clean_selection: 0,

            cache_file: None,
            route_file: None,
            turn_penalties: TurnPenalties::default(),
            depot_coords: None,
            num_vehicles: 1,
            solver_id: "clarke_wright".to_string(),
            optimize_status: Status::Ready,

            vrp_input_file: None,
            vrp_output_dir: "routes/".to_string(),
            vrp_vehicles: 1,
            vrp_algo: "greedy".to_string(),
            vrp_capacity: Some(100.0),
            vrp_depots: Vec::new(),
            vrp_waypoints_file: None,
            vrp_status: Status::Ready,

            cached_maps: Vec::new(),
            saved_routes: Vec::new(),
            browse_selection: 0,

            file_browser: None,

            input_mode: InputMode {
                active: false,
                field: InputField::BoundingBox,
                buffer: String::new(),
            },

            start_time: Instant::now(),
        }
    }

    pub fn start_file_browser(&mut self, field: InputField) {
        let mut browser = FileBrowser::new(field);
        if let Err(e) = browser.refresh_entries() {
            self.log(
                LogLevel::Error,
                format!("Failed to open file browser: {}", e),
            );
            return;
        }
        self.file_browser = Some(browser);
        self.current_view = View::FileBrowser;
        self.log(LogLevel::Info, "File browser opened");
    }

    pub fn close_file_browser(&mut self, selected_path: Option<PathBuf>) {
        if let Some(path) = selected_path {
            let path_str = path.to_string_lossy().to_string();

            if let Some(browser) = &self.file_browser {
                match browser.target_field {
                    InputField::InputFile => {
                        self.input_file = Some(path_str.clone());
                        if self.output_file.is_none() {
                            let out = path_str
                                .replace(".geojson", ".rmp")
                                .replace(".json", ".rmp");
                            self.output_file = Some(out);
                        }
                        self.log(
                            LogLevel::Success,
                            format!("Input file selected: {}", path_str),
                        );
                        self.current_view = View::Compile;
                    }
                    InputField::OutputFile => {
                        self.output_file = Some(path_str.clone());
                        self.log(
                            LogLevel::Success,
                            format!("Output file selected: {}", path_str),
                        );
                        self.current_view = View::Compile;
                    }
                    InputField::CacheFile => {
                        self.cache_file = Some(path_str.clone());
                        self.log(
                            LogLevel::Success,
                            format!("Cache file selected: {}", path_str),
                        );
                        self.current_view = View::Optimize;
                    }
                    InputField::RouteFile => {
                        self.route_file = Some(path_str.clone());
                        self.log(
                            LogLevel::Success,
                            format!("Route file selected: {}", path_str),
                        );
                        self.current_view = View::Optimize;
                    }
                    InputField::CleanInputFile => {
                        self.clean_input_file = Some(path_str.clone());
                        if self.clean_output_file.is_none() {
                            let out = path_str
                                .replace(".geojson", ".cleaned.geojson")
                                .replace(".json", ".cleaned.json");
                            self.clean_output_file = Some(out);
                        }
                        self.log(
                            LogLevel::Success,
                            format!("Clean input file selected: {}", path_str),
                        );
                        self.current_view = View::Clean;
                    }
                    InputField::CleanOutputFile => {
                        self.clean_output_file = Some(path_str.clone());
                        self.log(
                            LogLevel::Success,
                            format!("Clean output file selected: {}", path_str),
                        );
                        self.current_view = View::Clean;
                    }
                    _ => {
                        self.current_view = View::Home;
                    }
                }
            }
        }

        self.file_browser = None;
    }

    pub fn log(&mut self, level: LogLevel, message: impl Into<String>) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.log_entries.push(LogEntry {
            timestamp,
            level,
            message: message.into(),
        });
        if self.log_entries.len() > 500 {
            self.log_entries.remove(0);
        }
        self.log_scroll = self.log_entries.len().saturating_sub(1);
    }

    pub fn start_input(&mut self, field: InputField) {
        self.input_mode.active = true;
        self.input_mode.field = field;
        self.input_mode.buffer.clear();
    }

    pub fn confirm_input(&mut self) {
        let value = self.input_mode.buffer.trim().to_string();
        if value.is_empty() {
            self.input_mode.active = false;
            return;
        }

        match self.input_mode.field.clone() {
            InputField::BoundingBox => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() == 4 {
                    if let (Ok(min_lon), Ok(min_lat), Ok(max_lon), Ok(max_lat)) = (
                        parts[0].parse::<f64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        let bbox = BoundingBox {
                            min_lon,
                            min_lat,
                            max_lon,
                            max_lat,
                        };
                        self.log(LogLevel::Success, format!("Bounding box set: {bbox}"));
                        self.bounding_box = Some(bbox);
                    } else {
                        self.log(LogLevel::Error, "Invalid coordinates".to_string());
                    }
                } else {
                    self.log(
                        LogLevel::Error,
                        "Expected format: min_lon,min_lat,max_lon,max_lat",
                    );
                }
            }
            InputField::InputFile => {
                self.input_file = Some(value.clone());
                if self.output_file.is_none() {
                    let out = value.replace(".geojson", ".rmp").replace(".json", ".rmp");
                    self.output_file = Some(out);
                }
                self.log(LogLevel::Success, format!("Input set: {}", value));
            }
            InputField::OutputFile => {
                self.output_file = Some(value.clone());
                self.log(LogLevel::Success, format!("Output set: {}", value));
            }
            InputField::CacheFile => {
                self.cache_file = Some(value.clone());
                self.log(LogLevel::Success, format!("Cache file set: {}", value));
            }
            InputField::RouteFile => {
                self.route_file = Some(value.clone());
                self.log(LogLevel::Success, format!("Route file set: {}", value));
            }
            InputField::LeftTurnPenalty => {
                if let Ok(v) = value.parse::<f64>() {
                    self.turn_penalties.left = v;
                    self.log(LogLevel::Success, format!("Left turn penalty set: {}", v));
                }
            }
            InputField::RightTurnPenalty => {
                if let Ok(v) = value.parse::<f64>() {
                    self.turn_penalties.right = v;
                    self.log(LogLevel::Success, format!("Right turn penalty set: {}", v));
                }
            }
            InputField::UTurnPenalty => {
                if let Ok(v) = value.parse::<f64>() {
                    self.turn_penalties.u_turn = v;
                    self.log(LogLevel::Success, format!("U-turn penalty set: {}", v));
                }
            }
            InputField::CleanInputFile => {
                self.clean_input_file = Some(value.clone());
                if self.clean_output_file.is_none() {
                    let out = value
                        .replace(".geojson", ".cleaned.geojson")
                        .replace(".json", ".cleaned.json");
                    self.clean_output_file = Some(out);
                }
                self.log(LogLevel::Success, format!("Clean input set: {}", value));
            }
            InputField::CleanOutputFile => {
                self.clean_output_file = Some(value.clone());
                self.log(LogLevel::Success, format!("Clean output set: {}", value));
            }
            InputField::VrpInputFile => {
                self.vrp_input_file = Some(value.clone());
                self.log(LogLevel::Success, format!("VRP input set: {}", value));
            }
            InputField::VrpOutputDir => {
                self.vrp_output_dir = value.clone();
                self.log(LogLevel::Success, format!("VRP output dir: {}", value));
            }
            InputField::VrpWaypointsFile => {
                self.vrp_waypoints_file = Some(value.clone());
                self.log(LogLevel::Success, format!("VRP waypoints file: {}", value));
            }
            InputField::VrpAlgorithm => {
                self.vrp_algo = value.clone();
                self.log(LogLevel::Success, format!("VRP algorithm: {}", value));
            }
            InputField::VrpCapacity => {
                if let Ok(v) = value.parse::<f64>() {
                    self.vrp_capacity = Some(v);
                    self.log(LogLevel::Success, format!("VRP capacity: {}", v));
                }
            }
            InputField::VrpDepot => {
                self.vrp_depots.push(value.clone());
                self.log(LogLevel::Success, format!("VRP depot added: {}", value));
            }
            InputField::DepotCoordinates => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() == 2 {
                    if let (Ok(lat), Ok(lon)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                        self.depot_coords = Some((lat, lon));
                        self.log(
                            LogLevel::Success,
                            format!("Depot set: {:.4},{:.4}", lat, lon),
                        );
                    }
                }
            }
            InputField::NumVehicles => {
                if let Ok(v) = value.parse::<usize>() {
                    self.num_vehicles = v;
                    self.log(LogLevel::Success, format!("Number of vehicles set: {}", v));
                }
            }
            InputField::SolverId => {
                self.solver_id = value.clone();
                self.log(LogLevel::Success, format!("Solver ID set: {}", value));
            }
        }
        self.input_mode.active = false;
    }

    pub fn cancel_input(&mut self) {
        self.input_mode.active = false;
        self.input_mode.buffer.clear();
        self.log(LogLevel::Info, "Input cancelled");
    }

    pub fn navigate_up(&mut self) {
        match self.current_view {
            View::Home => {
                self.workflow_selection = (self.workflow_selection + 6) % 7;
            }
            View::BrowseMaps => {
                let max = self.cached_maps.len().max(1);
                if max > 0 {
                    self.browse_selection = (self.browse_selection + max - 1) % max;
                }
            }
            View::BrowseRoutes => {
                let max = self.saved_routes.len().max(1);
                if max > 0 {
                    self.browse_selection = (self.browse_selection + max - 1) % max;
                }
            }
            _ => {}
        }
    }

    pub fn navigate_down(&mut self) {
        match self.current_view {
            View::Home => {
                self.workflow_selection = (self.workflow_selection + 1) % 7;
            }
            View::BrowseMaps => {
                let max = self.cached_maps.len().max(1);
                if max > 0 {
                    self.browse_selection = (self.browse_selection + 1) % max;
                }
            }
            View::BrowseRoutes => {
                let max = self.saved_routes.len().max(1);
                if max > 0 {
                    self.browse_selection = (self.browse_selection + 1) % max;
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_initialization() {
        let app = App::new();
        assert!(app.running);
        assert_eq!(app.workflow_selection, 0);
        assert!(app.log_entries.is_empty());
        assert_eq!(app.log_scroll, 0);

        assert_eq!(app.data_source, DataSource::Osm);
        assert!(app.bounding_box.is_none());
        assert_eq!(app.extract_status, Status::Ready);

        assert!(app.input_file.is_none());
        assert!(app.output_file.is_none());
        assert_eq!(app.compile_status, Status::Ready);

        assert_eq!(app.clean_options, CleanOptions::default());
        assert!(app.clean_input_file.is_none());
        assert!(app.clean_output_file.is_none());
        assert_eq!(app.clean_status, Status::Ready);
        assert_eq!(app.clean_selection, 0);

        assert!(app.cache_file.is_none());
        assert!(app.route_file.is_none());
        assert_eq!(app.turn_penalties, TurnPenalties::default());
        assert!(app.depot_coords.is_none());
        assert_eq!(app.num_vehicles, 1);
        assert_eq!(app.solver_id, "clarke_wright");
        assert_eq!(app.optimize_status, Status::Ready);

        assert!(app.cached_maps.is_empty());
        assert!(app.saved_routes.is_empty());
        assert_eq!(app.browse_selection, 0);

        assert!(app.file_browser.is_none());

        assert!(!app.input_mode.active);
        assert_eq!(app.input_mode.field, InputField::BoundingBox);
        assert!(app.input_mode.buffer.is_empty());
    }

    #[test]
    fn test_app_navigation() {
        let mut app = App::new();
        // Home view workflow selection
        assert_eq!(app.workflow_selection, 0);
        app.navigate_down();
        assert_eq!(app.workflow_selection, 1);
        app.navigate_up();
        assert_eq!(app.workflow_selection, 0);
    }


    #[test]
    fn test_app_cancel_input() {
        let mut app = App::new();
        app.input_mode.active = true;
        app.input_mode.buffer = String::from("some text");

        app.cancel_input();

        assert!(!app.input_mode.active);
        assert!(app.input_mode.buffer.is_empty());
        assert_eq!(app.log_entries.last().unwrap().message, "Input cancelled");
    }
    #[test]
    fn test_app_logging() {
        let mut app = App::new();
        app.log(LogLevel::Info, "test message");
        assert_eq!(app.log_entries.len(), 1);
        assert_eq!(app.log_entries[0].message, "test message");
    }
}
