use std::time::Instant;

use crate::core::optimize::TurnPenalties;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum View {
    Home,
    Extract,
    Compile,
    Optimize,
    BrowseMaps,
    BrowseRoutes,
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
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

pub struct InputMode {
    pub active: bool,
    pub field: InputField,
    pub buffer: String,
}

#[derive(Debug, Clone, PartialEq)]
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

    // Optimize state
    pub cache_file: Option<String>,
    pub route_file: Option<String>,
    pub turn_penalties: TurnPenalties,
    pub depot_coords: Option<(f64, f64)>,
    pub optimize_status: Status,

    // Browse state
    pub cached_maps: Vec<String>,
    pub saved_routes: Vec<String>,
    pub browse_selection: usize,

    // Input mode
    pub input_mode: InputMode,

    // Timing
    pub start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Scan current directory for .rmp cache files
    fn scan_cached_maps() -> Vec<String> {
        use std::fs;
        let mut maps = Vec::new();

        if let Ok(entries) = fs::read_dir(".") {
            for entry in entries.flatten() {
                if let Ok(file_name) = entry.file_name().into_string() {
                    if file_name.ends_with(".rmp") {
                        maps.push(file_name);
                    }
                }
            }
        }

        maps.sort();
        maps
    }

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

            cache_file: None,
            route_file: None,
            turn_penalties: TurnPenalties::default(),
            depot_coords: None,
            optimize_status: Status::Ready,

            cached_maps: Self::scan_cached_maps(),
            saved_routes: Vec::new(),
            browse_selection: 0,

            input_mode: InputMode {
                active: false,
                field: InputField::BoundingBox,
                buffer: String::new(),
            },

            start_time: Instant::now(),
        }
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

        match self.input_mode.field {
            InputField::BoundingBox => {
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() == 4 {
                    if let (Ok(min_lon), Ok(min_lat), Ok(max_lon), Ok(max_lat)) = (
                        parts[0].parse::<f64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    ) {
                        if min_lon >= max_lon || min_lat >= max_lat {
                            self.log(LogLevel::Error, "Invalid bounding box: min must be less than max".to_string());
                            self.input_mode.active = false;
                            return;
                        }
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
                self.workflow_selection = (self.workflow_selection + 4) % 5;
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
                self.workflow_selection = (self.workflow_selection + 1) % 5;
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

        // Basic state
        assert!(app.running);
        assert_eq!(app.current_view, View::Home);
        assert_eq!(app.workflow_selection, 0);
        assert!(app.log_entries.is_empty());
        assert_eq!(app.log_scroll, 0);

        // Extract state
        assert_eq!(app.data_source, DataSource::Osm);
        assert!(app.bounding_box.is_none());
        assert_eq!(app.extract_status, Status::Ready);

        // Compile state
        assert!(app.input_file.is_none());
        assert!(app.output_file.is_none());
        assert_eq!(app.compile_status, Status::Ready);

        // Optimize state
        assert!(app.cache_file.is_none());
        assert!(app.route_file.is_none());
        assert_eq!(app.turn_penalties.left, 1.0);
        assert_eq!(app.turn_penalties.right, 0.0);
        assert_eq!(app.turn_penalties.u_turn, 5.0);
        assert!(app.depot_coords.is_none());
        assert_eq!(app.optimize_status, Status::Ready);

        // Browse state
        // app.cached_maps depends on filesystem, but should be a Vec
        assert!(app.saved_routes.is_empty());
        assert_eq!(app.browse_selection, 0);

        // Input mode
        assert!(!app.input_mode.active);
        assert_eq!(app.input_mode.field, InputField::BoundingBox);
        assert!(app.input_mode.buffer.is_empty());

    }

    #[test]
    fn test_log_truncation() {
        let mut app = App::new();

        // Add 501 entries
        for i in 0..501 {
            app.log(LogLevel::Info, format!("Message {}", i));
        }

        // Verify length is capped at 500
        assert_eq!(app.log_entries.len(), 500);

        // Verify oldest was removed (first entry should be "Message 1")
        assert_eq!(app.log_entries[0].message, "Message 1");

        // Verify latest is correct
        assert_eq!(app.log_entries[499].message, "Message 500");

        // Verify scroll position
        assert_eq!(app.log_scroll, 499);
    }

    #[test]
    fn test_invalid_bbox_input() {
        let mut app = App::new();
        app.input_mode.field = InputField::BoundingBox;

        // Invalid: min > max
        app.input_mode.buffer = "10.0,20.0,5.0,25.0".to_string();
        app.confirm_input();
        assert!(app.bounding_box.is_none());
        assert_eq!(app.log_entries.last().unwrap().level, LogLevel::Error);
        assert!(app.log_entries.last().unwrap().message.contains("Invalid bounding box"));

        // Valid
        app.input_mode.active = true;
        app.input_mode.buffer = "5.0,15.0,10.0,25.0".to_string();
        app.confirm_input();
        assert!(app.bounding_box.is_some());
        assert_eq!(app.log_entries.last().unwrap().level, LogLevel::Success);
    }
}
