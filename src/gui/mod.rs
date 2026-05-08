pub mod browse;
pub mod clean;
pub mod compile;
pub mod extract;
pub mod help;
pub mod home;
pub mod optimize;
pub mod vrp;

use eframe::egui;
use std::path::PathBuf;

use crate::core::clean::CleanOptions;
use crate::core::optimize::TurnPenalties;

// -- View enum --

#[derive(Debug, Clone, PartialEq)]
pub enum View {
    Home,
    Extract,
    Compile,
    Clean,
    Optimize,
    Vrp,
    BrowseMaps,
    BrowseRoutes,
    Help,
}

// -- Log entries --

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
            LogLevel::Success => write!(f, "OK"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERR"),
        }
    }
}

// -- Status --

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
                write!(f, "Running ({}%) {}", progress, message)
            }
            Status::Done(msg) => write!(f, "Done - {}", msg),
            Status::Error(msg) => write!(f, "Error: {}", msg),
        }
    }
}

// -- Data source --

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

// -- Bounding box --

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
            "{:.4},{:.4} to {:.4},{:.4}",
            self.min_lat, self.min_lon, self.max_lat, self.max_lon
        )
    }
}

// -- App state --

pub struct GuiApp {
    pub current_view: View,
    pub running: bool,

    // Logs
    pub log_entries: Vec<LogEntry>,

    // Extract
    pub data_source: DataSource,
    pub bounding_box: Option<BoundingBox>,
    pub extract_status: Status,
    pub bbox_input: String,

    // Compile
    pub input_file: Option<String>,
    pub output_file: Option<String>,
    pub compile_status: Status,

    // Clean
    pub clean_options: CleanOptions,
    pub clean_input_file: Option<String>,
    pub clean_output_file: Option<String>,
    pub clean_status: Status,

    // Optimize
    pub cache_file: Option<String>,
    pub route_file: Option<String>,
    pub turn_penalties: TurnPenalties,
    pub depot_coords: Option<(f64, f64)>,
    pub num_vehicles: usize,
    pub solver_id: String,
    pub optimize_status: Status,
    pub oneway_mode: crate::core::optimize::OnewayMode,
    pub optimize_bbox: Option<(f64, f64, f64, f64)>,
    pub optimize_bbox_input: String,

    // VRP
    pub vrp_input_file: Option<String>,
    pub vrp_csv_file: Option<String>,
    pub vrp_output_dir: String,
    pub vrp_vehicles: usize,
    pub vrp_algo: String,
    pub vrp_capacity: Option<f64>,
    pub vrp_depots: Vec<String>,
    pub vrp_depot_input: String,
    pub vrp_status: Status,

    // Browse
    pub cached_maps: Vec<String>,
    pub saved_routes: Vec<String>,
    pub browse_selection: usize,

    // Map view
    pub map_nodes: Vec<crate::core::optimize::RmpNode>,
    pub map_edges: Vec<crate::core::optimize::RmpEdge>,
    pub map_file_label: String,
    pub map_bounds: Option<(f64, f64, f64, f64)>,
    pub cpp_output: Option<crate::core::optimize::CppOutput>,
    pub map_solve_error: Option<String>,
    pub map_depot_text: String,
    pub map_oneway_mode: crate::core::optimize::OnewayMode,
    pub map_solving: bool,
    pub map_pan: egui::Vec2,
    pub map_zoom: f32,
    pub map_edge_width: f32,
}

impl Default for GuiApp {
    fn default() -> Self {
        Self {
            current_view: View::Home,
            running: true,
            log_entries: Vec::new(),

            data_source: DataSource::Osm,
            bounding_box: None,
            extract_status: Status::Ready,
            bbox_input: String::new(),

            input_file: None,
            output_file: None,
            compile_status: Status::Ready,

            clean_options: CleanOptions::default(),
            clean_input_file: None,
            clean_output_file: None,
            clean_status: Status::Ready,

            cache_file: None,
            route_file: None,
            turn_penalties: TurnPenalties::default(),
            depot_coords: None,
            num_vehicles: 1,
            solver_id: "clarke_wright".to_string(),
            optimize_status: Status::Ready,
            oneway_mode: crate::core::optimize::OnewayMode::default(),
            optimize_bbox: None,
            optimize_bbox_input: String::new(),

            vrp_input_file: None,
            vrp_csv_file: None,
            vrp_output_dir: String::new(),
            vrp_vehicles: 1,
            vrp_algo: "clarke_wright".to_string(),
            vrp_capacity: None,
            vrp_depots: Vec::new(),
            vrp_depot_input: String::new(),
            vrp_status: Status::Ready,

            cached_maps: Vec::new(),
            saved_routes: Vec::new(),
            browse_selection: 0,

            map_nodes: Vec::new(),
            map_edges: Vec::new(),
            map_file_label: String::new(),
            map_bounds: None,
            cpp_output: None,
            map_solve_error: None,
            map_depot_text: String::new(),
            map_oneway_mode: crate::core::optimize::OnewayMode::default(),
            map_solving: false,
            map_pan: egui::Vec2::ZERO,
            map_zoom: 1.0,
            map_edge_width: 1.5,
        }
    }
}

impl GuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();
        app.log(LogLevel::Info, format!("rmpca v{} started", env!("CARGO_PKG_VERSION")));
        app.log(LogLevel::Info, "Ready - select a workflow step to begin");
        app
    }

    pub fn log(&mut self, level: LogLevel, msg: impl Into<String>) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.log_entries.push(LogEntry {
            timestamp,
            level,
            message: msg.into(),
        });
        if self.log_entries.len() > 500 {
            self.log_entries.drain(..self.log_entries.len() - 500);
        }
    }

    pub fn refresh_cached_maps(&mut self) {
        self.cached_maps.clear();
        if let Some(cache_dir) = dirs::cache_dir() {
            let rmp_dir = cache_dir.join("rmpca");
            if rmp_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&rmp_dir) {
                    for entry in entries.flatten() {
                        if let Some(ext) = entry.path().extension() {
                            if ext == "rmp" {
                                if let Some(name) = entry.file_name().to_str() {
                                    self.cached_maps.push(name.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        self.cached_maps.sort();
    }

    pub fn refresh_saved_routes(&mut self) {
        self.saved_routes.clear();
        if let Some(data_dir) = dirs::data_dir() {
            let routes_dir = data_dir.join("rmpca").join("routes");
            if routes_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&routes_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file() {
                            if let Some(name) = entry.file_name().to_str() {
                                self.saved_routes.push(name.to_string());
                            }
                        }
                    }
                }
            }
        }
        self.saved_routes.sort();
    }

    fn load_rmp(&mut self, path: &std::path::Path) {
        match std::fs::read(path) {
            Ok(data) => match crate::core::optimize::read_rmp_file(&data) {
                Ok((nodes, edges)) => {
                    let label = path.display().to_string();
                    self.set_network(nodes, edges, label);
                    self.log(
                        LogLevel::Success,
                        format!("Loaded {} nodes, {} edges", self.map_nodes.len(), self.map_edges.len()),
                    );
                }
                Err(e) => {
                    self.log(LogLevel::Error, format!("Invalid .rmp file: {}", e));
                }
            },
            Err(e) => {
                self.log(LogLevel::Error, format!("Failed to read file: {}", e));
            }
        }
    }

    fn set_network(
        &mut self,
        nodes: Vec<crate::core::optimize::RmpNode>,
        edges: Vec<crate::core::optimize::RmpEdge>,
        label: String,
    ) {
        if nodes.is_empty() {
            self.map_bounds = None;
        } else {
            let mut min_lat = f64::MAX;
            let mut max_lat = f64::MIN;
            let mut min_lon = f64::MAX;
            let mut max_lon = f64::MIN;
            for n in &nodes {
                min_lat = min_lat.min(n.lat);
                max_lat = max_lat.max(n.lat);
                min_lon = min_lon.min(n.lon);
                max_lon = max_lon.max(n.lon);
            }
            let pad = 0.002;
            self.map_bounds = Some((min_lat - pad, max_lat + pad, min_lon - pad, max_lon + pad));
        }
        self.map_nodes = nodes;
        self.map_edges = edges;
        self.map_file_label = label;
        self.cpp_output = None;
        self.map_solve_error = None;
        self.map_pan = egui::Vec2::ZERO;
        self.map_zoom = 1.0;
    }
}

// -- eframe::App implementation --

impl eframe::App for GuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Top bar
        egui::Panel::top("top_panel").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(format!("rmpca v{}", env!("CARGO_PKG_VERSION")));
                ui.separator();
                if ui.small_button("Home").clicked() {
                    self.current_view = View::Home;
                }
                if ui.small_button("Help").clicked() {
                    self.current_view = View::Help;
                }
            });
        });

        // Bottom log panel
        egui::Panel::bottom("log_panel")
            .min_size(80.0)
            .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Logs");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").clicked() {
                        self.log_entries.clear();
                    }
                });
            });
            egui::ScrollArea::vertical()
                .max_height(120.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in &self.log_entries {
                        ui.horizontal(|ui| {
                            let ts_color = egui::Color32::from_rgb(80, 180, 220);
                            let level_color = match entry.level {
                                LogLevel::Info => egui::Color32::from_rgb(80, 180, 220),
                                LogLevel::Success => egui::Color32::from_rgb(80, 220, 80),
                                LogLevel::Warn => egui::Color32::from_rgb(220, 200, 60),
                                LogLevel::Error => egui::Color32::from_rgb(220, 80, 80),
                            };
                            ui.colored_label(ts_color, &entry.timestamp);
                            ui.colored_label(level_color, format!("[{}]", entry.level));
                            ui.label(&entry.message);
                        });
                    }
                });
        });

        // Left sidebar
        egui::Panel::left("sidebar")
            .min_size(180.0)
            .default_size(200.0)
            .show_inside(ui, |ui| {
            ui.vertical(|ui| {
                ui.heading("Workflow");
                ui.separator();

                let items = [
                    ("Extract Data", View::Extract),
                    ("Clean GeoJSON", View::Clean),
                    ("Compile Map", View::Compile),
                    ("CPP Solver", View::Optimize),
                    ("VRP Solver", View::Vrp),
                    ("Cached Maps", View::BrowseMaps),
                    ("Saved Routes", View::BrowseRoutes),
                ];

                for (label, view) in items {
                    let is_selected = self.current_view == view;
                    if ui.selectable_label(is_selected, label).clicked() {
                        self.current_view = view.clone();
                        if view == View::BrowseMaps {
                            self.refresh_cached_maps();
                            self.browse_selection = 0;
                        }
                        if view == View::BrowseRoutes {
                            self.refresh_saved_routes();
                            self.browse_selection = 0;
                        }
                    }
                }
            });
        });

        // Central panel
        egui::CentralPanel::default().show_inside(ui, |ui| {
            match self.current_view {
                View::Home => home::draw(ui, self),
                View::Extract => extract::draw(ui, self),
                View::Compile => compile::draw(ui, self),
                View::Clean => clean::draw(ui, self),
                View::Optimize => optimize::draw(ui, self),
                View::Vrp => vrp::draw(ui, self),
                View::BrowseMaps => browse::draw_maps(ui, self),
                View::BrowseRoutes => browse::draw_routes(ui, self),
                View::Help => help::draw(ui),
            }
        });

        ctx.request_repaint();
    }
}

// -- File picker helpers (native only) --

pub fn pick_file(ui: &mut egui::Ui, extensions: &[&str]) -> Option<PathBuf> {
    if ui.button("Browse...").clicked() {
        return rfd::FileDialog::new()
            .add_filter("Supported", extensions)
            .pick_file();
    }
    None
}

pub fn pick_save_file(ui: &mut egui::Ui, extensions: &[&str]) -> Option<PathBuf> {
    if ui.button("Save as...").clicked() {
        return rfd::FileDialog::new()
            .add_filter("Supported", extensions)
            .pick_file();
    }
    None
}

pub fn pick_folder(ui: &mut egui::Ui) -> Option<PathBuf> {
    if ui.button("Browse...").clicked() {
        return rfd::FileDialog::new().pick_folder();
    }
    None
}

/// Draw a status indicator label.
pub fn status_label(ui: &mut egui::Ui, status: &Status) {
    match status {
        Status::Ready => {
            ui.colored_label(egui::Color32::from_rgb(160, 160, 160), "Status: Ready");
        }
        Status::Running { progress, message } => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.colored_label(
                    egui::Color32::from_rgb(80, 180, 220),
                    format!("Status: Running ({}%) {}", progress, message),
                );
            });
        }
        Status::Done(msg) => {
            ui.colored_label(egui::Color32::from_rgb(80, 220, 80), format!("Status: Done - {}", msg));
        }
        Status::Error(msg) => {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), format!("Status: Error: {}", msg));
        }
    }
}

/// Map a WGS-84 (lat, lon) to screen-space position for map rendering.
pub fn project_latlon(
    lat: f64,
    lon: f64,
    canvas_center: egui::Pos2,
    canvas_size: egui::Vec2,
    bounds: (f64, f64, f64, f64),
    zoom: f32,
    pan: egui::Vec2,
) -> egui::Pos2 {
    let (min_lat, max_lat, min_lon, max_lon) = bounds;
    let lat_range = (max_lat - min_lat).max(0.001);
    let lon_range = (max_lon - min_lon).max(0.001);
    let scale_x = canvas_size.x / lon_range as f32;
    let scale_y = canvas_size.y / lat_range as f32;
    let scale = scale_x.min(scale_y) * zoom;
    let x = (lon - min_lon) as f32 * scale + (canvas_size.x - lon_range as f32 * scale) * 0.5;
    let y = (max_lat - lat) as f32 * scale + (canvas_size.y - lat_range as f32 * scale) * 0.5;
    egui::pos2(
        canvas_center.x + x - canvas_size.x * 0.5 + pan.x,
        canvas_center.y + y - canvas_size.y * 0.5 + pan.y,
    )
}

/// Render the map canvas (reused across views that need map viz).
pub fn draw_map_canvas(ui: &mut egui::Ui, app: &mut GuiApp) {
    let (rect, response) = ui.allocate_exact_size(
        ui.available_size(),
        egui::Sense::click_and_drag(),
    );

    if ui.is_rect_visible(rect) {
        let painter = ui.painter_at(rect);

        if app.map_nodes.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No map loaded. Open a .rmp file to visualize the network.",
                egui::FontId::proportional(16.0),
                egui::Color32::from_gray(140),
            );
            return;
        }

        let canvas_center = rect.center();
        let canvas_size = rect.size();

        // Zoom & pan with scroll and drag
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta);
            app.map_zoom = (app.map_zoom + scroll.y * 0.005).max(0.1).min(50.0);
        }
        if response.dragged() {
            app.map_pan += response.drag_delta();
        }
        if response.double_clicked() {
            app.map_pan = egui::Vec2::ZERO;
            app.map_zoom = 1.0;
        }

        let bounds = match app.map_bounds {
            Some(b) => b,
            None => return,
        };

        let screen_pts: Vec<egui::Pos2> = app
            .map_nodes
            .iter()
            .map(|n| {
                project_latlon(
                    n.lat,
                    n.lon,
                    canvas_center,
                    canvas_size,
                    bounds,
                    app.map_zoom,
                    app.map_pan,
                )
            })
            .collect();

        let base_edge_color = egui::Color32::from_gray(80);
        let circuit_edge_color = egui::Color32::from_rgb(255, 180, 40);

        let circuit_set: Option<std::collections::HashSet<(u32, u32)>> =
            app.cpp_output.as_ref().map(|out| {
                let mut set = std::collections::HashSet::new();
                for w in out.circuit.windows(2) {
                    set.insert((w[0], w[1]));
                    set.insert((w[1], w[0]));
                }
                set
            });

        for edge in &app.map_edges {
            let a = screen_pts[edge.from as usize];
            let b = screen_pts[edge.to as usize];
            let color = if let Some(ref cs) = circuit_set {
                if cs.contains(&(edge.from, edge.to)) {
                    circuit_edge_color
                } else {
                    base_edge_color
                }
            } else {
                base_edge_color
            };
            painter.line_segment([a, b], egui::Stroke::new(app.map_edge_width, color));
        }

        let node_radius = (2.0 * app.map_zoom).max(0.8).min(6.0);
        let node_color = egui::Color32::from_rgb(100, 180, 255);
        let circuit_node_color = egui::Color32::from_rgb(255, 220, 80);

        for (i, pt) in screen_pts.iter().enumerate() {
            let on_circuit = app
                .cpp_output
                .as_ref()
                .map_or(false, |o| o.circuit.contains(&(i as u32)));
            painter.circle_filled(*pt, node_radius, if on_circuit { circuit_node_color } else { node_color });
        }

        if let Some(ref out) = app.cpp_output {
            if out.circuit.len() > 1 {
                let circuit_pts: Vec<egui::Pos2> = out
                    .circuit
                    .iter()
                    .map(|&idx| screen_pts[idx as usize])
                    .collect();
                if !circuit_pts.is_empty() {
                    painter.add(egui::Shape::line(
                        circuit_pts,
                        egui::Stroke::new(app.map_edge_width * 1.8, circuit_edge_color),
                    ));
                }
            }
        }

        // Cursor position
        if let Some(mouse) = response.hover_pos() {
            let (min_lat, max_lat, min_lon, max_lon) = bounds;
            let lat_range = (max_lat - min_lat).max(0.001);
            let lon_range = (max_lon - min_lon).max(0.001);
            let scale_x = canvas_size.x / lon_range as f32;
            let scale_y = canvas_size.y / lat_range as f32;
            let scale = scale_x.min(scale_y) * app.map_zoom;
            let rel_x = mouse.x - canvas_center.x - app.map_pan.x + canvas_size.x * 0.5;
            let rel_y = mouse.y - canvas_center.y - app.map_pan.y + canvas_size.y * 0.5;
            let lon = rel_x / scale + min_lon as f32;
            let lat = max_lat as f32 - rel_y / scale;
            painter.text(
                egui::pos2(rect.min.x + 8.0, rect.max.y - 24.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{:.5}, {:.5}", lat, lon),
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(180),
            );
        }
    }
}
