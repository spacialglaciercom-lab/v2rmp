use eframe::egui;

use crate::core::clean::CleanOptions;
use crate::gui::{GuiApp, LogLevel, Status};

pub fn draw(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("🧹 Clean GeoJSON");
        ui.label("Repair geometry, remove duplicates, keep largest component");
        ui.separator();

        // Input / Output files
        ui.group(|ui| {
            ui.heading("Files");
            ui.horizontal(|ui| {
                ui.label("Input:");
                if let Some(ref path) = app.clean_input_file {
                    ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
                } else {
                    ui.colored_label(egui::Color32::from_rgb(220, 200, 60), "(not set)");
                }
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("GeoJSON", &["geojson", "json"])
                        .pick_file()
                    {
                        app.clean_input_file = Some(path.display().to_string());
                        app.log(LogLevel::Success, format!("Clean input set: {}", path.display()));
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Output:");
                if let Some(ref path) = app.clean_output_file {
                    ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
                } else {
                    ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "(auto-derived)");
                }
                if ui.button("Save as…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("GeoJSON", &["geojson", "json"])
                        .pick_file()
                    {
                        app.clean_output_file = Some(path.display().to_string());
                        app.log(LogLevel::Success, format!("Clean output set: {}", path.display()));
                    }
                }
            });
        });

        // Options
        ui.group(|ui| {
            ui.heading("Options");
            let opts = &mut app.clean_options;
            ui.checkbox(&mut opts.make_valid, "Make valid geometries");
            ui.checkbox(&mut opts.drop_invalid, "Drop invalid features");
            ui.checkbox(&mut opts.remove_selfloops, "Remove self-loops");
            ui.checkbox(&mut opts.dedupe_edges, "Dedupe edges");
            ui.checkbox(&mut opts.remove_isolates, "Remove isolates");
            ui.checkbox(&mut opts.merge_node_positions, "Merge node positions");
            ui.checkbox(&mut opts.include_polygons, "Include polygons");
            ui.checkbox(&mut opts.include_points, "Include points");
            ui.checkbox(&mut opts.merge_parallel_edges, "Merge parallel edges");
            ui.checkbox(&mut opts.merge_parallel_edge_properties, "Merge parallel edge properties");
        });

        // Numeric parameters
        ui.group(|ui| {
            ui.heading("Parameters");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.clean_options.min_length_m).speed(0.1).range(0.0..=100.0));
                ui.label("Min length (m)");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.clean_options.node_snap_m).speed(0.5).range(0.0..=100.0));
                ui.label("Node snap (m)");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.clean_options.max_components).speed(1).range(0..=100));
                ui.label("Max components");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.clean_options.simplify_tolerance_m).speed(0.5).range(0.0..=100.0));
                ui.label("Simplify tolerance (m)");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.clean_options.node_precision_decimals).speed(1).range(1..=12));
                ui.label("Precision decimals");
            });
        });

        // Reset & Run
        ui.group(|ui| {
            super::status_label(ui, &app.clean_status);
            ui.horizontal(|ui| {
                if ui.button("↺ Reset to Defaults").clicked() {
                    app.clean_options = CleanOptions::default();
                    app.log(LogLevel::Info, "Clean options reset to defaults");
                }
                let can_run = app.clean_input_file.is_some();
                if ui.add_enabled(can_run, egui::Button::new("🚀 Run Clean")).clicked() {
                    run_clean(app);
                }
            });
        });
    });
}

fn run_clean(app: &mut GuiApp) {
    let input_path = match app.clean_input_file.clone() {
        Some(p) => p,
        None => {
            app.log(LogLevel::Warn, "Set an input file first");
            return;
        }
    };

    app.clean_status = Status::Running { progress: 0, message: "Cleaning…".to_string() };
    app.log(LogLevel::Info, "Starting GeoJSON cleaning");

    let output_path = app.clean_output_file.clone().unwrap_or_else(|| {
        input_path.replace(".geojson", ".cleaned.geojson").replace(".json", ".cleaned.json")
    });

    let mut input_data = Vec::new();
    match std::fs::File::open(&input_path) {
        Ok(mut file) => {
            if let Err(e) = std::io::Read::read_to_end(&mut file, &mut input_data) {
                app.clean_status = Status::Error(format!("Failed to read input: {}", e));
                app.log(LogLevel::Error, format!("Read error: {}", e));
                return;
            }
        }
        Err(e) => {
            app.clean_status = Status::Error(format!("Failed to open input: {}", e));
            app.log(LogLevel::Error, format!("Open error: {}", e));
            return;
        }
    }

    let geojson: geojson::FeatureCollection = match serde_json::from_slice(&input_data) {
        Ok(fc) => fc,
        Err(e) => {
            app.clean_status = Status::Error(format!("Failed to parse GeoJSON: {}", e));
            app.log(LogLevel::Error, format!("Parse error: {}", e));
            return;
        }
    };

    match crate::core::clean::clean_geojson(&geojson, &app.clean_options) {
        Ok((cleaned, stats, warnings)) => {
            let output_json = serde_json::to_string_pretty(&cleaned).unwrap_or_default();
            match std::fs::write(&output_path, &output_json) {
                Ok(_) => {
                    for warning in &warnings {
                        app.log(LogLevel::Warn, warning.clone());
                    }
                    let summary = stats.summary();
                    app.clean_status = Status::Done(summary.clone());
                    app.log(LogLevel::Success, format!(
                        "Cleaning complete: {} → {} features",
                        stats.input_features, stats.output_features
                    ));
                    app.log(LogLevel::Info, format!("Output saved to: {}", output_path));
                }
                Err(e) => {
                    app.clean_status = Status::Error(format!("Failed to write output: {}", e));
                    app.log(LogLevel::Error, format!("Write error: {}", e));
                }
            }
        }
        Err(e) => {
            app.clean_status = Status::Error(e.to_string());
            app.log(LogLevel::Error, format!("Cleaning failed: {}", e));
        }
    }
}
