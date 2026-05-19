use eframe::egui;

use crate::gui::{GuiApp, LogLevel, Status};

pub fn draw(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("🔨 Compile Map");
        ui.label("Convert GeoJSON → binary .rmp format for instant loading");
        ui.separator();

        // Input file
        ui.group(|ui| {
            ui.heading("Input GeoJSON");
            if let Some(ref path) = app.input_file {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
            } else {
                ui.colored_label(egui::Color32::from_rgb(220, 200, 60), "(not set)");
            }
            ui.horizontal(|ui| {
                if ui.button("📂 Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("GeoJSON", &["geojson", "json"])
                        .pick_file()
                    {
                        app.input_file = Some(path.display().to_string());
                        app.log(LogLevel::Success, format!("Input set: {}", path.display()));
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.input_file = None;
                }
            });
        });

        // Output file
        ui.group(|ui| {
            ui.heading("Output .rmp (optional — auto-derived from input)");
            if let Some(ref path) = app.output_file {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(140, 140, 140),
                    "(auto-derived from input)",
                );
            }
            ui.horizontal(|ui| {
                if ui.button("📂 Save as…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("RMP", &["rmp"])
                        .pick_file()
                    {
                        app.output_file = Some(path.display().to_string());
                        app.log(LogLevel::Success, format!("Output set: {}", path.display()));
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.output_file = None;
                }
            });
        });

        // Compile button
        ui.group(|ui| {
            super::status_label(ui, &app.compile_status);

            let can_run = app.input_file.is_some();
            if ui
                .add_enabled(can_run, egui::Button::new("🔨 Compile"))
                .clicked()
            {
                run_compile(app);
            }
        });
    });
}

fn run_compile(app: &mut GuiApp) {
    let input = match &app.input_file {
        Some(p) => p.clone(),
        None => {
            app.log(LogLevel::Warn, "Set an input GeoJSON file first");
            return;
        }
    };

    let output = app
        .output_file
        .clone()
        .unwrap_or_else(|| input.replace(".geojson", ".rmp").replace(".json", ".rmp"));

    app.compile_status = Status::Running {
        progress: 0,
        message: "Compiling…".to_string(),
    };
    app.log(
        LogLevel::Info,
        format!("Compiling {} → {}", input, output.clone()),
    );

    let req = crate::core::compile::CompileRequest {
        input_geojson: input,
        output_rmp: output.clone(),
        compress: true,
        road_classes: Vec::new(),
        clean_options: None,
        prune_disconnected: false,
        prune_spurs: false,
    };

    match crate::core::compile::run_compile(&req) {
        Ok(result) => {
            app.compile_status = Status::Done(format!(
                "{} nodes, {} edges, {:.1} KB",
                result.node_count,
                result.edge_count,
                result.output_size_bytes as f64 / 1024.0
            ));
            app.log(
                LogLevel::Success,
                format!(
                    "Compile complete: {} nodes, {} edges",
                    result.node_count, result.edge_count
                ),
            );

            // Copy the compiled .rmp to the cache directory so it appears in Cached Maps
            if let Some(cache_dir) = dirs::cache_dir() {
                let rmp_dir = cache_dir.join("rmpca");
                if let Err(e) = std::fs::create_dir_all(&rmp_dir) {
                    app.log(LogLevel::Warn, format!("Could not create cache dir: {}", e));
                } else {
                    let file_name = std::path::Path::new(&output)
                        .file_name()
                        .unwrap_or_default();
                    let dest = rmp_dir.join(file_name);
                    if let Err(e) = std::fs::copy(&output, &dest) {
                        app.log(LogLevel::Warn, format!("Could not cache map: {}", e));
                    }
                }
            }
            app.refresh_cached_maps();
        }
        Err(e) => {
            app.compile_status = Status::Error(e.to_string());
            app.log(LogLevel::Error, format!("Compile failed: {}", e));
        }
    }
}
