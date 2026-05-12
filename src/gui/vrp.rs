use eframe::egui;

use crate::gui::{GuiApp, LogLevel, Status};

pub fn draw(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("🚚 VRP Solver — Multi-Vehicle Route Planning");
        ui.separator();

        // Coordinates CSV (primary input)
        ui.group(|ui| {
            ui.heading("📍 Coordinates CSV (required)");
            ui.label("Columns: lat,lon [, label, demand, type]");
            if let Some(ref path) = app.vrp_csv_file {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
            } else {
                ui.colored_label(egui::Color32::from_rgb(220, 200, 60), "(not set — browse for a CSV)");
            }
            ui.horizontal(|ui| {
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("CSV", &["csv"])
                        .pick_file()
                    {
                        let path_str = path.display().to_string();
                        app.vrp_csv_file = Some(path_str.clone());
                        app.log(LogLevel::Success, format!("Coordinates CSV set: {}", path_str));
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.vrp_csv_file = None;
                }
            });
        });

        // Road network .rmp (optional, for preview only)
        ui.group(|ui| {
            ui.heading("🗺️ Road Network (.rmp) — optional");
            ui.label("Used for map preview; not required for solving.");
            if let Some(ref path) = app.vrp_input_file {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
            } else {
                ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "(not set)");
            }
            ui.horizontal(|ui| {
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("RMP", &["rmp"])
                        .pick_file()
                    {
                        let path_str = path.display().to_string();
                        app.vrp_input_file = Some(path_str.clone());
                        app.load_rmp(&path);
                        app.log(LogLevel::Success, format!("Road network loaded: {}", path_str));
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.vrp_input_file = None;
                }
            });
        });

        // Output dir
        ui.group(|ui| {
            ui.heading("Output Directory");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut app.vrp_output_dir);
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        app.vrp_output_dir = path.display().to_string();
                        app.log(LogLevel::Success, format!("Output dir: {}", app.vrp_output_dir));
                    }
                }
            });
        });

        // Vehicles & Algorithm
        ui.group(|ui| {
            ui.heading("Vehicles & Algorithm");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.vrp_vehicles).speed(1).range(1..=100));
                ui.label("Vehicles");
            });

            let algo_options = ["greedy", "savings", "local_search", "simulated_annealing"];
            egui::ComboBox::from_id_salt("vrp_algo_select")
                .selected_text(app.vrp_algo.clone())
                .show_ui(ui, |ui| {
                    for opt in &algo_options {
                        ui.selectable_value(&mut app.vrp_algo, opt.to_string(), *opt);
                    }
            });
        });

        // Capacity
        ui.group(|ui| {
            ui.heading("Capacity");
            let mut cap_val = app.vrp_capacity.unwrap_or(0.0);
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut cap_val).speed(1.0).range(0.0..=10000.0));
                ui.label("Vehicle capacity");
                if cap_val > 0.0 {
                    app.vrp_capacity = Some(cap_val);
                } else {
                    app.vrp_capacity = None;
                }
            });
            if app.vrp_capacity.is_none() {
                ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "(unlimited)");
            }
        });

        // Run
        ui.group(|ui| {
            super::status_label(ui, &app.vrp_status);
            let can_run = app.vrp_csv_file.is_some();
            if ui.add_enabled(can_run, egui::Button::new("🚀 Run VRP Solver")).clicked() {
                run_vrp(app);
            }
        });

        // Map canvas
        if !app.map_nodes.is_empty() {
            ui.group(|ui| {
                ui.heading("Map Preview");
                ui.horizontal(|ui| {
                    ui.label("Edge width:");
                    ui.add(egui::Slider::new(&mut app.map_edge_width, 0.5..=5.0));
                });
                super::draw_map_canvas(ui, app);
            });
        }
    });
}

fn run_vrp(app: &mut GuiApp) {
    let csv_path = match &app.vrp_csv_file {
        Some(p) => p.clone(),
        None => {
            app.log(LogLevel::Warn, "Load a coordinates CSV first");
            return;
        }
    };

    app.vrp_status = Status::Running { progress: 0, message: "Solving VRP…".to_string() };
    app.log(LogLevel::Info, format!("Starting VRP with {} vehicles, algo={}", app.vrp_vehicles, app.vrp_algo));

    // Parse stops from CSV
    let (stops, _depot_indices) = match crate::core::vrp::utils::parse_csv_stops(&csv_path) {
        Ok(result) => result,
        Err(e) => {
            app.vrp_status = Status::Error(e.clone());
            app.log(LogLevel::Error, format!("CSV parse failed: {}", e));
            return;
        }
    };

    app.log(LogLevel::Info, format!("Loaded {} stops from CSV ({} depots)", stops.len(), _depot_indices.len()));

    let matrix = crate::core::vrp::utils::build_haversine_matrix(&stops, 40.0);
    let input = crate::core::vrp::types::VRPSolverInput {
        locations: stops,
        num_vehicles: app.vrp_vehicles,
        vehicle_capacity: app.vrp_capacity.unwrap_or(100.0),
        objective: crate::core::vrp::types::VrpObjective::MinDistance,
        matrix: Some(matrix),
        service_time_secs: None,
        use_time_windows: false,
        window_open: None,
        window_close: None,
        hyperparams: None,
    };

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    match rt.block_on(crate::core::vrp::registry::solve_with(&app.vrp_algo, &input)) {
        Ok(output) => {
            app.vrp_status = Status::Done(format!("{} km, {} stops", output.total_distance_km, output.stops.len()));
            app.log(LogLevel::Success, format!(
                "VRP complete: {} km total distance, {} stops",
                output.total_distance_km, output.stops.len()
            ));
        }
        Err(e) => {
            app.vrp_status = Status::Error(e.clone());
            app.log(LogLevel::Error, format!("VRP failed: {}", e));
        }
    }
}
