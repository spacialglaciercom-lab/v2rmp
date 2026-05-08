use eframe::egui;

use crate::gui::{GuiApp, LogLevel, Status};

pub fn draw(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("🚗 Optimize Route");
        ui.label("Optimize route with turn penalties on a cached map");
        ui.separator();

        // Cache file
        ui.group(|ui| {
            ui.heading("Cache File (.rmp)");
            if let Some(ref path) = app.cache_file {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
            } else {
                ui.colored_label(egui::Color32::from_rgb(220, 200, 60), "(not set)");
            }
            ui.horizontal(|ui| {
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("RMP", &["rmp"])
                        .pick_file()
                    {
                        let path_str = path.display().to_string();
                        app.cache_file = Some(path_str.clone());
                        app.load_rmp(&path);
                        app.log(LogLevel::Success, format!("Cache file set: {}", path_str));
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.cache_file = None;
                }
            });
        });

        // Bounding box filter
        ui.group(|ui| {
            ui.heading("Bounding Box Filter (optional)");
            ui.label("Only optimize nodes within this area. Leave empty for full map.");
            if let Some((min_lat, max_lat, min_lon, max_lon)) = app.optimize_bbox {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80),
                    format!("{:.4},{:.4} to {:.4},{:.4}", min_lat, min_lon, max_lat, max_lon));
            } else {
                ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "(full map — no filter)");
            }
            ui.horizontal(|ui| {
                ui.label("min_lat,min_lon,max_lat,max_lon:");
                let _resp = ui.text_edit_singleline(&mut app.optimize_bbox_input);
                if ui.button("Set BBox").clicked() {
                    let parts: Vec<&str> = app.optimize_bbox_input.split(',').collect();
                    if parts.len() == 4 {
                        if let (Ok(mla), Ok(mlo), Ok(xla), Ok(xlo)) = (
                            parts[0].trim().parse::<f64>(),
                            parts[1].trim().parse::<f64>(),
                            parts[2].trim().parse::<f64>(),
                            parts[3].trim().parse::<f64>(),
                        ) {
                            app.optimize_bbox = Some((mla, xla, mlo, xlo));
                            app.log(LogLevel::Success, format!("BBox filter set: {:.4},{:.4} to {:.4},{:.4}", mla, mlo, xla, xlo));
                        } else {
                            app.log(LogLevel::Error, "Invalid coordinates");
                        }
                    } else {
                        app.log(LogLevel::Error, "Use format: min_lat,min_lon,max_lat,max_lon");
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.optimize_bbox = None;
                    app.optimize_bbox_input.clear();
                    app.log(LogLevel::Info, "BBox filter cleared — using full map");
                }
            });
        });

        // Route file (optional)
        ui.group(|ui| {
            ui.heading("Route File (optional)");
            if let Some(ref path) = app.route_file {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80), path);
            } else {
                ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "(not set)");
            }
            ui.horizontal(|ui| {
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Route", &["json", "geojson"])
                        .pick_file()
                    {
                        app.route_file = Some(path.display().to_string());
                        app.log(LogLevel::Success, format!("Route file set: {}", path.display()));
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.route_file = None;
                }
            });
        });

        // Turn penalties
        ui.group(|ui| {
            ui.heading("Turn Penalties");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.turn_penalties.left).speed(0.1).range(0.0..=60.0));
                ui.label("Left turn (s)");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.turn_penalties.right).speed(0.1).range(0.0..=60.0));
                ui.label("Right turn (s)");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.turn_penalties.u_turn).speed(0.1).range(0.0..=60.0));
                ui.label("U-turn (s)");
            });
        });

        // Depot & Vehicles
        ui.group(|ui| {
            ui.heading("Depot & Vehicles");
            if let Some((lat, lon)) = app.depot_coords {
                ui.colored_label(egui::Color32::from_rgb(80, 220, 80), format!("Depot: {:.4},{:.4}", lat, lon));
            } else {
                ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "(not set)");
            }
            ui.horizontal(|ui| {
                ui.label("Depot (lat,lon):");
                let _response = ui.text_edit_singleline(&mut app.map_depot_text);
                if ui.button("Set Depot").clicked() {
                    let parts: Vec<&str> = app.map_depot_text.split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(lat), Ok(lon)) = (parts[0].trim().parse::<f64>(), parts[1].trim().parse::<f64>()) {
                            app.depot_coords = Some((lat, lon));
                            app.log(LogLevel::Success, format!("Depot set: {:.4},{:.4}", lat, lon));
                        }
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut app.num_vehicles).speed(1).range(1..=100));
                ui.label("Vehicles");
            });
        });

        // Solver & Oneway
        ui.group(|ui| {
            ui.heading("Solver");
            let solver_options = crate::core::vrp::registry::get_algorithm_options();
            egui::ComboBox::from_id_salt("solver_select")
                .selected_text(app.solver_id.clone())
                .show_ui(ui, |ui| {
                    for (id, label) in &solver_options {
                        ui.selectable_value(&mut app.solver_id, id.clone(), label);
                    }
                });

            ui.horizontal(|ui| {
                ui.label("One-way mode:");
                egui::ComboBox::from_id_salt("oneway_select")
                    .selected_text(format!("{:?}", app.oneway_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut app.oneway_mode, crate::core::optimize::OnewayMode::Ignore, "Ignore");
                        ui.selectable_value(&mut app.oneway_mode, crate::core::optimize::OnewayMode::Respect, "Respect");
                        ui.selectable_value(&mut app.oneway_mode, crate::core::optimize::OnewayMode::Reverse, "Reverse");
                    });
            });
        });

        // Run
        ui.group(|ui| {
            super::status_label(ui, &app.optimize_status);
            let can_run = app.cache_file.is_some();
            if ui.add_enabled(can_run, egui::Button::new("🚀 Optimize Route")).clicked() {
                run_optimize(app);
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

fn run_optimize(app: &mut GuiApp) {
    let cache_path = match &app.cache_file {
        Some(p) => p.clone(),
        None => {
            app.log(LogLevel::Warn, "Set a cache file first");
            return;
        }
    };

    app.optimize_status = Status::Running { progress: 0, message: "Optimizing…".to_string() };
    app.log(LogLevel::Info, "Starting route optimization");

    let depot = app.depot_coords;

    let req = crate::core::optimize::OptimizeRequest {
        cache_file: cache_path,
        route_file: app.route_file.clone(),
        turn_penalties: app.turn_penalties.clone(),
        depot,
        oneway_mode: app.oneway_mode,
        mode: crate::core::optimize::SolverMode::Cpp,
        num_vehicles: app.num_vehicles,
        solver_id: app.solver_id.clone(),
    };

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    match rt.block_on(crate::core::optimize::run_optimize(&req)) {
        Ok(result) => {
            app.optimize_status = Status::Done(format!(
                "{:.2} km, {} segments, {:.1}% efficiency",
                result.total_distance_km, result.total_segments, result.efficiency_pct
            ));
            app.log(LogLevel::Success, format!(
                "Optimization complete: {:.2} km, {} segments",
                result.total_distance_km, result.total_segments
            ));

            // Also solve CPP for map visualization (use filtered subset if bbox is set)
            if !app.map_nodes.is_empty() {
                let (solve_nodes, solve_edges) = crate::core::optimize::filter_bbox(
                    &app.map_nodes, &app.map_edges, app.optimize_bbox
                );
                if solve_nodes.is_empty() {
                    app.log(LogLevel::Warn, "No nodes in selected bounding box");
                } else {
                    app.log(LogLevel::Info, format!("BBox filter: {} nodes, {} edges", solve_nodes.len(), solve_edges.len()));
                }
                match crate::core::optimize::solve_cpp(&solve_nodes, &solve_edges, app.oneway_mode, depot) {
                    Ok(cpp) => {
                        app.cpp_output = Some(cpp);
                    }
                    Err(e) => {
                        app.map_solve_error = Some(e.to_string());
                    }
                }
            }
        }
        Err(e) => {
            app.optimize_status = Status::Error(e.to_string());
            app.log(LogLevel::Error, format!("Optimization failed: {}", e));
        }
    }
}
