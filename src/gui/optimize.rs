use eframe::egui;

use crate::gui::{GuiApp, LogLevel, Status};

pub fn draw(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("🔄 Chinese Postman Problem (CPP)");
        ui.label("Find a route that traverses every street edge at least once.");
        ui.colored_label(
            egui::Color32::from_rgb(140, 140, 200),
            "CPP covers all edges — use VRP Solver to visit specific stops instead.",
        );
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
                ui.colored_label(
                    egui::Color32::from_rgb(80, 220, 80),
                    format!(
                        "{:.4},{:.4} to {:.4},{:.4}",
                        min_lat, min_lon, max_lat, max_lon
                    ),
                );
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(140, 140, 140),
                    "(full map — no filter)",
                );
            }
            ui.horizontal(|ui| {
                ui.label("min_lon,min_lat,max_lon,max_lat:");
                let _resp = ui.text_edit_singleline(&mut app.optimize_bbox_input);
                if ui.button("Set BBox").clicked() {
                    let parts: Vec<&str> = app.optimize_bbox_input.split(',').collect();
                    if parts.len() == 4 {
                        if let (Ok(mlo), Ok(mla), Ok(xlo), Ok(xla)) = (
                            parts[0].trim().parse::<f64>(),
                            parts[1].trim().parse::<f64>(),
                            parts[2].trim().parse::<f64>(),
                            parts[3].trim().parse::<f64>(),
                        ) {
                            app.optimize_bbox = Some((mla, xla, mlo, xlo));
                            app.log(
                                LogLevel::Success,
                                format!(
                                    "BBox filter set: {:.4},{:.4} to {:.4},{:.4}",
                                    mlo, mla, xlo, xla
                                ),
                            );
                        } else {
                            app.log(LogLevel::Error, "Invalid coordinates");
                        }
                    } else {
                        app.log(
                            LogLevel::Error,
                            "Use format: min_lon,min_lat,max_lon,max_lat",
                        );
                    }
                }
                if ui.button("✕ Clear").clicked() {
                    app.optimize_bbox = None;
                    app.optimize_bbox_input.clear();
                    app.log(LogLevel::Info, "BBox filter cleared — using full map");
                }
            });
        });

        // GPX output (auto-generated)
        ui.group(|ui| {
            ui.heading("Output");
            if let Some(ref path) = app.cache_file {
                let gpx_path = path.replace(".rmp", "_cpp.gpx");
                ui.colored_label(
                    egui::Color32::from_rgb(80, 220, 80),
                    format!("GPX → {}", gpx_path),
                );
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(140, 140, 140),
                    "(load .rmp to see output path)",
                );
            }
            if let Some(ref path) = app.route_file {
                ui.colored_label(
                    egui::Color32::from_rgb(80, 220, 80),
                    format!("JSON → {}", path),
                );
            }
        });

        // Turn penalties
        ui.group(|ui| {
            ui.heading("Turn Penalties");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut app.turn_penalties.left)
                        .speed(0.1)
                        .range(0.0..=60.0),
                );
                ui.label("Left turn (s)");
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut app.turn_penalties.right)
                        .speed(0.1)
                        .range(0.0..=60.0),
                );
                ui.label("Right turn (s)");
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut app.turn_penalties.u_turn)
                        .speed(0.1)
                        .range(0.0..=60.0),
                );
                ui.label("U-turn (s)");
            });
        });

        // Depot
        ui.group(|ui| {
            ui.heading("Depot (optional)");
            ui.label("Starting point for the CPP tour.");
            if let Some((lat, lon)) = app.depot_coords {
                ui.colored_label(
                    egui::Color32::from_rgb(80, 220, 80),
                    format!("Depot: {:.4},{:.4}", lat, lon),
                );
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(140, 140, 140),
                    "(not set — arbitrary start node)",
                );
            }
            ui.horizontal(|ui| {
                ui.label("Depot (lat,lon):");
                let _response = ui.text_edit_singleline(&mut app.map_depot_text);
                if ui.button("Set Depot").clicked() {
                    let parts: Vec<&str> = app.map_depot_text.split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(lat), Ok(lon)) = (
                            parts[0].trim().parse::<f64>(),
                            parts[1].trim().parse::<f64>(),
                        ) {
                            app.depot_coords = Some((lat, lon));
                            app.log(
                                LogLevel::Success,
                                format!("Depot set: {:.4},{:.4}", lat, lon),
                            );
                        }
                    }
                }
            });
        });

        // One-way mode
        ui.group(|ui| {
            ui.heading("One-way Streets");
            ui.horizontal(|ui| {
                ui.label("Mode:");
                egui::ComboBox::from_id_salt("oneway_select")
                    .selected_text(format!("{:?}", app.oneway_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app.oneway_mode,
                            crate::core::optimize::OnewayMode::Ignore,
                            "Ignore",
                        );
                        ui.selectable_value(
                            &mut app.oneway_mode,
                            crate::core::optimize::OnewayMode::Respect,
                            "Respect",
                        );
                        ui.selectable_value(
                            &mut app.oneway_mode,
                            crate::core::optimize::OnewayMode::Reverse,
                            "Reverse",
                        );
                    });
            });
        });

        // Run CPP
        ui.group(|ui| {
            super::status_label(ui, &app.optimize_status);
            let can_run = app.cache_file.is_some();
            if ui
                .add_enabled(can_run, egui::Button::new("🔄 Run CPP"))
                .clicked()
            {
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

    app.optimize_status = Status::Running {
        progress: 0,
        message: "Running CPP\u{2026}".to_string(),
    };
    app.log(LogLevel::Info, "Starting Chinese Postman optimization");

    // Read the RMP file
    let file_data = match std::fs::read(&cache_path) {
        Ok(d) => d,
        Err(e) => {
            app.optimize_status = Status::Error(e.to_string());
            app.log(LogLevel::Error, format!("Failed to read cache file: {}", e));
            return;
        }
    };
    let (mut nodes, mut edges) = match crate::core::optimize::read_rmp_file(&file_data) {
        Ok(n) => n,
        Err(e) => {
            app.optimize_status = Status::Error(e.to_string());
            app.log(LogLevel::Error, format!("Invalid .rmp file: {}", e));
            return;
        }
    };

    let depot = app.depot_coords;

    // Filter by bounding box if set
    if app.optimize_bbox.is_some() {
        let (filtered_nodes, filtered_edges) =
            crate::core::optimize::filter_bbox(&nodes, &edges, app.optimize_bbox);
        if filtered_nodes.is_empty() {
            app.optimize_status = Status::Error("No nodes in bounding box".to_string());
            app.log(LogLevel::Error, "Bounding box contains no nodes");
            return;
        }
        app.log(
            LogLevel::Info,
            format!(
                "BBox filter: {} of {} nodes, {} of {} edges",
                filtered_nodes.len(),
                nodes.len(),
                filtered_edges.len(),
                edges.len()
            ),
        );
        nodes = filtered_nodes;
        edges = filtered_edges;
    }

    // Solve CPP on the (possibly filtered) graph
    match crate::core::optimize::solve_cpp(&nodes, &edges, app.oneway_mode, depot) {
        Ok(cpp) => {
            app.optimize_status = Status::Done(format!(
                "{:.2} km, {} segments, {:.1}% efficiency",
                cpp.summary.total_distance_km,
                cpp.summary.total_segments,
                cpp.summary.efficiency_pct
            ));
            app.log(
                LogLevel::Success,
                format!(
                    "CPP complete: {:.2} km, {} segments, {:.2} km deadhead",
                    cpp.summary.total_distance_km,
                    cpp.summary.total_segments,
                    cpp.summary.deadhead_distance_km
                ),
            );

            // Write GPX output
            let gpx_path = cache_path.replace(".rmp", "_cpp.gpx");
            match crate::core::optimize::write_gpx_cpp(&gpx_path, &nodes, &cpp.circuit) {
                Ok(_) => {
                    app.log(LogLevel::Success, format!("GPX written: {}", gpx_path));
                }
                Err(e) => {
                    app.log(LogLevel::Warn, format!("Failed to write GPX: {}", e));
                }
            }

            // Also write JSON if a route file was set
            if let Some(ref route_path) = app.route_file {
                let route_json = serde_json::json!({
                    "route": cpp.circuit,
                    "total_distance_km": cpp.summary.total_distance_km,
                    "deadhead_distance_km": cpp.summary.deadhead_distance_km,
                    "efficiency_pct": cpp.summary.efficiency_pct,
                    "nodes": nodes.iter().enumerate().map(|(i, n)| serde_json::json!({"id": i, "lat": n.lat, "lon": n.lon})).collect::<Vec<_>>(),
                });
                match std::fs::write(
                    route_path,
                    serde_json::to_string_pretty(&route_json).unwrap_or_default(),
                ) {
                    Ok(_) => app.log(
                        LogLevel::Success,
                        format!("Route JSON written: {}", route_path),
                    ),
                    Err(e) => app.log(LogLevel::Warn, format!("Failed to write route JSON: {}", e)),
                }
            }

            // Update map visualization with filtered data
            app.cpp_output = Some(cpp);
            app.map_solve_error = None;
            app.set_network(nodes, edges, cache_path);
        }
        Err(e) => {
            app.optimize_status = Status::Error(e.to_string());
            app.log(LogLevel::Error, format!("CPP failed: {}", e));
        }
    }
}
