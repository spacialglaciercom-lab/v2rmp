use eframe::egui;

use crate::gui::{GuiApp, DataSource, BoundingBox, LogLevel, Status};

pub fn draw(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("📥 Extract Data");
        ui.label("Extract road network data from OSM or Overture Maps");
        ui.separator();

        // Data source selection
        ui.group(|ui| {
            ui.heading("Data Source");
            ui.horizontal(|ui| {
                ui.radio_value(&mut app.data_source, DataSource::Osm, "OpenStreetMap (OSM)");
                ui.radio_value(&mut app.data_source, DataSource::Overture, "Overture Maps");
            });
        });

        // Bounding box
        ui.group(|ui| {
            ui.heading("Bounding Box");
            if let Some(ref bbox) = app.bounding_box {
                ui.colored_label(
                    egui::Color32::from_rgb(80, 220, 80),
                    format!("Set: {}", bbox),
                );
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 200, 60),
                    "Not set — enter coordinates below",
                );
            }
            ui.horizontal(|ui| {
                ui.label("min_lat,min_lon,max_lat,max_lon:");
                let response = ui.text_edit_singleline(&mut app.bbox_input);
                if ui.button("Set BBox").clicked() || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    if let Some(bbox) = parse_bbox(&app.bbox_input) {
                        app.bounding_box = Some(bbox.clone());
                        app.log(LogLevel::Success, format!("Bounding box set: {}", bbox));
                    } else {
                        app.log(LogLevel::Error, "Invalid bounding box format. Use: min_lon,min_lat,max_lon,max_lat");
                    }
                }
            });
            if ui.button("Clear BBox").clicked() {
                app.bounding_box = None;
                app.bbox_input.clear();
                app.log(LogLevel::Info, "Bounding box cleared");
            }
        });

        // PBF file (OSM only)
        if app.data_source == DataSource::Osm {
            ui.group(|ui| {
                ui.heading("OSM PBF File (optional)");
                ui.label("If you have a local .osm.pbf file, select it instead of downloading.");
                if ui.button("📂 Browse PBF…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("OSM PBF", &["pbf"])
                        .pick_file()
                    {
                        app.log(LogLevel::Info, format!("PBF selected: {}", path.display()));
                    }
                }
            });
        }

        // Extract button
        ui.group(|ui| {
            super::status_label(ui, &app.extract_status);

            let can_run = app.bounding_box.is_some();
            if ui.add_enabled(can_run, egui::Button::new("🚀 Start Extraction")).clicked() {
                run_extract(app);
            }
        });
    });
}

fn parse_bbox(input: &str) -> Option<BoundingBox> {
    let parts: Vec<&str> = input.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let min_lon = parts[0].trim().parse::<f64>().ok()?;
    let min_lat = parts[1].trim().parse::<f64>().ok()?;
    let max_lon = parts[2].trim().parse::<f64>().ok()?;
    let max_lat = parts[3].trim().parse::<f64>().ok()?;
    if min_lat >= max_lat || min_lon >= max_lon {
        return None;
    }
    Some(BoundingBox {
        min_lat,
        min_lon,
        max_lat,
        max_lon,
    })
}

fn run_extract(app: &mut GuiApp) {
    let bbox = match &app.bounding_box {
        Some(b) => b.clone(),
        None => {
            app.log(LogLevel::Warn, "Set a bounding box first");
            return;
        }
    };

    app.extract_status = Status::Running {
        progress: 0,
        message: "Starting extraction…".to_string(),
    };
    app.log(LogLevel::Info, format!("Starting extraction from {}", app.data_source));

    let source = match app.data_source {
        DataSource::Osm => crate::core::extract::ExtractSource::Osm,
        DataSource::Overture => crate::core::extract::ExtractSource::Overture,
    };

    let req = crate::core::extract::ExtractRequest {
        source,
        bbox: crate::core::extract::BBoxRequest {
            min_lon: bbox.min_lon,
            min_lat: bbox.min_lat,
            max_lon: bbox.max_lon,
            max_lat: bbox.max_lat,
        },
        road_classes: crate::core::extract::RoadClass::all_vehicle(),
        output_path: format!("extract_{:.4}_{:.4}.geojson", bbox.min_lat, bbox.min_lon),
        pbf_path: None,
    };

    // Run extraction synchronously (blocking the UI — could be moved to a thread)
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    match rt.block_on(crate::core::extract::run_extract(&req)) {
        Ok(result) => {
            app.extract_status = Status::Done(format!("{} nodes, {} edges", result.nodes, result.edges));
            app.log(LogLevel::Success, format!(
                "Extraction complete: {} nodes, {} edges",
                result.nodes,
                result.edges
            ));
        }
        Err(e) => {
            app.extract_status = Status::Error(e.to_string());
            app.log(LogLevel::Error, format!("Extraction failed: {}", e));
        }
    }
}
