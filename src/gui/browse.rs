use eframe::egui;

use crate::gui::{GuiApp, LogLevel, View};

pub fn draw_maps(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("🗺️ Cached Maps");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("🔄 Refresh").clicked() {
                app.refresh_cached_maps();
                app.log(LogLevel::Info, "Refreshed cached maps list");
            }
        });

        if app.cached_maps.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "No cached maps found");
            ui.label("Compile a GeoJSON file to create a .rmp map");
        } else {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut to_delete = None;
                let mut to_load = None;
                for (i, name) in app.cached_maps.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let is_selected = app.browse_selection == i;
                        if ui.selectable_label(is_selected, name).clicked() {
                            app.browse_selection = i;
                        }
                        if ui.small_button("📂 Load").clicked() {
                            to_load = Some(name.clone());
                        }
                        if ui.small_button("🗑").clicked() {
                            to_delete = Some(name.clone());
                        }
                    });
                }

                if let Some(name) = to_load {
                    if let Some(cache_dir) = dirs::cache_dir() {
                        let path = cache_dir.join("rmpca").join(&name);
                        app.load_rmp(&path);
                        app.current_view = View::Optimize;
                    }
                }

                if let Some(name) = to_delete {
                    if let Some(cache_dir) = dirs::cache_dir() {
                        let path = cache_dir.join("rmpca").join(&name);
                        if let Err(e) = std::fs::remove_file(&path) {
                            app.log(LogLevel::Error, format!("Failed to delete {}: {}", name, e));
                        } else {
                            app.log(LogLevel::Info, format!("Deleted: {}", name));
                            app.refresh_cached_maps();
                        }
                    }
                }
            });
        }

        // Map canvas if loaded
        if !app.map_nodes.is_empty() {
            ui.separator();
            ui.group(|ui| {
                ui.heading("Map Preview");
                super::draw_map_canvas(ui, app);
            });
        }
    });
}

pub fn draw_routes(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("📋 Saved Routes");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("🔄 Refresh").clicked() {
                app.refresh_saved_routes();
                app.log(LogLevel::Info, "Refreshed saved routes list");
            }
        });

        if app.saved_routes.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(140, 140, 140), "No saved routes found");
            ui.label("Run route optimization to create a saved route");
        } else {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut to_delete = None;
                for (i, name) in app.saved_routes.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let is_selected = app.browse_selection == i;
                        if ui.selectable_label(is_selected, name).clicked() {
                            app.browse_selection = i;
                        }
                        if ui.small_button("🗑").clicked() {
                            to_delete = Some(name.clone());
                        }
                    });
                }

                if let Some(name) = to_delete {
                    if let Some(data_dir) = dirs::data_dir() {
                        let path = data_dir.join("rmpca").join("routes").join(&name);
                        if let Err(e) = std::fs::remove_file(&path) {
                            app.log(LogLevel::Error, format!("Failed to delete {}: {}", name, e));
                        } else {
                            app.log(LogLevel::Info, format!("Deleted: {}", name));
                            app.refresh_saved_routes();
                        }
                    }
                }
            });
        }
    });
}
