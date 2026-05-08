use eframe::egui;

use crate::gui::GuiApp;

const MENU_ITEMS: &[(&str, &str)] = &[
    ("📥", "Extract Data (OSM/Overture)"),
    ("🧹", "Clean GeoJSON (Repair & Optimize)"),
    ("🔨", "Compile Map (GeoJSON → .rmp)"),
    ("🚗", "Optimize Route"),
    ("🚚", "VRP Solver (Multi-Vehicle)"),
    ("🗺️", "Browse Cached Maps"),
    ("📋", "Browse Saved Routes"),
];

pub fn draw(ui: &mut egui::Ui, app: &mut GuiApp) {
    ui.vertical(|ui| {
        ui.heading("Welcome to rmpca");
        ui.label("Route Optimization & Map Processing Agent");
        ui.separator();

        ui.horizontal(|ui| {
            // Left: workflow menu
            ui.vertical(|ui| {
                ui.group(|ui| {
                    ui.heading("Workflow Steps");
                    let views = [
                        super::View::Extract,
                        super::View::Clean,
                        super::View::Compile,
                        super::View::Optimize,
                        super::View::Vrp,
                        super::View::BrowseMaps,
                        super::View::BrowseRoutes,
                    ];
                    for (i, (icon, label)) in MENU_ITEMS.iter().enumerate() {
                        let is_selected = app.current_view == views[i];
                        if ui.selectable_label(is_selected, format!("{} {}", icon, label)).clicked() {
                            app.current_view = views[i].clone();
                            if views[i] == super::View::BrowseMaps {
                                app.refresh_cached_maps();
                            }
                            if views[i] == super::View::BrowseRoutes {
                                app.refresh_saved_routes();
                            }
                        }
                    }
                });

                ui.group(|ui| {
                    ui.heading("Quick Actions");
                    if ui.button("📂 Open .rmp File").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("RMP", &["rmp"])
                            .pick_file()
                        {
                            app.load_rmp(&path);
                        }
                    }
                });
            });

            // Right: recent logs
            ui.vertical(|ui| {
                ui.group(|ui| {
                    ui.heading("Recent Logs");
                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            for entry in app.log_entries.iter().rev().take(50) {
                                ui.horizontal(|ui| {
                                    let level_color = match entry.level {
                                        super::LogLevel::Info => egui::Color32::from_rgb(80, 180, 220),
                                        super::LogLevel::Success => egui::Color32::from_rgb(80, 220, 80),
                                        super::LogLevel::Warn => egui::Color32::from_rgb(220, 200, 60),
                                        super::LogLevel::Error => egui::Color32::from_rgb(220, 80, 80),
                                    };
                                    ui.colored_label(egui::Color32::from_rgb(100, 100, 100), &entry.timestamp);
                                    ui.colored_label(level_color, format!("[{}]", entry.level));
                                    ui.label(&entry.message);
                                });
                            }
                        });
                });
            });
        });
    });
}
