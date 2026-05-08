use eframe::egui;

pub fn draw(ui: &mut egui::Ui) {
    ui.vertical(|ui| {
        ui.heading("📖 Help — rmpca Route Optimizer");
        ui.separator();

        ui.group(|ui| {
            ui.heading("Keyboard Shortcuts");
            ui.label("Esc — Return to Home");
            ui.label("q — Quit (TUI mode only)");
            ui.label("F1 / h — Toggle Help");
        });

        ui.group(|ui| {
            ui.heading("Mouse Controls (Map)");
            ui.label("Scroll — Zoom in/out");
            ui.label("Click & Drag — Pan the map");
            ui.label("Double-click — Reset view");
        });

        ui.group(|ui| {
            ui.heading("Workflow Steps");
            ui.label("1. Extract road data from OSM or Overture");
            ui.label("2. Clean GeoJSON (repair, dedupe, simplify)");
            ui.label("3. Compile GeoJSON into binary .rmp format");
            ui.label("4. Optimize route with turn penalties");
            ui.label("5. VRP multi-vehicle solver");
            ui.label("6. Browse cached maps and saved routes");
        });

        ui.group(|ui| {
            ui.heading("Core Algorithms");
            ui.label("CPP (Eulerian circuit) — covers all edges");
            ui.label("TSP (2-opt) — optimizes stop visits");
            ui.label("VRP (OR-Tools style) — multi-vehicle routing");
            ui.label("Haversine distance, bearing-based turns");
        });

        ui.group(|ui| {
            ui.heading("File Formats");
            ui.label(".geojson — Road network input (GeoJSON FeatureCollection)");
            ui.label(".rmp — Compiled binary map format (fast loading)");
            ui.label(".osm.pbf — OpenStreetMap Protobuf extract");
        });
    });
}
