use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    if app.cached_maps.is_empty() {
        super::draw_empty_placeholder(
            f,
            area,
            "Cached Maps (0)",
            "No cached maps found",
            "Compile a GeoJSON file to create a .rmp map",
        );
    } else {
        super::draw_selectable_list(f, area, "Cached Maps", &app.cached_maps, app.browse_selection);
    }
}
