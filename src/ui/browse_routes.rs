use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    if app.saved_routes.is_empty() {
        super::draw_empty_placeholder(
            f,
            area,
            "Saved Routes (0)",
            "No saved routes found",
            "Run route optimization to create a saved route",
        );
    } else {
        super::draw_selectable_list(
            f,
            area,
            "Saved Routes",
            &app.saved_routes,
            app.browse_selection,
        );
    }
}
