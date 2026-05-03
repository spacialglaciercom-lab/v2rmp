use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let count = app.saved_routes.len();
    let block = ratatui::widgets::Block::default()
        .title(format!(" Saved Routes ({}) ", count))
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.saved_routes.is_empty() {
        let lines = vec![
            ratatui::text::Line::from(""),
            ratatui::text::Line::from("No saved routes found"),
            ratatui::text::Line::from(""),
            ratatui::text::Line::from(ratatui::text::Span::styled(
                "Run route optimization to create a saved route",
                ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray),
            )),
            ratatui::text::Line::from(""),
            ratatui::text::Line::from(ratatui::text::Span::styled(
                "(press Esc to return home)",
                ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray),
            )),
        ];
        let paragraph = ratatui::widgets::Paragraph::new(lines);
        f.render_widget(paragraph, inner);
    } else {
        let items: Vec<ratatui::widgets::ListItem> = app
            .saved_routes
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let style = if i == app.browse_selection {
                    ratatui::style::Style::default()
                        .fg(ratatui::style::Color::Yellow)
                        .add_modifier(ratatui::style::Modifier::BOLD)
                } else {
                    ratatui::style::Style::default().fg(ratatui::style::Color::White)
                };
                let prefix = if i == app.browse_selection {
                    " > "
                } else {
                    "   "
                };
                ratatui::widgets::ListItem::new(ratatui::text::Span::styled(
                    format!("{}{}", prefix, name),
                    style,
                ))
            })
            .collect();

        let list = ratatui::widgets::List::new(items);
        f.render_widget(list, inner);
    }
}
