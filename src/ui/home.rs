use ratatui::Frame;

use crate::app::App;

const MENU_ITEMS: &[&str] = &[
    "Extract Data (OSM/Overture)",
    "Clean GeoJSON (Repair & Optimize)",
    "Compile Map (GeoJSON -> .rmp)",
    "Optimize Route",
    "VRP Solver (Multi-Vehicle)",
    "Browse Cached Maps",
    "Browse Saved Routes",
];

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let columns = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints([
            ratatui::layout::Constraint::Percentage(45),
            ratatui::layout::Constraint::Percentage(55),
        ])
        .split(area);

    draw_workflow(f, app, columns[0]);
    draw_expanded_logs(f, app, columns[1]);
}

fn draw_workflow(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" Workflow ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let items: Vec<ratatui::widgets::ListItem> = MENU_ITEMS
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let style = if i == app.workflow_selection {
                ratatui::style::Style::default()
                    .fg(ratatui::style::Color::Yellow)
                    .add_modifier(ratatui::style::Modifier::BOLD)
            } else {
                ratatui::style::Style::default().fg(ratatui::style::Color::White)
            };
            let prefix = if i == app.workflow_selection {
                " > "
            } else {
                "   "
            };
            ratatui::widgets::ListItem::new(ratatui::text::Span::styled(
                format!("{}{}", prefix, item),
                style,
            ))
        })
        .collect();

    let list = ratatui::widgets::List::new(items);
    f.render_widget(list, inner);
}

fn draw_expanded_logs(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" Logs ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let visible = inner.height as usize;
    let entries: Vec<ratatui::text::Line> = app
        .log_entries
        .iter()
        .rev()
        .take(visible)
        .rev()
        .map(|entry| {
            let ts_style = ratatui::style::Style::default().fg(ratatui::style::Color::Cyan);
            let level_style = match entry.level {
                crate::app::LogLevel::Info => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Cyan)
                }
                crate::app::LogLevel::Success => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Green)
                }
                crate::app::LogLevel::Warn => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Yellow)
                }
                crate::app::LogLevel::Error => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Red)
                }
            };
            ratatui::text::Line::from(vec![
                ratatui::text::Span::styled(format!("{} ", entry.timestamp), ts_style),
                ratatui::text::Span::styled(format!("[{}] ", entry.level), level_style),
                ratatui::text::Span::raw(entry.message.clone()),
            ])
        })
        .collect();

    let paragraph = ratatui::widgets::Paragraph::new(entries);
    f.render_widget(paragraph, inner);
}
