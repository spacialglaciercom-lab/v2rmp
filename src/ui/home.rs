use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use crate::app::App;

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(10)])
        .split(f.area());
    
    let title = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {} ", module_name()));
    
    let content = Paragraph::new("Module under construction")
        .block(title);
    
    f.render_widget(content, chunks[0]);
    
    render_logs(f, app, chunks[1]);
}

fn module_name() -> &'static str {
    env!("CARGO_PKG_NAME")
}

fn render_logs(f: &mut Frame, app: &App, area: Rect) {
    let logs: Vec<ListItem> = app
        .log_entries
        .iter()
        .rev()
        .take(area.height as usize - 2)
        .rev()
        .map(|entry| {
            let style = match entry.level {
                crate::app::LogLevel::Info => Style::default().fg(Color::Cyan),
                crate::app::LogLevel::Success => Style::default().fg(Color::Green),
                crate::app::LogLevel::Warn => Style::default().fg(Color::Yellow),
                crate::app::LogLevel::Error => Style::default().fg(Color::Red),
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("[{}] ", entry.timestamp), Style::default().fg(Color::DarkGray)),
                Span::styled(&entry.message, style),
            ]))
        })
        .collect();

    let logs_widget = List::new(logs)
        .block(Block::default().borders(Borders::ALL).title(" Logs "));
    
    f.render_widget(logs_widget, area);
}
