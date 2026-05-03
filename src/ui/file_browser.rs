use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let browser = match &app.file_browser {
        Some(b) => b,
        None => return,
    };

    let block = Block::default()
        .title(" File Browser ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Split into header, file list, and footer
    let chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(3), // header
            ratatui::layout::Constraint::Min(5),    // file list
            ratatui::layout::Constraint::Length(3), // footer
        ])
        .split(inner);

    // Header: current path and filter
    draw_header(f, browser, chunks[0]);

    // File list
    draw_file_list(f, browser, chunks[1]);

    // Footer: instructions
    draw_footer(f, browser, chunks[2]);
}

fn draw_header(f: &mut Frame, browser: &crate::app::FileBrowser, area: Rect) {
    let path_display = browser.current_path.to_string_lossy();
    let filter_display = browser.filter.description();

    let lines = vec![
        Line::from(vec![
            Span::styled("Path: ", Style::default().fg(Color::Cyan)),
            Span::raw(path_display.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Filter: ", Style::default().fg(Color::Cyan)),
            Span::styled(filter_display, Style::default().fg(Color::Yellow)),
        ]),
    ];

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, area);
}

fn draw_file_list(f: &mut Frame, browser: &crate::app::FileBrowser, area: Rect) {
    if browser.entries.is_empty() {
        let empty_msg = Paragraph::new(Line::from(Span::styled(
            "No files found",
            Style::default().fg(Color::DarkGray),
        )));
        f.render_widget(empty_msg, area);
        return;
    }

    let items: Vec<ListItem> = browser
        .entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let icon = if entry.is_dir { "📁" } else { "📄" };

            let size_str = if let Some(size) = entry.size {
                format_size(size)
            } else {
                String::new()
            };

            let modified_str = if let Some(modified) = entry.modified {
                if let Ok(duration) = modified.elapsed() {
                    format_duration(duration)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let name_style = if idx == browser.selection {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else if entry.is_dir {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default().fg(Color::White)
            };

            let line = if entry.is_dir {
                Line::from(vec![
                    Span::raw(format!("{} ", icon)),
                    Span::styled(&entry.name, name_style),
                ])
            } else {
                Line::from(vec![
                    Span::raw(format!("{} ", icon)),
                    Span::styled(&entry.name, name_style),
                    Span::raw("  "),
                    Span::styled(size_str, Style::default().fg(Color::DarkGray)),
                    Span::raw("  "),
                    Span::styled(modified_str, Style::default().fg(Color::DarkGray)),
                ])
            };

            ListItem::new(line)
        })
        .collect();

    let list = List::new(items);
    f.render_widget(list, area);
}

fn draw_footer(f: &mut Frame, browser: &crate::app::FileBrowser, area: Rect) {
    let target_desc = match browser.target_field {
        crate::app::InputField::InputFile => "Select input file",
        crate::app::InputField::OutputFile => "Select output file",
        crate::app::InputField::CacheFile => "Select cache file",
        crate::app::InputField::RouteFile => "Select route file",
        _ => "Select file",
    };

    let lines = vec![
        Line::from(Span::styled(
            target_desc,
            Style::default().fg(Color::Yellow),
        )),
        Line::from(vec![
            Span::raw("[↑↓] Navigate  "),
            Span::raw("[Enter] Select  "),
            Span::raw("[Backspace] Parent  "),
            Span::raw("[h] Toggle hidden  "),
            Span::raw("[Esc] Cancel"),
        ]),
    ];

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, area);
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();

    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else if secs < 604800 {
        format!("{}d ago", secs / 86400)
    } else {
        format!("{}w ago", secs / 604800)
    }
}
