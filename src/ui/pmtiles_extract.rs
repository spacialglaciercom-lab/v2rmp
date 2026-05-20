use ratatui::Frame;

use crate::app::{App, Status};

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let columns = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Horizontal)
        .constraints([
            ratatui::layout::Constraint::Percentage(55),
            ratatui::layout::Constraint::Percentage(45),
        ])
        .split(area);

    draw_form(f, app, columns[0]);
    draw_logs(f, app, columns[1]);
}

fn draw_form(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" PMTiles Extract ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Green));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(2), // PMTiles file
            ratatui::layout::Constraint::Length(2), // Bounding box
            ratatui::layout::Constraint::Length(2), // Output path
            ratatui::layout::Constraint::Length(2), // Layer name
            ratatui::layout::Constraint::Length(2), // Zoom
            ratatui::layout::Constraint::Length(1), // spacer
            ratatui::layout::Constraint::Length(2), // Status
            ratatui::layout::Constraint::Min(0),    // Keybindings
        ])
        .split(inner);

    let label_style = ratatui::style::Style::default().fg(ratatui::style::Color::Cyan);
    let value_style = ratatui::style::Style::default().fg(ratatui::style::Color::White);
    let dim_style = ratatui::style::Style::default().fg(ratatui::style::Color::DarkGray);

    // PMTiles file
    let pmtiles_val = app.pmtiles_path.as_deref().unwrap_or("<not set>");
    let pmtiles_line = ratatui::text::Line::from(vec![
        ratatui::text::Span::styled(" File:   ", label_style),
        ratatui::text::Span::styled(pmtiles_val, value_style),
    ]);
    f.render_widget(ratatui::widgets::Paragraph::new(pmtiles_line), chunks[0]);

    // Bounding box
    let bbox_val = app.pmtiles_bbox.as_ref().map(|b| format!("{:.4},{:.4},{:.4},{:.4}", b.min_lon, b.min_lat, b.max_lon, b.max_lat)).unwrap_or_else(|| "<not set>".to_string());
    let bbox_line = ratatui::text::Line::from(vec![
        ratatui::text::Span::styled(" BBox:   ", label_style),
        ratatui::text::Span::styled(&bbox_val, value_style),
    ]);
    f.render_widget(ratatui::widgets::Paragraph::new(bbox_line), chunks[1]);

    // Output path
    let output_val = app.pmtiles_output_path.as_deref().unwrap_or("<auto>");
    let output_line = ratatui::text::Line::from(vec![
        ratatui::text::Span::styled(" Output: ", label_style),
        ratatui::text::Span::styled(output_val, value_style),
    ]);
    f.render_widget(ratatui::widgets::Paragraph::new(output_line), chunks[2]);

    // Layer name
    let layer_val = app.pmtiles_layer_name.as_deref().unwrap_or("<all layers>");
    let layer_line = ratatui::text::Line::from(vec![
        ratatui::text::Span::styled(" Layer:  ", label_style),
        ratatui::text::Span::styled(layer_val, value_style),
    ]);
    f.render_widget(ratatui::widgets::Paragraph::new(layer_line), chunks[3]);

    // Zoom
    let zoom_val = app.pmtiles_zoom.map(|z| z.to_string()).unwrap_or_else(|| "auto (max zoom)".to_string());
    let zoom_line = ratatui::text::Line::from(vec![
        ratatui::text::Span::styled(" Zoom:   ", label_style),
        ratatui::text::Span::styled(&zoom_val, value_style),
    ]);
    f.render_widget(ratatui::widgets::Paragraph::new(zoom_line), chunks[4]);

    // Status
    let status_text = match &app.pmtiles_status {
        Status::Ready => "Ready".to_string(),
        Status::Running { progress, message } => format!("{}% - {}", progress, message),
        Status::Done(msg) => format!("✓ {}", msg),
        Status::Error(msg) => format!("✗ {}", msg),
    };
    let status_style = match &app.pmtiles_status {
        Status::Ready => dim_style,
        Status::Running { .. } => ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
        Status::Done(_) => ratatui::style::Style::default().fg(ratatui::style::Color::Green),
        Status::Error(_) => ratatui::style::Style::default().fg(ratatui::style::Color::Red),
    };
    let status_line = ratatui::text::Line::from(vec![
        ratatui::text::Span::styled(" Status: ", label_style),
        ratatui::text::Span::styled(status_text, status_style),
    ]);
    f.render_widget(ratatui::widgets::Paragraph::new(status_line), chunks[6]);

    // Keybindings
    let keys = vec![
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(" [f]", dim_style),
            ratatui::text::Span::raw(" File  "),
            ratatui::text::Span::styled("[b]", dim_style),
            ratatui::text::Span::raw(" BBox  "),
            ratatui::text::Span::styled("[o]", dim_style),
            ratatui::text::Span::raw(" Output  "),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(" [l]", dim_style),
            ratatui::text::Span::raw(" Layer "),
            ratatui::text::Span::styled("[z]", dim_style),
            ratatui::text::Span::raw(" Zoom  "),
            ratatui::text::Span::styled("[Enter]", dim_style),
            ratatui::text::Span::raw(" Extract "),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::styled(" [Esc]", dim_style),
            ratatui::text::Span::raw(" Back"),
        ]),
    ];
    let keys_widget = ratatui::widgets::Paragraph::new(keys);
    f.render_widget(keys_widget, chunks[7]);
}

fn draw_logs(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
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
                crate::app::LogLevel::Info => ratatui::style::Style::default().fg(ratatui::style::Color::Cyan),
                crate::app::LogLevel::Success => ratatui::style::Style::default().fg(ratatui::style::Color::Green),
                crate::app::LogLevel::Warn => ratatui::style::Style::default().fg(ratatui::style::Color::Yellow),
                crate::app::LogLevel::Error => ratatui::style::Style::default().fg(ratatui::style::Color::Red),
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
