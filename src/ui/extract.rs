use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" Extract Data ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let cyan = ratatui::style::Color::Cyan;
    let yellow = ratatui::style::Color::Yellow;
    let green = ratatui::style::Color::Green;
    let gray = ratatui::style::Color::DarkGray;
    let magenta = ratatui::style::Color::Magenta;

    let bbox_text = match &app.bounding_box {
        Some(bb) => format!("Set: {}", bb),
        None => "Not set (press 'b' to set)".to_string(),
    };

    let status_text = format!("Status: {}", app.extract_status);
    let status_color = match &app.extract_status {
        crate::app::Status::Ready => gray,
        crate::app::Status::Running { .. } => magenta,
        crate::app::Status::Done(_) => green,
        crate::app::Status::Error(_) => ratatui::style::Color::Red,
    };

    let lines = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Extract Data",
            ratatui::style::Style::default()
                .fg(cyan)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Data Source: "),
            ratatui::text::Span::styled(
                app.data_source.to_string(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Bounding Box: "),
            ratatui::text::Span::styled(bbox_text, ratatui::style::Style::default().fg(green)),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::raw(
            "────────────────────────────────────",
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Controls:",
            ratatui::style::Style::default().fg(cyan),
        )),
        ratatui::text::Line::from("  [Tab/S]  Toggle data source (OSM <-> Overture)"),
        ratatui::text::Line::from("  [B]      Set bounding box (enter coordinates)"),
        ratatui::text::Line::from("  [Enter]  Start extraction"),
        ratatui::text::Line::from("  [Esc]    Return to home"),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            status_text,
            ratatui::style::Style::default().fg(status_color),
        )),
    ];

    let paragraph = ratatui::widgets::Paragraph::new(lines);
    f.render_widget(paragraph, inner);

    super::draw_input_prompt(f, app, area);
}
