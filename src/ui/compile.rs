use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" Compile Map ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let cyan = ratatui::style::Color::Cyan;
    let yellow = ratatui::style::Color::Yellow;
    let gray = ratatui::style::Color::DarkGray;
    let green = ratatui::style::Color::Green;
    let magenta = ratatui::style::Color::Magenta;

    let input_display = match &app.input_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(not set - press 'i')".to_string(),
            ratatui::style::Style::default().fg(yellow),
        ),
    };

    let output_display = match &app.output_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(auto-derived from input)".to_string(),
            ratatui::style::Style::default().fg(gray),
        ),
    };

    let status_text = format!("Status: {}", app.compile_status);
    let status_color = match &app.compile_status {
        crate::app::Status::Ready => gray,
        crate::app::Status::Running { .. } => magenta,
        crate::app::Status::Done(_) => green,
        crate::app::Status::Error(_) => ratatui::style::Color::Red,
    };

    let lines = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Compile Map",
            ratatui::style::Style::default()
                .fg(cyan)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from("Convert GeoJSON -> binary .rmp format for instant loading"),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![ratatui::text::Span::raw("Input:  "), input_display]),
        ratatui::text::Line::from(vec![ratatui::text::Span::raw("Output: "), output_display]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::raw(
            "────────────────────────────────────",
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Controls:",
            ratatui::style::Style::default().fg(cyan),
        )),
        ratatui::text::Line::from("  [I]      Set input GeoJSON file path"),
        ratatui::text::Line::from("  [O]      Set output .rmp file path (optional)"),
        ratatui::text::Line::from("  [Enter]  Start compilation"),
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
