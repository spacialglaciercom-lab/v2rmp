use ratatui::Frame;

use crate::app::{App, Status};

#[allow(dead_code)]
pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" Clean GeoJSON ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let cyan = ratatui::style::Color::Cyan;
    let yellow = ratatui::style::Color::Yellow;
    let green = ratatui::style::Color::Green;
    let gray = ratatui::style::Color::DarkGray;
    let white = ratatui::style::Color::White;

    let opts = &app.clean_options;
    let sel = app.clean_selection;

    let toggles: Vec<(usize, &str, bool)> = vec![
        (0, "Make valid geometries", opts.make_valid),
        (1, "Drop invalid features", opts.drop_invalid),
        (2, "Remove self-loops", opts.remove_selfloops),
        (3, "Dedupe edges", opts.dedupe_edges),
        (4, "Remove isolates", opts.remove_isolates),
        (5, "Merge node positions", opts.merge_node_positions),
        (6, "Include polygons", opts.include_polygons),
        (7, "Include points", opts.include_points),
        (8, "Merge parallel edges", opts.merge_parallel_edges),
        (
            9,
            "Merge parallel edge properties",
            opts.merge_parallel_edge_properties,
        ),
    ];

    let mut lines: Vec<ratatui::text::Line> = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Clean GeoJSON",
            ratatui::style::Style::default()
                .fg(cyan)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from("Repair geometry, remove duplicates, keep largest component."),
        ratatui::text::Line::from(""),
    ];

    let input_display = match &app.clean_input_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(not set - press 'i')".to_string(),
            ratatui::style::Style::default().fg(yellow),
        ),
    };
    let output_display = match &app.clean_output_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(auto-derived from input)".to_string(),
            ratatui::style::Style::default().fg(gray),
        ),
    };

    lines.push(ratui_line(vec![span_raw("Input:   "), input_display]));
    lines.push(ratui_line(vec![span_raw("Output:  "), output_display]));
    lines.push(ratatui::text::Line::from(""));

    lines.push(ratui_line(vec![
        span_raw("  Min length (m): "),
        span_val(format!("{:.1}", opts.min_length_m), sel == 10),
        span_raw("   Node snap (m): "),
        span_val(format!("{:.1}", opts.node_snap_m), sel == 11),
    ]));
    lines.push(ratui_line(vec![
        span_raw("  Max components: "),
        span_val(format!("{}", opts.max_components), sel == 12),
        span_raw("   Simplify tol (m): "),
        span_val(format!("{:.1}", opts.simplify_tolerance_m), sel == 13),
    ]));
    lines.push(ratui_line(vec![
        span_raw("  Precision decimals: "),
        span_val(format!("{}", opts.node_precision_decimals), sel == 14),
    ]));
    lines.push(ratatui::text::Line::from(""));

    lines.push(ratatui::text::Line::from("  Options:"));
    for (idx, label, val) in &toggles {
        let check = if *val { "[x]" } else { "[ ]" };
        let style = if *idx == sel {
            ratatui::style::Style::default()
                .fg(yellow)
                .add_modifier(ratatui::style::Modifier::BOLD)
        } else {
            ratatui::style::Style::default().fg(white)
        };
        lines.push(ratatui::text::Line::from(ratatui::text::Span::styled(
            format!("    {} {}  ", check, label),
            style,
        )));
    }
    lines.push(ratatui::text::Line::from(""));

    let status_text = format!("Status: {}", app.clean_status);
    let status_color = match &app.clean_status {
        Status::Ready => gray,
        Status::Running { .. } => ratatui::style::Color::Magenta,
        Status::Done(_) => green,
        Status::Error(_) => ratatui::style::Color::Red,
    };
    lines.push(ratui_line(vec![ratatui::text::Span::styled(
        status_text,
        ratatui::style::Style::default().fg(status_color),
    )]));

    if let Status::Done(ref msg) = app.clean_status {
        if !msg.is_empty() {
            lines.push(ratatui::text::Line::from(""));
            for line in msg.lines() {
                lines.push(ratatui::text::Line::from(format!("  {}", line)));
            }
        }
    }

    lines.push(ratatui::text::Line::from(""));
    lines.push(ratui_line(vec![ratatui::text::Span::styled(
        "Controls: ",
        ratatui::style::Style::default().fg(cyan),
    )]));
    lines.push(ratatui::text::Line::from(
        "  [Space] Toggle option    [Enter] Run clean",
    ));
    lines.push(ratatui::text::Line::from(
        "  [I] Set input file       [O] Set output file",
    ));
    lines.push(ratatui::text::Line::from(
        "  [Esc] Return to home     [R] Reset to defaults",
    ));

    let paragraph = ratatui::widgets::Paragraph::new(lines);
    f.render_widget(paragraph, inner);

    super::draw_input_prompt(f, app, area);
}

#[allow(dead_code)]
fn span_raw(s: &str) -> ratatui::text::Span<'static> {
    ratatui::text::Span::raw(s.to_string())
}

#[allow(dead_code)]
fn span_val(s: String, highlighted: bool) -> ratatui::text::Span<'static> {
    if highlighted {
        ratatui::text::Span::styled(
            s,
            ratatui::style::Style::default()
                .fg(ratatui::style::Color::Yellow)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )
    } else {
        ratatui::text::Span::styled(
            s,
            ratatui::style::Style::default().fg(ratatui::style::Color::Green),
        )
    }
}

#[allow(dead_code)]
fn ratui_line(spans: Vec<ratatui::text::Span<'static>>) -> ratatui::text::Line<'static> {
    ratatui::text::Line::from(spans)
}
