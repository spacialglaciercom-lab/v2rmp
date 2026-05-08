use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" Optimize Route ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let cyan = ratatui::style::Color::Cyan;
    let yellow = ratatui::style::Color::Yellow;
    let gray = ratatui::style::Color::DarkGray;
    let green = ratatui::style::Color::Green;

    let cache_display = match &app.cache_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(yellow),
        ),
    };

    let route_display = match &app.route_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(gray),
        ),
    };

    let depot_display = match &app.depot_coords {
        Some((lat, lon)) => ratatui::text::Span::styled(
            format!("{:.4},{:.4}", lat, lon),
            ratatui::style::Style::default().fg(green),
        ),
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(gray),
        ),
    };

    let status_text = format!("Status: {}", app.optimize_status);
    let status_color = app.optimize_status.color();

    let lines = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Optimize Route",
            ratatui::style::Style::default()
                .fg(cyan)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Cache File:  "),
            cache_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Route File:  "),
            route_display,
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            status_text,
            ratatui::style::Style::default().fg(status_color),
        )),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Left Turn Penalty:  "),
            ratatui::text::Span::styled(
                app.turn_penalties.left.to_string(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Right Turn Penalty: "),
            ratatui::text::Span::styled(
                app.turn_penalties.right.to_string(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("U-Turn Penalty:     "),
            ratatui::text::Span::styled(
                app.turn_penalties.u_turn.to_string(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Depot Coordinates:  "),
            depot_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Number of Vehicles: "),
            ratatui::text::Span::styled(
                app.num_vehicles.to_string(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Solver Algorithm:   "),
            ratatui::text::Span::styled(
                app.solver_id.clone(),
                ratatui::style::Style::default().fg(yellow),
            ),
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
        ratatui::text::Line::from("  [C]  Set cached map file"),
        ratatui::text::Line::from("  [r]  Set route input file"),
        ratatui::text::Line::from("  [V]  Set number of vehicles"),
        ratatui::text::Line::from("  [S]  Set solver ID"),
        ratatui::text::Line::from("  [L]  Set left turn penalty"),
        ratatui::text::Line::from("  [R]  Set right turn penalty"),
        ratatui::text::Line::from("  [U]  Set U-turn penalty"),
        ratatui::text::Line::from("  [D]  Set depot coordinates"),
        ratatui::text::Line::from("  [Enter] Start optimization"),
        ratatui::text::Line::from("  [Esc] Return to home"),
    ];

    let paragraph = ratatui::widgets::Paragraph::new(lines);
    f.render_widget(paragraph, inner);

    super::draw_input_prompt(f, app, area);
}
