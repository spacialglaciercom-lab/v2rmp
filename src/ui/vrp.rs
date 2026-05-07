use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" VRP Solver ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let cyan = ratatui::style::Color::Cyan;
    let yellow = ratatui::style::Color::Yellow;
    let gray = ratatui::style::Color::DarkGray;
    let green = ratatui::style::Color::Green;

    let input_display = match &app.vrp_input_file {
        Some(p) => ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green)),
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(yellow),
        ),
    };

    let waypoints_display = match &app.vrp_waypoints_file {
        Some(p) => ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green)),
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(gray),
        ),
    };

    let depots_display = if app.vrp_depots.is_empty() {
        ratatui::text::Span::styled("(none)".to_string(), ratatui::style::Style::default().fg(yellow))
    } else {
        ratatui::text::Span::styled(
            app.vrp_depots.join(", "),
            ratatui::style::Style::default().fg(green),
        )
    };

    let capacity_display = match app.vrp_capacity {
        Some(c) => ratatui::text::Span::styled(format!("{:.1}", c), ratatui::style::Style::default().fg(yellow)),
        None => ratatui::text::Span::styled("(unlimited)", ratatui::style::Style::default().fg(gray)),
    };

    let status_text = format!("Status: {}", app.vrp_status);
    let status_color = app.vrp_status.color();

    let lines = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "VRP Solver — Multi-Vehicle Route Planning",
            ratatui::style::Style::default()
                .fg(cyan)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Input (.rmp):      "),
            input_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Waypoints File:    "),
            waypoints_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Output Directory:  "),
            ratatui::text::Span::styled(
                app.vrp_output_dir.clone(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            status_text,
            ratatui::style::Style::default().fg(status_color),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Vehicles:          "),
            ratatui::text::Span::styled(
                app.vrp_vehicles.to_string(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Algorithm:         "),
            ratatui::text::Span::styled(
                app.vrp_algo.clone(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Capacity:          "),
            capacity_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Depots (lat,lon):  "),
            depots_display,
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
        ratatui::text::Line::from("  [I]  Set input .rmp file"),
        ratatui::text::Line::from("  [W]  Set waypoints JSON file"),
        ratatui::text::Line::from("  [O]  Set output directory"),
        ratatui::text::Line::from("  [V]  Set number of vehicles"),
        ratatui::text::Line::from("  [A]  Change algorithm (greedy|savings|local_search|simulated_annealing)"),
        ratatui::text::Line::from("  [C]  Set vehicle capacity"),
        ratatui::text::Line::from("  [D]  Add depot (lat,lon)"),
        ratatui::text::Line::from("  [X]  Clear all depots"),
        ratatui::text::Line::from("  [Enter]  Run VRP solver"),
        ratatui::text::Line::from("  [Esc]  Return to home"),
    ];

    let paragraph = ratatui::widgets::Paragraph::new(lines);
    f.render_widget(paragraph, inner);

    super::draw_input_prompt(f, app, area);
}
