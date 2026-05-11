use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let inner = super::draw_panel(f, "VRP Optimization", area);

    let cyan = ratatui::style::Color::Cyan;
    let yellow = ratatui::style::Color::Yellow;
    let gray = ratatui::style::Color::DarkGray;
    let green = ratatui::style::Color::Green;

    let csv_display = match &app.vrp_csv_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(yellow),
        ),
    };

    let rmp_display = match &app.vrp_input_file {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(gray),
        ),
    };

    let capacity_display = match app.vrp_capacity {
        Some(c) => ratatui::text::Span::styled(
            format!("{:.1}", c),
            ratatui::style::Style::default().fg(yellow),
        ),
        None => {
            ratatui::text::Span::styled("(unlimited)", ratatui::style::Style::default().fg(gray))
        }
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
            ratatui::text::Span::raw("Coordinates CSV:   "),
            csv_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("  columns: lat,lon [, label, demand, type]"),
            ratatui::text::Span::styled("", ratatui::style::Style::default().fg(gray)),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Road Network (.rmp): "),
            rmp_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("  (optional, for map preview only)"),
            ratatui::text::Span::styled("", ratatui::style::Style::default().fg(gray)),
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
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::raw(
            "────────────────────────────────────",
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Controls:",
            ratatui::style::Style::default().fg(cyan),
        )),
        ratatui::text::Line::from("  [C]  Set coordinates CSV file"),
        ratatui::text::Line::from("  [I]  Set road network .rmp (optional)"),
        ratatui::text::Line::from("  [O]  Set output directory"),
        ratatui::text::Line::from("  [V]  Set number of vehicles"),
        ratatui::text::Line::from(
            "  [A]  Change algorithm (greedy|savings|local_search|simulated_annealing)",
        ),
        ratatui::text::Line::from("  [K]  Set vehicle capacity"),
        ratatui::text::Line::from("  [Enter]  Run VRP solver"),
        ratatui::text::Line::from("  [Esc]  Return to home"),
    ];

    let paragraph = ratatui::widgets::Paragraph::new(lines);
    f.render_widget(paragraph, inner);

    super::draw_input_prompt(f, app, area);
}
