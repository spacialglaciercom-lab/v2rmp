use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::app::App;
use crate::core::drone::{DroneModel, DroneSpec};

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Drone Specs
            Constraint::Min(5),     // Results
            Constraint::Length(3),  // Action bar
        ])
        .split(area);

    draw_specs(f, app, chunks[0]);
    draw_results(f, app, chunks[1]);
    draw_actions(f, app, chunks[2]);
}

fn draw_specs(f: &mut Frame, app: &App, area: Rect) {
    let spec = match app.drone_model {
        DroneModel::FlyCart30 => DroneSpec::flycart30(),
        DroneModel::Wing => DroneSpec::wing(),
    };

    let block = Block::default()
        .title(" Drone Specifications ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let text = vec![
        Line::from(vec![
            Span::styled("Model: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{:?}", spec.model)),
        ]),
        Line::from(vec![
            Span::styled("Payload: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{:.1} kg / {:.1} kg max", 0.0, spec.max_payload_kg)),
        ]),
        Line::from(vec![
            Span::styled("Battery: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{:.1} Wh", spec.battery_capacity_wh)),
        ]),
        Line::from(vec![
            Span::styled("Cruise Speed: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{:.1} m/s", spec.cruise_speed_ms)),
        ]),
        Line::from(vec![
            Span::styled("Wind Speed: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:.1} m/s", app.drone_wind_speed), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Dorling et al. (2017) Energy Model active ",
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )),
    ];

    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, area);
}

fn draw_results(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Drone VRP Results ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));

    let text = if let Some(res) = &app.drone_vrp_result {
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Routes: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(format!("{}", res.routes.len())),
            ]),
            Line::from(vec![
                Span::styled("Total Energy: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(format!("{:.2} Wh", res.energy_used_wh), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(""),
        ];

        for (i, route) in res.routes.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(format!("Route #{}: ", i + 1), Style::default().fg(Color::Cyan)),
                Span::raw(format!("{:?}", route)),
            ]));
        }

        if !res.violations.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("Violations:", Style::default().fg(Color::Red))));
            for v in &res.violations {
                lines.push(Line::from(Span::styled(format!(" - {}", v), Style::default().fg(Color::Red))));
            }
        }
        lines
    } else {
        vec![Line::from("No solution generated. Press [Enter] to solve.")]
    };

    let paragraph = Paragraph::new(text).block(block).wrap(Wrap { trim: true });
    f.render_widget(paragraph, area);
}

fn draw_actions(f: &mut Frame, _app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let text = Line::from(vec![
        Span::styled("[Enter]", Style::default().fg(Color::Yellow)),
        Span::raw(" Solve VRP  "),
        Span::styled("[d]", Style::default().fg(Color::Yellow)),
        Span::raw(" Toggle Model  "),
        Span::styled("[w]", Style::default().fg(Color::Yellow)),
        Span::raw(" Set Wind  "),
        Span::styled("[Esc]", Style::default().fg(Color::Yellow)),
        Span::raw(" Back"),
    ]);

    let paragraph = Paragraph::new(text).block(block).alignment(ratatui::layout::Alignment::Center);
    f.render_widget(paragraph, area);
}
