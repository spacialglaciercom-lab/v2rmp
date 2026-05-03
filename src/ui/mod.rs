pub mod browse_maps;
pub mod browse_routes;
pub mod clean;
pub mod compile;
pub mod extract;
pub mod file_browser;
pub mod home;
pub mod optimize;

use ratatui::Frame;

use crate::app::{App, View};

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(3), // header
            ratatui::layout::Constraint::Min(10),   // main content
            ratatui::layout::Constraint::Length(8), // logs
            ratatui::layout::Constraint::Length(1), // footer
        ])
        .split(f.area());

    draw_header(f, chunks[0]);
    draw_main(f, app, chunks[1]);
    draw_logs(f, app, chunks[2]);
    draw_footer(f, app, chunks[3]);
}

fn draw_header(f: &mut Frame, area: ratatui::layout::Rect) {
    let title = " rmpca - Route Optimization TUI [v0.3.4] ";
    let block = ratatui::widgets::Block::default()
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));
    let paragraph = ratatui::widgets::Paragraph::new(ratatui::text::Span::styled(
        title,
        ratatui::style::Style::default().fg(ratatui::style::Color::Cyan),
    ))
    .block(block)
    .alignment(ratatui::layout::Alignment::Center);
    f.render_widget(paragraph, area);
}

fn draw_main(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    match app.current_view {
        View::Home => home::draw(f, app, area),
        View::Extract => extract::draw(f, app, area),
        View::Compile => compile::draw(f, app, area),
        View::Optimize => optimize::draw(f, app, area),
        View::BrowseMaps => browse_maps::draw(f, app, area),
        View::BrowseRoutes => browse_routes::draw(f, app, area),
        View::FileBrowser => file_browser::draw(f, app, area),
        View::Help => draw_help(f, area),
        View::Clean => home::draw(f, app, area),
    }
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
                crate::app::LogLevel::Info => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Cyan)
                }
                crate::app::LogLevel::Success => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Green)
                }
                crate::app::LogLevel::Warn => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Yellow)
                }
                crate::app::LogLevel::Error => {
                    ratatui::style::Style::default().fg(ratatui::style::Color::Red)
                }
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

fn draw_footer(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let text = match app.current_view {
        View::Home | View::Extract | View::Optimize => {
            "[q] Quit  [Esc] Home  [h/F1] Help  [↑↓] Navigate  [Enter] Select"
        }
        View::Compile => "[q] Quit  [Esc] Home  [I] Input file  [O] Output file  [Enter] Compile",
        View::BrowseMaps => {
            "[↑↓] Navigate  [Enter] Select -> Optimize  [d] Delete  [r] Refresh  [Esc] Home"
        }
        View::BrowseRoutes => "[↑↓] Navigate  [Enter] View  [d] Delete  [r] Refresh  [Esc] Home",
        View::FileBrowser => {
            "[↑↓] Navigate  [Enter] Select  [Backspace] Parent  [h] Toggle hidden  [Esc] Cancel"
        }
        View::Help => "[Esc] Home  [q] Quit",
        View::Clean => "[Esc] Home  [q] Quit",
    };

    let paragraph = ratatui::widgets::Paragraph::new(ratatui::text::Span::styled(
        text,
        ratatui::style::Style::default()
            .fg(ratatui::style::Color::DarkGray)
            .add_modifier(ratatui::style::Modifier::REVERSED),
    ));
    f.render_widget(paragraph, area);
}

fn draw_help(f: &mut Frame, area: ratatui::layout::Rect) {
    let block = ratatui::widgets::Block::default()
        .title(" Help ")
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Cyan));

    let lines = vec![
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "rmpca — Route Optimization TUI",
            ratatui::style::Style::default()
                .fg(ratatui::style::Color::Cyan)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from("Global Keys:"),
        ratatui::text::Line::from("  [q]      Quit application"),
        ratatui::text::Line::from("  [Esc]    Return to home"),
        ratatui::text::Line::from("  [h/F1]   Toggle this help"),
        ratatui::text::Line::from("  [↑↓]     Navigate menus"),
        ratatui::text::Line::from("  [Enter]  Select / confirm"),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from("Workflow Steps:"),
        ratatui::text::Line::from("  1. Extract road data from OSM or Overture"),
        ratatui::text::Line::from("  2. Compile GeoJSON into binary .rmp format"),
        ratatui::text::Line::from("  3. Optimize route with turn penalties"),
        ratatui::text::Line::from("  4. Browse cached maps and saved routes"),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from("Core algorithms: CPP (Eulerian circuit), TSP (2-opt),"),
        ratatui::text::Line::from("VRP (OR-Tools), haversine distance, bearing-based turns."),
    ];

    let paragraph = ratatui::widgets::Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

/// Draw an input prompt overlay when input mode is active.
pub fn draw_input_prompt(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    if !app.input_mode.active {
        return;
    }

    let label = match app.input_mode.field {
        crate::app::InputField::BoundingBox => "Bounding Box (min_lat,min_lon,max_lat,max_lon)",
        crate::app::InputField::InputFile => "Input GeoJSON file path",
        crate::app::InputField::OutputFile => "Output .rmp file path",
        crate::app::InputField::CacheFile => "Cache map file path",
        crate::app::InputField::RouteFile => "Route input file path",
        crate::app::InputField::LeftTurnPenalty => "Left turn penalty",
        crate::app::InputField::RightTurnPenalty => "Right turn penalty",
        crate::app::InputField::UTurnPenalty => "U-turn penalty",
        crate::app::InputField::DepotCoordinates => "Depot coordinates (lat,lon)",
        crate::app::InputField::CleanInputFile => "Clean input GeoJSON file path",
        crate::app::InputField::CleanOutputFile => "Clean output GeoJSON file path",
    };

    let popup_area = ratatui::layout::Rect {
        x: area.x + 2,
        y: area.y + area.height.saturating_sub(4),
        width: area.width.saturating_sub(4),
        height: 3,
    };

    let input_block = ratatui::widgets::Block::default()
        .title(format!(" {} ", label))
        .borders(ratatui::widgets::Borders::ALL)
        .border_style(ratatui::style::Style::default().fg(ratatui::style::Color::Yellow));

    let display = format!("{}█", app.input_mode.buffer);
    let paragraph = ratatui::widgets::Paragraph::new(ratatui::text::Span::styled(
        display,
        ratatui::style::Style::default().fg(ratatui::style::Color::White),
    ))
    .block(input_block);

    f.render_widget(paragraph, popup_area);
}
