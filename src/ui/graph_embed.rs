use ratatui::Frame;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let inner = super::draw_panel(f, "Graph Embeddings", area);

    let cyan = ratatui::style::Color::Cyan;
    let yellow = ratatui::style::Color::Yellow;
    let gray = ratatui::style::Color::DarkGray;
    let green = ratatui::style::Color::Green;

    let input_display = match &app.graph_embed_input {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(not set)".to_string(),
            ratatui::style::Style::default().fg(yellow),
        ),
    };

    let output_display = match &app.graph_embed_output {
        Some(p) => {
            ratatui::text::Span::styled(p.clone(), ratatui::style::Style::default().fg(green))
        }
        None => ratatui::text::Span::styled(
            "(stdout)".to_string(),
            ratatui::style::Style::default().fg(gray),
        ),
    };

    let method_color = if app.graph_embed_method == "node2vec" {
        ratatui::style::Color::Magenta
    } else if app.graph_embed_method == "line" {
        ratatui::style::Color::LightCyan
    } else if app.graph_embed_method == "fastrp" {
        ratatui::style::Color::LightGreen
    } else {
        ratatui::style::Color::LightYellow
    };

    let edges_label = if app.graph_embed_include_edges {
        "ON"
    } else {
        "OFF"
    };
    let edges_color = if app.graph_embed_include_edges {
        green
    } else {
        gray
    };

    let status_text = format!("Status: {}", app.graph_embed_status);
    let status_color = app.graph_embed_status.color();

    let lines = vec![
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Road Network Node Embeddings",
            ratatui::style::Style::default()
                .fg(cyan)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Input (.rmp):       "),
            input_display,
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Output (.json):     "),
            output_display,
        ]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Method:             "),
            ratatui::text::Span::styled(
                app.graph_embed_method.clone(),
                ratatui::style::Style::default()
                    .fg(method_color)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Dimensions:         "),
            ratatui::text::Span::styled(
                app.graph_embed_dimensions.to_string(),
                ratatui::style::Style::default().fg(yellow),
            ),
        ]),
        ratatui::text::Line::from(vec![
            ratatui::text::Span::raw("Include Edges:      "),
            ratatui::text::Span::styled(
                edges_label.to_string(),
                ratatui::style::Style::default().fg(edges_color),
            ),
        ]),
        ratatui::text::Line::from(""),
        // Method-specific params
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Method Parameters:",
            ratatui::style::Style::default().fg(cyan),
        )),
        ratatui::text::Line::from(vec![ratatui::text::Span::styled(
            format!(
                "  Walk length: {}  |  Num walks: {}",
                app.graph_embed_walk_length, app.graph_embed_num_walks
            ),
            ratatui::style::Style::default().fg(gray),
        )]),
        ratatui::text::Line::from(vec![ratatui::text::Span::styled(
            format!(
                "  p (return): {}  |  q (in-out): {}",
                app.graph_embed_p, app.graph_embed_q
            ),
            ratatui::style::Style::default().fg(gray),
        )]),
        ratatui::text::Line::from(vec![ratatui::text::Span::styled(
            format!("  Epochs (LINE): {}", app.graph_embed_epochs),
            ratatui::style::Style::default().fg(gray),
        )]),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            status_text,
            ratatui::style::Style::default().fg(status_color),
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::raw(
            "────────────────────────────────────",
        )),
        ratatui::text::Line::from(""),
        ratatui::text::Line::from(ratatui::text::Span::styled(
            "Controls:",
            ratatui::style::Style::default().fg(cyan),
        )),
        ratatui::text::Line::from("  [I]  Set input .rmp file (or browse)"),
        ratatui::text::Line::from("  [O]  Set output .json file"),
        ratatui::text::Line::from("  [M]  Cycle method (node2vec/line/fastrp/spatial)"),
        ratatui::text::Line::from("  [D]  Set dimensions"),
        ratatui::text::Line::from("  [E]  Toggle edge embeddings"),
        ratatui::text::Line::from("  [Enter] Run embedding"),
        ratatui::text::Line::from("  [Esc] Return to home"),
    ];

    let paragraph = ratatui::widgets::Paragraph::new(lines);
    f.render_widget(paragraph, inner);

    super::draw_input_prompt(f, app, area);
}
