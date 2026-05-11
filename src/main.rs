#![cfg(feature = "cli")]
mod app;
mod cli;
mod core;
mod event;
mod ui;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // No arguments = launch the TUI; any arguments = CLI mode
    if std::env::args().len() == 1 {
        run_tui().await
    } else {
        cli::run().await?;
        Ok(())
    }
}

async fn run_tui() -> anyhow::Result<()> {
    use std::io;
    use std::time::Duration;

    use crossterm::{
        event::{DisableMouseCapture, EnableMouseCapture},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::Terminal;

    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // App init
    let mut app = app::App::new();
    app.log(
        app::LogLevel::Info,
        format!("rmpca v{} started", env!("CARGO_PKG_VERSION")),
    );
    app.log(
        app::LogLevel::Info,
        "Ready — select a workflow step to begin",
    );

    // Main loop
    let tick_rate = Duration::from_millis(100);
    while app.running {
        terminal.draw(|f| ui::draw(f, &app))?;

        if let Some(ev) = event::poll_event(tick_rate)? {
            event::handle_event(&mut app, ev).await?;
        }
    }

    // Terminal teardown
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}
