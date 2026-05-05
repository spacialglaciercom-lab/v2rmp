mod app;
mod cli;
mod core;
mod event;
mod ui;

fn main() -> anyhow::Result<()> {
    // No arguments = launch the TUI; any arguments = CLI mode
    if std::env::args().len() == 1 {
        run_tui()
    } else {
        cli::run()?;
        Ok(())
    }
}

fn run_tui() -> anyhow::Result<()> {
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
    app.log(app::LogLevel::Info, "rmpca v0.3.8 started");
    app.log(
        app::LogLevel::Info,
        "Ready — select a workflow step to begin",
    );

    // Main loop
    let tick_rate = Duration::from_millis(100);
    while app.running {
        terminal.draw(|f| ui::draw(f, &app))?;

        if let Some(ev) = event::poll_event(tick_rate)? {
            event::handle_event(&mut app, ev)?;
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
