#[cfg(feature = "cli")]
use v2rmp::{app, cli, event, ui};
#[cfg(feature = "cli")]
use ratatui::backend::CrosstermBackend;
#[cfg(feature = "cli")]
use ratatui::Terminal;
#[cfg(feature = "cli")]
use std::io;
#[cfg(feature = "cli")]
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    #[cfg(feature = "cli")]
    {
        let stdout = io::stdout();
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut app = app::App::new();
        app.log(
            app::LogLevel::Info,
            format!("rmpca v{} - Route Optimization Engine", env!("CARGO_PKG_VERSION")),
        );
        app.log(
            app::LogLevel::Info,
            "Press 'h' for help, 'q' to quit.".into(),
        );

        let tick_rate = Duration::from_millis(250);
        loop {
            terminal.draw(|f| ui::draw(f, &app))?;

            if let Some(ev) = event::poll_event(tick_rate)? {
                event::handle_event(&mut app, ev).await?;
            }

            if app.should_quit {
                break;
            }
        }

        Ok(())
    }
    #[cfg(not(feature = "cli"))]
    {
        println!("rmpca v{} - Route Optimization Engine (CLI feature disabled)", env!("CARGO_PKG_VERSION"));
        println!("This binary requires the 'cli' feature for the TUI.");
        Ok(())
    }
}
