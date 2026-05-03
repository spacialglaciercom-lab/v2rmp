use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use v2rmp::app::{App, InputField, LogLevel, View};

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new();
    app.log(LogLevel::Info, "rmpca v0.1.0 - Route Optimization TUI");
    app.log(LogLevel::Info, "Press 'q' to quit, '?' for help");
    app.log(
        LogLevel::Info,
        "Clipboard: Ctrl+C (copy), Ctrl+V (paste), Ctrl+X (cut)",
    );
    let mut clipboard = arboard::Clipboard::new().ok();
    if clipboard.is_none() {
        app.log(
            LogLevel::Warn,
            "Clipboard unavailable - copy/paste disabled",
        );
    }
    let res = run_app(&mut terminal, &mut app, &mut clipboard);
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    if let Err(err) = res {
        eprintln!("Error: {:?}", err);
    }
    Ok(())
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    clipboard: &mut Option<arboard::Clipboard>,
) -> Result<()> {
    loop {
        terminal.draw(|f| match app.current_view {
            View::Home => v2rmp::ui::home::render(f, app),
            View::Extract => v2rmp::ui::extract::render(f, app),
            View::Compile => v2rmp::ui::compile::render(f, app),
            View::Optimize => v2rmp::ui::optimize::render(f, app),
            View::BrowseMaps => v2rmp::ui::browse_maps::render(f, app),
            View::BrowseRoutes => v2rmp::ui::browse_routes::render(f, app),
            View::Help => v2rmp::ui::home::render(f, app),
        })?;
        if !app.running {
            break;
        }
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if app.input_mode.active {
                    match key.code {
                        KeyCode::Enter => app.confirm_input(),
                        KeyCode::Esc => app.cancel_input(),
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            if let Some(ref mut clip) = clipboard {
                                if clip.set_text(&app.input_mode.buffer).is_ok() {
                                    app.log(LogLevel::Success, "Copied to clipboard");
                                }
                            }
                        }
                        KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            if let Some(ref mut clip) = clipboard {
                                if let Ok(text) = clip.get_text() {
                                    app.input_mode.buffer.push_str(&text);
                                    app.log(LogLevel::Success, "Pasted from clipboard");
                                }
                            }
                        }
                        KeyCode::Char('x') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            if let Some(ref mut clip) = clipboard {
                                if clip.set_text(&app.input_mode.buffer).is_ok() {
                                    app.input_mode.buffer.clear();
                                    app.log(LogLevel::Success, "Cut to clipboard");
                                }
                            }
                        }
                        KeyCode::Char(c) => app.input_mode.buffer.push(c),
                        KeyCode::Backspace => {
                            app.input_mode.buffer.pop();
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Char('q') => app.running = false,
                        KeyCode::Char('?') => app.current_view = View::Help,
                        KeyCode::Esc | KeyCode::Char('h') => app.current_view = View::Home,
                        KeyCode::Up | KeyCode::Char('k') => app.navigate_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.navigate_down(),
                        KeyCode::Enter => handle_enter(app),
                        KeyCode::Char('1') => app.current_view = View::Extract,
                        KeyCode::Char('2') => app.current_view = View::Compile,
                        KeyCode::Char('3') => app.current_view = View::Optimize,
                        KeyCode::Char('4') => app.current_view = View::BrowseMaps,
                        KeyCode::Char('5') => app.current_view = View::BrowseRoutes,
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(())
}

fn handle_enter(app: &mut App) {
    match app.current_view {
        View::Home => match app.workflow_selection {
            0 => app.current_view = View::Extract,
            1 => app.current_view = View::Compile,
            2 => app.current_view = View::Optimize,
            3 => app.current_view = View::BrowseMaps,
            4 => app.current_view = View::BrowseRoutes,
            _ => {}
        },
        View::Extract => app.start_input(InputField::BoundingBox),
        View::Compile => app.start_input(InputField::InputFile),
        View::Optimize => app.start_input(InputField::CacheFile),
        _ => {}
    }
}
