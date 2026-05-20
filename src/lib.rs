//! v2rmp library - Route optimization and data extraction

#[cfg(feature = "cli")]
pub mod app;
#[cfg(feature = "cli")]
pub mod cli;
pub mod core;
#[cfg(feature = "cli")]
pub mod event;
#[cfg(feature = "gui")]
pub mod gui;
#[cfg(feature = "cli")]
pub mod ui;
