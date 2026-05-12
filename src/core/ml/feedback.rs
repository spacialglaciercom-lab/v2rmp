//! Online learning feedback loop.
//!
//! Logs solve results (features, solver, quality) for periodic retraining.

use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveLogEntry {
    pub timestamp: String,
    pub instance_features: Vec<f32>,
    pub solver_id: String,
    pub total_distance_km: f64,
    pub elapsed_ms: u64,
    pub gap_to_bks: Option<f64>,
}

pub fn default_log_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".v2rmp")
        .join("solve_logs.jsonl")
}

pub fn log_solve(entry: SolveLogEntry, log_path: Option<&Path>) -> Result<()> {
    let path = log_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_log_path);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json = serde_json::to_string(&entry)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    writeln!(file, "{}", json)?;
    Ok(())
}
