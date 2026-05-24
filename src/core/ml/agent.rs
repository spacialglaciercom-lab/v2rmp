//! Self-Improving Agentic Harness for v2rmp.
//!
//! Implements the "Continual Harness" principles:
//! - Online adaptation (reset-free learning)
//! - Automated harness refinement (updating prompts, skills, and memory)
//! - Sub-agent specialization

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::core::ml::telemetry::{TelemetryRecord, log_telemetry};

/// The core agent structure representing the self-improving harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2RMPAgent {
    /// System Prompt (p): High-level instructions that evolve over time.
    pub prompt: String,
    
    /// Skills (K): Reusable code snippets or tool-chaining logic.
    /// Maps skill name to executable bash/script logic.
    pub skills: HashMap<String, String>,
    
    /// Memory (M): Structured rules, pitfalls, and performance metrics.
    pub memory: AgentMemory,
    
    /// Execution history for online adaptation.
    #[serde(skip)]
    pub history: Vec<TelemetryRecord>,
}

/// Structured memory for the agent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMemory {
    /// Project-specific rules (e.g., "Always use Overture for North America").
    pub rules: Vec<String>,
    
    /// Common pitfalls (e.g., "rust-optimizer fails on graphs >200K edges").
    pub pitfalls: Vec<String>,
    
    /// Per-solver performance metrics (avg speed, efficiency).
    pub metrics: HashMap<String, SolverMetrics>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverMetrics {
    pub avg_speed_ms: f64,
    pub efficiency_score: f64,
    pub failure_count: usize,
    pub total_runs: usize,
}

/// Interface for sub-agents (G).
pub trait SubAgent: Send + Sync {
    fn name(&self) -> &str;
    fn role(&self) -> &str;
    fn execute(&self, task_json: &str) -> Result<String>;
}

/// DataAgent: Manages data sources (Overture, OSM, PMTiles).
pub struct DataAgent;
impl SubAgent for DataAgent {
    fn name(&self) -> &str { "DataAgent" }
    fn role(&self) -> &str { "Manage data sources and extraction" }
    fn execute(&self, _task_json: &str) -> Result<String> {
        // Implementation for dynamic source selection based on bbox
        Ok("DataAgent: Extraction plan generated".to_string())
    }
}

/// SolverAgent: Dynamically selects CPP/VRP solvers.
pub struct SolverAgent;
impl SubAgent for SolverAgent {
    fn name(&self) -> &str { "SolverAgent" }
    fn role(&self) -> &str { "Optimize solver selection and execution" }
    fn execute(&self, _task_json: &str) -> Result<String> {
        // Implementation for neural solver prediction
        Ok("SolverAgent: Neural prediction complete".to_string())
    }
}

impl V2RMPAgent {
    pub fn new(initial_prompt: Option<String>) -> Self {
        let prompt = initial_prompt.unwrap_or_else(|| {
            "You are a route optimization expert. Use v2rmp tools to solve CPP/VRP problems efficiently. \
             Prefer neural_guided for VRP if memory allows. Validate outputs with 'rmpca clean'."
            .to_string()
        });

        let mut skills = HashMap::new();
        skills.insert("auto_repair".to_string(), "rmpca clean --fix-disconnected".to_string());
        skills.insert("check_size".to_string(), "rmpca list --resource maps".to_string());

        Self {
            prompt,
            skills,
            memory: AgentMemory {
                rules: vec![
                    "Always run 'rmpca clean' before 'rmpca optimize' for better route quality.".to_string(),
                    "Prefer Overture Maps for urban bounding boxes.".to_string(),
                ],
                pitfalls: Vec::new(),
                metrics: HashMap::new(),
            },
            history: Vec::new(),
        }
    }

    /// Get the list of sub-agents available to this harness.
    pub fn sub_agents(&self) -> Vec<Box<dyn SubAgent>> {
        vec![
            Box::new(DataAgent),
            Box::new(SolverAgent),
        ]
    }

    /// Load the agent state from the default path (~/.v2rmp/agent_state.json).
    pub fn load_default() -> Result<Self> {
        let path = Self::default_state_path();
        if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            Ok(serde_json::from_str(&data)?)
        } else {
            Ok(Self::new(None))
        }
    }

    /// Save the agent state.
    pub fn save(&self) -> Result<()> {
        let path = Self::default_state_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn default_state_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".v2rmp")
            .join("agent_state.json")
    }

    /// Add a new skill to the harness.
    pub fn add_skill(&mut self, name: &str, code: &str) {
        self.skills.insert(name.to_string(), code.to_string());
    }

    /// Record a rule in memory.
    pub fn add_rule(&mut self, rule: &str) {
        if !self.memory.rules.contains(&rule.to_string()) {
            self.memory.rules.push(rule.to_string());
        }
    }

    /// Online Adaptation: Refine the harness based on recent performance.
    pub fn refine(&mut self, records: &[TelemetryRecord]) -> Result<bool> {
        let mut updated = false;

        for record in records {
            // 1. Update Metrics
            let solver = record.selected_solver.clone();
            let metrics = self.memory.metrics.entry(solver.clone()).or_default();
            
            metrics.total_runs += 1;
            if !record.success {
                metrics.failure_count += 1;
            }
            
            // Running average for speed
            metrics.avg_speed_ms = (metrics.avg_speed_ms * (metrics.total_runs - 1) as f64 
                                   + record.runtime_ms as f64) / metrics.total_runs as f64;

            // 2. Detect Pitfalls
            if !record.success && metrics.failure_count >= 3 {
                let stops = record.instance_features.as_ref()
                    .and_then(|v| v.get(0))
                    .copied()
                    .unwrap_or(0.0);
                
                let pitfall = format!("Solver '{}' frequently fails on instances with {} stops.", 
                                      solver, stops);
                if !self.memory.pitfalls.contains(&pitfall) {
                    self.memory.pitfalls.push(pitfall);
                    
                    // 3. Update Prompt (Refinement)
                    let rule = format!("- Avoid '{}' for large problem instances; fall back to 'clarke_wright'.", solver);
                    if !self.prompt.contains(&rule) {
                        self.prompt.push_str(&format!("\n{}", rule));
                        updated = true;
                    }
                }
            }
        }

        if updated {
            self.save()?;
        }

        Ok(updated)
    }
}
