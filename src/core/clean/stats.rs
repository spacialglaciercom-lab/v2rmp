use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CleanStats {
    pub input_features: usize,
    pub output_features: usize,
    pub invalid_dropped: usize,
    pub selfloops_removed: usize,
    pub short_edges_removed: usize,
    pub nodes_merged: usize,
    pub duplicate_edges_removed: usize,
    pub incomplete_edges_removed: usize,
    pub parallel_edges_merged: usize,
    pub isolates_removed: usize,
    pub components_removed: usize,
}

impl CleanStats {
    pub fn total_removed(&self) -> usize {
        self.invalid_dropped
            + self.selfloops_removed
            + self.short_edges_removed
            + self.duplicate_edges_removed
            + self.incomplete_edges_removed
            + self.isolates_removed
    }

    pub fn summary(&self) -> String {
        format!(
            "Input: {} → Output: {} ({} removed)\n\
             - Invalid: {}\n\
             - Self-loops: {}\n\
             - Short edges: {}\n\
             - Nodes merged: {}\n\
             - Duplicate edges: {}\n\
             - Incomplete edges: {}\n\
             - Parallel edges merged: {}\n\
             - Isolates: {}\n\
             - Components removed: {}",
            self.input_features,
            self.output_features,
            self.total_removed(),
            self.invalid_dropped,
            self.selfloops_removed,
            self.short_edges_removed,
            self.nodes_merged,
            self.duplicate_edges_removed,
            self.incomplete_edges_removed,
            self.parallel_edges_merged,
            self.isolates_removed,
            self.components_removed
        )
    }
}
