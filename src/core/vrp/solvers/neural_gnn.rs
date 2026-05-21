use crate::core::vrp::types::{VRPSolver, VRPSolverInput, VRPSolverOutput};
use anyhow::Result;
use async_trait::async_trait;
#[cfg(feature = "ort")]
use ort::session::Session;
#[cfg(feature = "ort")]
use ort::value::Value;
use serde::{Deserialize, Serialize};

pub struct NeuralGnnSolver;

#[derive(Debug, Deserialize, Serialize)]
struct NodeEmbedding {
    node_idx: usize,
    lat: f64,
    lon: f64,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EmbeddingResult {
    #[allow(dead_code)]
    method: String,
    #[allow(dead_code)]
    dimensions: usize,
    nodes: Vec<NodeEmbedding>,
}

#[async_trait]
impl VRPSolver for NeuralGnnSolver {
    fn id(&self) -> &str {
        "neural_gnn"
    }

    fn label(&self) -> &str {
        "Neural GNN (Drone Agent)"
    }

    fn requires_matrix(&self) -> bool {
        true
    }

    async fn solve(&self, input: &VRPSolverInput) -> Result<VRPSolverOutput, String> {
        #[cfg(not(feature = "ort"))]
        {
            let _ = input;
            return Err("Neural GNN solver requires the 'ort' feature.".to_string());
        }

        #[cfg(feature = "ort")]
        {
            let model_path = "models/gnn_drone_agent.onnx";
            if !std::path::Path::new(model_path).exists() {
                return Err(format!("GNN model not found at {}", model_path));
            }

            let embeddings_path = "mile_end_embeddings.json";
            let embedding_res = load_embeddings(embeddings_path)
                .map_err(|e| format!("Failed to load embeddings: {}", e))?;

            let matrix = input
                .matrix
                .as_ref()
                .ok_or("GNN solver requires a distance matrix")?;

            let n = input.locations.len();
            if n < 2 {
                return Err("Need at least 2 locations".to_string());
            }

            // 1. Build Inputs for GNN
            // node_embeddings: [N, 64]

            let mut node_emb_data = Vec::with_capacity(n * 64);
            for loc in &input.locations {
                // Find nearest embedding
                let mut best_dist = f64::MAX;
                let mut best_vec = None;

                for node_emb in &embedding_res.nodes {
                    let d = (loc.lat - node_emb.lat).powi(2) + (loc.lon - node_emb.lon).powi(2);
                    if d < best_dist {
                        best_dist = d;
                        best_vec = Some(&node_emb.vector);
                    }
                }

                let emb = best_vec.cloned().unwrap_or_else(|| vec![0.0; 64]);
                node_emb_data.extend(emb);
            }

            let mut adj_data = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    if i != j && matrix_get_dist(matrix, i, j) < 5.0 {
                        // threshold for adjacency
                        adj_data[i * n + j] = 1.0;
                    }
                }
            }

            let meta_data = vec![0.0f32, 1.0f32, 0.5f32]; // Wind=0, Battery=100%, Payload=50%

            // 2. Run Inference
            let mut session = Session::builder()
                .map_err(|e| format!("Failed to create session builder: {}", e))?
                .commit_from_file(model_path)
                .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

            let node_emb_val = Value::from_array(([n, 64], node_emb_data))
                .map_err(|e| format!("Failed to create node_emb Value: {}", e))?;
            let adj_val = Value::from_array(([n, n], adj_data))
                .map_err(|e| format!("Failed to create adj Value: {}", e))?;
            let meta_val = Value::from_array(([1, 3], meta_data))
                .map_err(|e| format!("Failed to create meta Value: {}", e))?;

            let outputs = session
                .run(ort::inputs![
                    "node_embeddings" => node_emb_val,
                    "adj_matrix" => adj_val,
                    "meta_state" => meta_val,
                ])
                .map_err(|e| format!("Inference failed: {}", e))?;

            let output_val = outputs
                .get("node_scores")
                .ok_or("Output 'node_scores' not found")?;
            let (_shape, scores) = output_val
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Failed to extract scores: {}", e))?;

            // 3. Greedy Construction guided by GNN scores
            let mut visited = HashSet::new();
            let mut current_route = vec![0]; // Start at depot
            visited.insert(0);

            while visited.len() < n {
                let curr = *current_route.last().unwrap();
                let mut best_next = None;
                let mut max_score = f32::NEG_INFINITY;

                for next in 1..n {
                    if !visited.contains(&next) {
                        // Combine GNN score with distance penalty
                        let dist = matrix_get_dist(matrix, curr, next) as f32;
                        let score = scores[next] - (dist * 0.1); // Weighting heuristic

                        if score > max_score {
                            max_score = score;
                            best_next = Some(next);
                        }
                    }
                }

                if let Some(next) = best_next {
                    current_route.push(next);
                    visited.insert(next);
                } else {
                    break;
                }
            }
            current_route.push(0); // Return to depot

            // 4. Polish with 2-Opt
            let improved = two_opt_improve(matrix, &current_route, 100);

            let mut total_dist = 0.0;
            let mut total_time = 0.0;
            for i in 0..improved.len() - 1 {
                total_dist += matrix_get_dist(matrix, improved[i], improved[i + 1]);
                total_time += matrix
                    .get(improved[i])
                    .and_then(|row| row.get(improved[i + 1]))
                    .map(|c| c.time)
                    .unwrap_or(0.0);
            }

            let mut stops = Vec::new();
            for &idx in &improved {
                stops.push(input.locations[idx].clone());
            }

            Ok(VRPSolverOutput {
                stops,
                routes: Some(vec![improved
                    .iter()
                    .map(|&i| input.locations[i].clone())
                    .collect()]),
                geometry: None,
                total_distance_km: format!("{:.2}", total_dist),
                total_time_min: (total_time / 60.0) as u32,
                route_stats: None,
                route_metrics: None,
                unassigned: None,
            })
        }
    }

    fn clone_box(&self) -> Box<dyn VRPSolver> {
        Box::new(NeuralGnnSolver)
    }
}

#[allow(dead_code)]
fn load_embeddings(path: &str) -> Result<EmbeddingResult> {
    if !std::path::Path::new(path).exists() {
        return Ok(EmbeddingResult {
            method: "none".into(),
            dimensions: 64,
            nodes: vec![],
        });
    }
    let content = std::fs::read_to_string(path)?;
    let res: EmbeddingResult = serde_json::from_str(&content)?;
    Ok(res)
}
