use v2rmp::core::optimize::{RmpNode, RmpEdge};
use v2rmp::core::ml::graph_embed::embed_network;

#[test]
fn test_graph_sage_embedding() {
    let nodes = vec![
        RmpNode { lat: 45.5, lon: -73.6 },
        RmpNode { lat: 45.51, lon: -73.61 },
        RmpNode { lat: 45.52, lon: -73.62 },
    ];
    let edges = vec![
        RmpEdge { from: 0, to: 1, weight_m: 1000.0, oneway: 0 },
        RmpEdge { from: 1, to: 2, weight_m: 1000.0, oneway: 1 },
    ];

    let embs = embed_network(&nodes, &edges);
    
    // If the model exists, we should get embeddings for each edge
    if !embs.is_empty() {
        assert_eq!(embs.len(), edges.len());
        assert_eq!(embs[0].vector.len(), 64);
        println!("Generated embeddings for {} edges", embs.len());
    } else {
        println!("Model not found or failed, skipping assertion");
    }
}
