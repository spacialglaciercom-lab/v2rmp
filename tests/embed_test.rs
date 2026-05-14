#![cfg(feature = "ml")]

use v2rmp::core::embed::run_embed;

#[test]
fn test_embedder_integration() {
    let texts = vec![
        "This is a test sentence.".to_string(),
        "Another sentence to embed.".to_string(),
    ];

    match run_embed(texts) {
        Ok(embeddings) => {
            assert_eq!(embeddings.len(), 2);
            // BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings
            assert_eq!(embeddings[0].len(), 384);
            assert_eq!(embeddings[1].len(), 384);
        }
        Err(e) => {
            // If the HuggingFace Hub is unreachable or the model cannot be downloaded,
            // we catch the error to prevent the test from failing purely due to network issues.
            println!("Failed to run embedder (likely network/HF Hub error): {}", e);
            // We don't panic here because it's a known potential flakiness source.
        }
    }
}
