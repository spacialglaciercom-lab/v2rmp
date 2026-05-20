sed -i 's/let embeddings:/let _embeddings:/' src/core/optimize.rs
sed -i 's/let elapsed_ms = /let _elapsed_ms = /' src/core/optimize.rs
sed -i 's/let mut weight =/let weight =/' src/core/vrp/utils.rs
