## 2025-05-15 - [Edge Traversal Optimization]
**Learning:** In graph processing tasks where edges have contiguous integer IDs, using a `Vec<T>` of size `edges.len()` for tracking metadata (like traversal counts) is significantly more efficient than a `HashMap<usize, T>`. It avoids hashing overhead and improves cache locality.
**Action:** Always prefer pre-allocated `Vec` over `HashMap` when the keys are dense integer indices.

## 2025-05-15 - [Compilation Awareness]
**Learning:** Refactored core modules in this codebase (like `src/core/optimize.rs`) might have broken call-sites in `src/cli.rs` or other binaries that are not caught by narrow module tests.
**Action:** Always run `cargo check` or `cargo test` on the entire workspace after modifying core function signatures or struct definitions.

## 2026-05-07 - [Road Network Compilation Optimization]
**Learning:** During GeoJSON to binary graph compilation, reusing the 'to_node' ID as the 'from_node' for the next segment in a LineString avoids redundant HashMap lookups (approx. 50% reduction). Additionally, packing two i32 coordinates into a single u64 key for the node HashMap reduces hashing overhead and memory usage compared to an (i64, i64) tuple.
**Action:** Always look for opportunities to carry over state between iterations in path-processing loops and use bit-packed integers for composite keys in performance-critical HashMaps.
