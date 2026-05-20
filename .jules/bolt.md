## 2025-05-15 - [Edge Traversal Optimization]
**Learning:** In graph processing tasks where edges have contiguous integer IDs, using a `Vec<T>` of size `edges.len()` for tracking metadata (like traversal counts) is significantly more efficient than a `HashMap<usize, T>`. It avoids hashing overhead and improves cache locality.
**Action:** Always prefer pre-allocated `Vec` over `HashMap` when the keys are dense integer indices.

## 2025-05-15 - [Compilation Awareness]
**Learning:** Refactored core modules in this codebase (like `src/core/optimize.rs`) might have broken call-sites in `src/cli.rs` or other binaries that are not caught by narrow module tests.
**Action:** Always run `cargo check` or `cargo test` on the entire workspace after modifying core function signatures or struct definitions.

## 2026-05-07 - [Road Network Compilation Optimization]
**Learning:** During GeoJSON to binary graph compilation, reusing the 'to_node' ID as the 'from_node' for the next segment in a LineString avoids redundant HashMap lookups (approx. 50% reduction). Additionally, packing two i32 coordinates into a single u64 key for the node HashMap reduces hashing overhead and memory usage compared to an (i64, i64) tuple.
**Action:** Always look for opportunities to carry over state between iterations in path-processing loops and use bit-packed integers for composite keys in performance-critical HashMaps.
## 2026-05-07 - [Subgraph Pruning]
**Learning:** Extracted road networks often contain disconnected "island" subgraphs. Adding an optional graph pruning step (finding the largest connected component via BFS/DFS) directly in the `compile.rs` step ensures that the `.rmp` binary file is robust for downstream Eulerian circuit and VRP solvers. Node and Edge index mapping (`old_to_new`) must be properly handled to keep the binary graph structure valid after pruning nodes.
**Action:** When serializing graphs to binary format, ensure isolated subgraphs can be optionally removed to guarantee solver robustness.

## 2026-05-08 - [Geometric Computation Optimization]
**Learning:** Redundant `to_radians()` conversions and trigonometric calls in large graph circuits can be a major bottleneck. Reusing bearings between adjacent segments in a circuit reduces `atan2` and trig calls by 50% during turn classification.
**Action:** Always pre-calculate radian coordinates and reuse intermediate geometric results (like bearings) when iterating over contiguous paths or circuits.

## 2026-05-08 - [Avoid Redundant Allocations]
**Learning:** Using `std::slice::from_ref(item)` is a zero-cost way to pass a single item to a function expecting a slice, avoiding unnecessary vector allocations or clones that occur with `&[item.clone()]`.
**Action:** Prefer `std::slice::from_ref` for performance-critical paths where single items are passed as slices.

## 2026-05-14 - [MCP Tool Addition Process]
**Learning:** Adding MCP tools required ensuring all parameters are handled with rigorous JSON unwrapping, proper string formatting, and error mapping to the JSON-RPC interface. Additionally, unused code paths highlighted by clippy in previously existing modules (e.g., `ml_legacy.rs`, `cli.rs`) required careful masking or `allow(dead_code)` application since they weren't explicitly used by the new handlers.
**Action:** When adding API handlers or tools that touch conditionally compiled or deprecated parts of a codebase, address or gracefully ignore existing `dead_code` to maintain clean CI/CD status without performing sweeping unsanctioned refactors.
