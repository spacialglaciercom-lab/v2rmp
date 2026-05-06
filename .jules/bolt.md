## 2025-05-15 - [Edge Traversal Optimization]
**Learning:** In graph processing tasks where edges have contiguous integer IDs, using a `Vec<T>` of size `edges.len()` for tracking metadata (like traversal counts) is significantly more efficient than a `HashMap<usize, T>`. It avoids hashing overhead and improves cache locality.
**Action:** Always prefer pre-allocated `Vec` over `HashMap` when the keys are dense integer indices.

## 2025-05-15 - [Compilation Awareness]
**Learning:** Refactored core modules in this codebase (like `src/core/optimize.rs`) might have broken call-sites in `src/cli.rs` or other binaries that are not caught by narrow module tests.
**Action:** Always run `cargo check` or `cargo test` on the entire workspace after modifying core function signatures or struct definitions.
