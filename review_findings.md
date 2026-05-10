# Review Findings: Data Ingestion & Graph Representation

This document outlines structural inefficiencies identified in the data ingestion and graph representation components of `v2rmp` (specifically in `src/core/optimize.rs`), along with proposed refactoring strategies to maximize CPU cache locality during Eulerian circuit traversal.

## 1. Hidden Heap Allocations

### `Vec<Vec<AdjEntry>>`
In `solve_cpp`, the graph's adjacency list is represented as a `Vec<Vec<AdjEntry>>`:
```rust
let mut adj: Vec<Vec<AdjEntry>> = vec![vec![]; n];
```
**Issue:** This representation results in $\mathcal{O}(V)$ separate heap allocations. As the graph grows (e.g., to 250,000 nodes for the standard benchmark grid), this scatters adjacency arrays across the heap, severely degrading CPU cache locality during the Eulerian circuit traversal.

**Refactoring Strategy:**
*   **Compressed Sparse Row (CSR):** Transition the adjacency list to a CSR-like format utilizing two contiguous arrays: one for node offsets (`Vec<usize>`) and one for edge data (`Vec<AdjEntry>`). Since Hierholzer's algorithm mutates the graph by removing edges, a modified CSR where "removed" edges are simply flagged or replaced (e.g., using tombstone values) would be highly cache-efficient.
*   **Arena Allocators (`bumpalo`):** If dynamic resizing per-node is strictly required, allocate the inner slices from a single arena (`Bump` from the `bumpalo` crate) to ensure elements remain packed in contiguous memory blocks.

## 2. Suboptimal Data Ingestion (Lack of Zero-Copy)

### `read_rmp_file`
The `.rmp` binary format parsing currently allocates entirely new vectors for nodes and edges:
```rust
let mut nodes = Vec::with_capacity(node_count);
// ...
let mut edges = Vec::with_capacity(edge_count);
```
**Issue:** Even though the underlying byte slice (`&[u8]`) may be mmap'd or already reside in memory, `read_rmp_file` eagerly deserializes the entire struct, paying the cost of copying memory byte-by-byte into new heaps (`Vec<RmpNode>` and `Vec<RmpEdge>`).

**Refactoring Strategy:**
*   **Zero-Copy Deserialization:** Adopt crates like `zerocopy` or `rkyv`. Since `RmpNode` and `RmpEdge` implement `Copy`, they can safely cast the aligned `&[u8]` buffer directly into `&[RmpNode]` and `&[RmpEdge]`. This eliminates parsing overhead and reduces the memory footprint by relying directly on the mapped file slice.

## 3. Suboptimal Lifetime Bounds & Slices

### Traversal Bounds
While `solve_cpp` accepts `nodes: &[RmpNode]` and `edges: &[RmpEdge]`, there are opportunities elsewhere to tighten bounds:
**Issue:** When filtering (e.g., `filter_bbox`), the application eagerly allocates new vectors (`Vec<RmpNode>`, `Vec<RmpEdge>`) even if no modifications are made or if only a subset is used.
**Refactoring Strategy:**
*   Where possible, define graph algorithms over tighter borrowed slices or iterators rather than owned collections. If applying zero-copy deserialization (as above), maintaining `&'a [RmpNode]` throughout the application lifetime will avoid extraneous clones.

## 4. Unnecessary Allocations in Traversal and Metric Calculation

### Circuit Edge Tracking
```rust
let mut circuit_with_edges: Vec<(u32, Option<AdjEntry>)> = Vec::new();
```
**Issue:** The Hierholzer traversal dynamically grows `circuit_with_edges` as nodes are popped off the stack.
**Refactoring Strategy:**
*   **Preallocation:** The exact number of edges to be traversed is known beforehand (original edges + duplicate edges from matching). `circuit_with_edges` should be initialized with `Vec::with_capacity(E)` to prevent reallocation during the inner while-loop of the Hierholzer traversal.