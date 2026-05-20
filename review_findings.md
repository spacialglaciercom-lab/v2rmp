# Review of Data Ingestion and Graph Representation

## 1. Hidden Heap Allocations in Graph Representation
**Current State:**
In `src/core/optimize.rs`, the graph is built using an adjacency list represented as `Vec<Vec<AdjEntry>>`:
```rust
let mut adj: Vec<Vec<AdjEntry>> = vec![vec![]; n];
```
**Issue:**
This structure creates $N$ separate heap allocations (one for each node's outgoing edge list). During Eulerian circuit traversal (Hierholzer's algorithm), accessing these scattered vectors leads to poor CPU cache locality and frequent cache misses.

**Proposed Refactoring Strategy:**
- **Compressed Sparse Row (CSR):** Replace `Vec<Vec<AdjEntry>>` with a CSR format using two flat `Vec`s (e.g., `edge_offsets: Vec<usize>` and `edge_data: Vec<AdjEntry>`). This guarantees contiguous memory layout, drastically improving cache locality during the while loop traversal.
- **Arena Allocators:** Alternatively, use an arena allocator like `bumpalo` to allocate all edge vectors in a single memory block, preventing scattered heap allocations and reducing initialization time.

## 2. Inefficient Data Ingestion (Allocations in Binary Parsing)
**Current State:**
In `read_rmp_file` (`src/core/optimize.rs`), the parsing of `.rmp` files manually deserializes bytes and allocates new vectors for nodes and edges:
```rust
let mut nodes = Vec::with_capacity(node_count);
// ... parsing logic pushing to nodes
let mut edges = Vec::with_capacity(edge_count);
// ... parsing logic pushing to edges
```
**Issue:**
This results in full copies of the data from the binary buffer into newly allocated `Vec<RmpNode>` and `Vec<RmpEdge>`. This is an unnecessary heap allocation and memory copy penalty.

**Proposed Refactoring Strategy:**
- **Zero-Copy Deserialization:** Utilize crates like `zerocopy` or `rkyv` (or standard `bytemuck` casts) to safely cast the incoming byte slice `&[u8]` directly to `&[RmpNode]` and `&[RmpEdge]`. This eliminates heap allocations entirely and maps the file data directly into memory, dramatically speeding up data ingestion and reducing memory footprint. The `RmpNode` and `RmpEdge` structs are already `[derive(Clone, Copy)]` plain-old-data (POD) types, making them ideal candidates for this approach.

## 3. Suboptimal Lifetime Bounds and Unnecessary Clones
**Current State:**
The VRP components (in `src/core/vrp/types.rs` and usage in `run_vrp_optimize`) clone the entire stops and locations array to pass into solvers:
```rust
let vrp_input = VRPSolverInput {
    locations: stops.clone(),
    // ...
```
**Issue:**
Cloning `stops` incurs a deep copy of all `VRPSolverStop` objects, including strings (like `label: format!("Node {}", i)`).

**Proposed Refactoring Strategy:**
- **Tighter Borrowed Slices:** Modify `VRPSolverInput` to accept slices (`&[VRPSolverStop]`) with explicit lifetime bounds (e.g., `&'a [VRPSolverStop]`) instead of owned `Vec`s. Strings within stops (e.g., `label: String`) could use `Cow<'a, str>` to avoid allocating strings where static or borrowed strings would suffice. This minimizes cloning and respects tighter lifecycle bounds through the VRP engine.
