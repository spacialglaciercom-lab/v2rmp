# Code Review: Data Ingestion and Graph Representation

## Current State & Issues Identified

### 1. Graph Representation: `Vec<Vec<AdjEntry>>`
In `src/core/optimize.rs`, the adjacency list is currently built as `let mut adj: Vec<Vec<AdjEntry>> = vec![vec![]; n];` within `solve_cpp()`.
* **Hidden Heap Allocations:** This structure forces an individual heap allocation for every single node in the network to store its adjacent edges. For large road networks, this results in significant allocation overhead and memory fragmentation.
* **CPU Cache Locality:** Traversing a `Vec<Vec<T>>` involves pointer chasing. Since the inner vectors are allocated at disjoint memory locations, iterating through adjacent nodes during the Eulerian circuit traversal will likely cause frequent CPU cache misses.

### 2. Data Ingestion: `read_rmp_file` and Structs
The `read_rmp_file` function reads the `.rmp` binary format, manually parses bytes, and pushes to `Vec<RmpNode>` and `Vec<RmpEdge>`.
* **Hidden Allocations:** While `Vec::with_capacity` is used, the parsing process still requires allocating two separate vectors and iterating through the entire file to populate them.
* **Suboptimal Lifetime Bounds:** Currently, the returned `RmpNode` and `RmpEdge` structs are fully owned by the caller. There's no lifetime bound tying the graph structs to the mapped file buffer, meaning zero-copy reading is impossible under the current schema.

## Proposed Refactoring Strategies

### 1. Transition to Compressed Sparse Row (CSR) or Arena Allocator
To address the overhead of `Vec<Vec<AdjEntry>>`, the graph representation should be refactored:
* **Compressed Sparse Row (CSR):** Store all `AdjEntry` items in a single, contiguous, flat `Vec<AdjEntry>`. Use a secondary array of offsets (`Vec<usize>`) mapping each node ID to the start of its edges in the flat vector. This completely eliminates the per-node heap allocations and significantly improves spatial locality for CPU caches during traversal.
* **Arena Allocators (`bumpalo`):** Alternatively, allocate the inner slices from a single arena (`bumpalo`). This simplifies lifetime management while still keeping the data relatively packed together in memory, though CSR is typically more optimal for static graphs.

### 2. Zero-copy Deserialization (`zerocopy` or `rkyv`)
Instead of manually allocating and copying `f64` and `u32` fields from bytes:
* **`zerocopy` / `rkyv`:** The file parsing should utilize zero-copy deserialization frameworks. With `zerocopy` or `rkyv`, the raw `&[u8]` slice from the memory-mapped `.rmp` file could be directly cast to `&[RmpNode]` and `&[RmpEdge]` slices in constant time (O(1)), bypassing allocation and parsing overhead entirely.

### 3. Tighter Borrowed Slices
Functions accepting graph data should strictly rely on tightly bounded borrowed slices (e.g., `&[RmpNode]`) instead of requiring owned structures or iterating to clone subsets, allowing the Eulerian circuit algorithm to operate strictly on the cache-friendly slices.
