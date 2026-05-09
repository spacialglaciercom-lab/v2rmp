# Formal Verification Specification: Hierholzer's Algorithm in Route Optimization

This document provides a highly detailed Lean 4-compatible specification for the core pathfinding and Hierholzer's algorithm implementation found in `src/core/optimize.rs` (`solve_cpp`). It explicitly maps the mathematical invariants, boundary conditions, and state mutations necessary to formally verify termination, correctness, and algorithmic complexity bounds.

## 1. State Representation and Types

To model the implementation in Lean 4, the state must be represented precisely.

```lean
structure AdjEntry where
  to : Nat
  weight_m : Float
  edge_idx : Nat
  deriving Repr, BEq

-- The adjacency list is an Array where indices represent node IDs.
abbrev Graph := Array (List AdjEntry)

-- Stack tracks the current path: (Node ID, Edge traversed to reach this node)
abbrev Stack := List (Nat × Option AdjEntry)

-- Circuit stores the completed cycles and paths in reverse order of traversal.
abbrev Circuit := List (Nat × Option AdjEntry)

structure HierholzerState where
  adj : Graph
  stack : Stack
  circuit_with_edges : Circuit
```

## 2. Pre-conditions

Before the `while` loop begins, the state satisfies the following conditions:

1.  **Eulerian Graph Property:** The `adj` multigraph is effectively Eulerian for the connected component containing `start_node`.
    *   For directed edges: `InDegree(v) == OutDegree(v)`.
    *   For undirected edges: represented as symmetric pairs in `adj`. The code removes the reverse edge dynamically.
2.  **Initial State Setup:**
    *   `stack` contains exactly one element: `[(start_node, none)]`.
    *   `circuit_with_edges` is empty: `[]`.

## 3. Mathematical & Loop Invariants

The core of the formal verification relies on the loop invariant for the `while` loop.

### 3.1. Edge Conservation Invariant
Let $E_{\text{init}}$ be the total multiset of edges initially in `adj`. At any point during the execution, the set of edges is strictly partitioned across the three state structures:
*   $\text{Edges}(adj) \cup \text{Edges}(stack) \cup \text{Edges}(circuit\_with\_edges) = E_{\text{init}}$
*   Every edge is visited exactly once (or in pairs for undirected edges, which are collapsed into a single traversal event).

### 3.2. Path and Cycle Connectivity
*   **Stack Connectivity:** The elements in `stack` form a continuous, valid walk in the original graph. For any adjacent elements `(u, e_{\text{prev}})` and `(v, e_{\text{curr}})` in the stack, `e_{\text{curr}}` is a valid edge from `u` to `v`.
*   **Circuit Attachment:** `circuit_with_edges` forms a sequence of valid closed cycles or paths. The top of the `stack` (the current node `v`) is exactly the node that connects to the most recently added segment in `circuit_with_edges`.

### 3.3. Reverse Edge Consistency (Undirected Subgraphs)
For undirected edges (represented by forward and reverse entries with the same `edge_idx` and `weight_m`), traversing the forward edge $u \to v$ guarantees the removal of the reverse edge $v \to u$ from `adj[v]`.
*   *Invariant:* If an edge with `edge_idx` is in `stack` or `circuit_with_edges`, neither its forward nor its reverse component exists in `adj`.

## 4. State Mutations

The `while let Some(&(v_u32, _)) = stack.last()` loop applies exactly one of two state mutations per iteration. Let `v = v_u32 as usize`.

### Mutation A: Edge Traversal (Push)
If `adj[v].pop()` returns `Some(edge)`:
1.  **Adjacency List Mutation:** The edge is removed from `adj[v]` (complexity $\mathcal{O}(1)$).
2.  **Reverse Edge Removal:** A linear search finds the reverse edge in `adj[edge.to]` matching `to == v`, `edge_idx`, and `weight_m`. If found, `swap_remove` deletes it (complexity $\mathcal{O}(\text{degree}(edge.to))$).
3.  **Stack Mutation:** `(edge.to, Some(edge))` is pushed onto `stack` (complexity $\mathcal{O}(1)$).

### Mutation B: Backtracking (Pop)
If `adj[v].pop()` returns `None`:
1.  **Stack Mutation:** The top element `(v_u32, e)` is popped from `stack` (complexity $\mathcal{O}(1)$).
2.  **Circuit Mutation:** `(v_u32, e)` is pushed onto `circuit_with_edges` (complexity $\mathcal{O}(1)$ amortized).

## 5. Termination Measure (Well-foundedness)

To prove termination in Lean 4, we define a strictly decreasing lexicographic measure on the state `(A, S)`:
*   $A = \sum_{v \in V} |adj[v]|$ (Total number of directed edges remaining in `adj`)
*   $S = |stack|$ (Number of elements in the stack)

**Proof Sketch for Termination:**
*   In **Mutation A**, $A$ strictly decreases by at least 1 (or 2 if a reverse edge is removed). $S$ increases by 1. Since $A$ strictly decreases, the measure $(A, S)$ is strictly smaller.
*   In **Mutation B**, $A$ remains unchanged. $S$ strictly decreases by 1. Thus, the measure $(A, S)$ is strictly smaller.
*   Because $A \ge 0$ and $S \ge 0$, the sequence of states must be finite. The algorithm is guaranteed to terminate.

## 6. Boundary Conditions

The formal proof must handle these specific edge cases gracefully:

1.  **Empty Graph:** If the graph has 0 edges, $A=0$. The loop executes exactly once (Mutation B), popping `start_node` to the circuit and terminating.
2.  **Disconnected Components:** Edges in components unreachable from `start_node` will never be visited. Their presence in `adj` does not affect the termination measure, as the stack only explores the connected component of `start_node`.
3.  **Degenerate Edges (Self-loops):** An edge where `from == to`. The reverse edge removal logic correctly handles this by scanning `adj[v]` and `swap_remove`-ing the duplicate entry, reducing $A$ by 2.
4.  **Parallel Edges (Multigraphs):** Handled safely because reverse edge removal explicitly checks `to`, `edge_idx`, and `weight_m`. Parallel edges have distinct `edge_idx` values, preventing incorrect removal.

## 7. Algorithmic Complexity Bounds

Let $V$ be the number of vertices, $E$ be the total number of edges, and $D$ be the maximum degree of any vertex.

*   **Time Complexity:**
    *   The loop executes at most $2E + 1$ times.
    *   Mutation B takes $\mathcal{O}(1)$.
    *   Mutation A takes $\mathcal{O}(D)$ in the worst case due to the `iter().position(...)` linear search for the reverse edge.
    *   Total Time Complexity is bounded by $\mathcal{O}(E \times D)$.
*   **Space Complexity:**
    *   The `adj` array requires $\mathcal{O}(V + E)$ space.
    *   The `stack` and `circuit_with_edges` can hold at most $E + 1$ elements.
    *   Total Space Complexity is strictly bounded by $\mathcal{O}(V + E)$.

The Lean 4 proofs will mechanically verify these limits by bounding the recursive depth and structural sizes of the data structures.
