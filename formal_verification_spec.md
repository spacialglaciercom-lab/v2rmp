# Formal Verification Specification: Hierholzer's Algorithm in Route Optimization

This document outlines the formal verification requirements for Lean 4, mapping the mathematical invariants, boundary conditions, and state mutations for the core pathfinding and Hierholzer's algorithm implementation in `src/core/optimize.rs`.

## 1. Algorithm Overview

The module `src/core/optimize.rs` implements a Chinese Postman Problem (CPP) route optimization on multigraphs representing road networks. The core process relies on:
1. Converting the multigraph into an Eulerian multigraph via minimum weight perfect matching on odd-degree vertices.
2. Finding an Eulerian circuit using Hierholzer's algorithm.

## 2. Mathematical Invariants & Properties

To verify correctness in Lean 4, the following properties must hold:

*   **Eulerian Graph Property:** After adding duplicate edges (Step 5 in the code), the degree of every vertex in the adjacency list `adj` must be even.
    *   *Invariant:* `∀ v ∈ V, degree(v) ≡ 0 (mod 2)`
*   **Edge Conservation:** The sum of all edges initially in `adj` must equal the total number of edges traversed in the circuit. Each edge must be visited exactly once.
*   **Hierholzer Loop Invariant:** At any point during the while loop execution:
    *   `edges_remaining_in(adj) + edges_in(stack) + edges_in(circuit_with_edges) = total_edges`
    *   The `stack` maintains a valid path sequence of vertices extending from `start_node`.
    *   The sequence in `circuit_with_edges` collects closed Eulerian cycles correctly attached to the nodes currently in `stack` as backtracking occurs.
*   **Termination:** The Lean 4 formal verification models termination using a well-founded lexicographically decreasing measure based on `(total remaining edges in the adjacency list, stack length)`. In each iteration, exactly one of two things happens:
    *   An edge is popped from `adj` (total remaining edges decreases).
    *   Or, if `adj` for the current vertex is empty, an element is popped from `stack` (stack length decreases while remaining edges stays constant).
    *   This strictly decreasing measure guarantees termination.

## 3. Boundary Conditions

The Lean 4 proofs must account for the following edge cases:

*   **Empty Graph:** If the graph has 0 nodes or 0 edges (`nodes.is_empty() || edges.is_empty()`), the optimization correctly short-circuits and returns an empty circuit with zero distance, without panicking or creating invalid states.
*   **Disconnected Graphs:** The algorithm runs Hierholzer's on the connected component containing `start_node`. If the graph has multiple disconnected components, edges in unreachable components remain in `adj` indefinitely, and only the connected component of `start_node` is traversed.
*   **Zero Odd Vertices:** If all vertices already have even degrees initially, the graph is inherently Eulerian. The duplicate edge list will be empty, and Hierholzer's algorithm begins immediately without modifications.
*   **Parallel Edges (Multigraph):** Multiple edges can exist between the exact same pair of nodes. The reverse-edge removal logic relies on an exact `position` search (`iter().position(...)`) checking the exact tuple `(to, edge_idx, weight_m)` to uniquely identify and remove the correct reverse edge, preventing the accidental deletion of distinct parallel edges.

## 4. State Mutations

The algorithm mutates state primarily in four ways during the execution of Hierholzer's loop:

1.  `adj[v].pop()`: Mutates the adjacency list of vertex `v` by removing the last outgoing edge.
    *   *Complexity:* `O(1)`
2.  `adj[edge.to as usize].swap_remove(pos)`: Mutates the adjacency list of the destination vertex to remove the corresponding reverse edge. The position `pos` is found via linear search (`iter().position(...)`).
    *   *Complexity:* `O(degree(edge.to))` for search, `O(1)` for removal.
3.  `stack.push(...)` / `stack.pop()`: Modifies the path stack to keep track of the current path and backtracking edges.
    *   *Complexity:* `O(1)`
4.  `circuit_with_edges.push(...)`: Appends to the final reversed Eulerian circuit when `adj` for a given node becomes empty.
    *   *Complexity:* `O(1)` amortized

## 5. Algorithmic Complexity Bounds

Let $V$ be the number of vertices and $E$ be the total number of edges (original + duplicated).

*   **Odd Vertices & Matching:** Finding odd vertices is $\mathcal{O}(V)$. The greedy minimum-weight perfect matching involves sorting odd vertices $\mathcal{O}(V_{\text{odd}} \log V_{\text{odd}})$ and a linear search bounded by $\mathcal{O}(V_{\text{odd}}^2)$.
*   **Hierholzer's Algorithm Loop:**
    *   The `while` loop runs exactly $2E + 1$ times.
    *   Finding the reverse edge in `adj` requires an $\mathcal{O}(\text{degree}(v))$ search.
    *   Overall Time Complexity: bounded by $\mathcal{O}(E \times \text{max\_degree})$.
    *   Space Complexity: $\mathcal{O}(E)$ for the `stack` and `circuit_with_edges` collections.

The formal verification suite should prove these complexity bounds as part of ensuring the implementation's efficiency guarantees.
