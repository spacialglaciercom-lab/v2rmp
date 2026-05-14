# Formal Verification Specification: Hierholzer's Algorithm in Route Optimization

This document outlines the rigorous mathematical specification for the Chinese Postman Problem (CPP) route optimization implemented in `src/core/optimize.rs`. It provides the foundational theorems, invariants, and state mappings required to prove algorithmic correctness and termination bounds in the **Lean 4** theorem prover.

## 1. Formal Mathematical Definitions

Let $G = (V, E)$ be a directed multigraph representing the road network, where:
*   $V$ is the set of vertices (nodes) indexed by $0 \le i < |V|$.
*   $E$ is a multiset of directed edges. An edge is represented as a tuple $e = (u, v, w, i)$ where $u \in V$ is the source, $v \in V$ is the destination, $w \in \mathbb{R}^+$ is the weight (distance), and $i \in \mathbb{N}$ is a unique edge index.

The algorithm constructs an adjacency list $adj: V \to \text{List}(E)$.

Let $E_{dup}$ be the set of duplicate deadhead edges added by the Minimum-Weight Perfect Matching (MWPM) step.
Let $G' = (V, E \cup E_{dup})$ be the augmented multigraph, represented in code by the fully populated `adj` array before Hierholzer's algorithm begins.

## 2. Invariants for Lean 4 Formalization

To verify the correctness of `solve_cpp`, the following mathematical invariants must be proven to hold over $G'$ and the algorithm state.

### 2.1 Eulerian Parity Invariant (Handshaking Lemma Consequence)
After the exact DP matching (or greedy fallback) matches all odd-degree vertices and appends paths to $E_{dup}$, the total degree (in-degree + out-degree) of every vertex in $G'$ is even. Since $G'$ is modeled as undirected edges pushed bidirectionally into $adj$, the length of `adj[v]` for all $v \in V$ is even.
*   **Lean 4 Theorem:** `∀ v ∈ V, adj[v].length % 2 = 0`

### 2.2 Edge Conservation Invariant
At any point during the execution of Hierholzer's loop, the total number of edges is conserved across three collections: the remaining adjacency list, the execution stack, and the final circuit list.
*   Let $E_{adj}$ be the multiset of edges currently in `adj`. Note that because edges are stored bidirectionally, $|E_{adj}|$ represents $2 \times$ the number of untraversed physical edges.
*   Let $E_{stack}$ be the sequence of edges currently held in `stack`.
*   Let $E_{circuit}$ be the sequence of edges currently held in `circuit_with_edges`.
*   **Lean 4 Invariant:** `(|E_{adj}| / 2) + |E_{stack}| + |E_{circuit}| = |E| + |E_{dup}|`

### 2.3 Valid Walk Invariant
The elements in `stack` must always form a continuous walk in $G'$.
*   If `stack` contains $[(v_0, e_0), (v_1, e_1), \dots, (v_k, e_k)]$, then for all $0 < j \le k$, the edge $e_j$ must transition from $v_{j-1}$ to $v_j$.
*   **Lean 4 Invariant:** `∀ j ∈ [1, k], e_j.from = v_{j-1} ∧ e_j.to = v_j`

## 3. State Mutations

Hierholzer's algorithm loop mutates state exactly $2 \times (|E| + |E_{dup}|) + 1$ times. The Lean 4 model must map these `O(1)` and `O(deg(v))` operations:

1.  **Forward Edge Traversal:**
    *   `adj[v].pop()` removes the last outgoing edge $e = (v, u, w, i)$ from $v$.
    *   **Complexity:** $\mathcal{O}(1)$.
2.  **Reverse Edge Pruning:**
    *   To prevent immediate backtracking, the corresponding reverse edge $(u, v, w, i)$ must be removed from `adj[u]`.
    *   The position `pos` is found via linear search: `adj[edge.to].iter().position(|e| e.to == v && e.edge_idx == edge.edge_idx)`.
    *   `adj[u].swap_remove(pos)` deletes it.
    *   **Complexity:** $\mathcal{O}(\text{deg}(u))$.
3.  **Path Extension / Backtracking:**
    *   If an edge is popped, the destination node $u$ is pushed: `stack.push((u, Some(e)))`. $\mathcal{O}(1)$.
    *   If `adj[v]` is empty, the node is fully explored. It is popped from `stack` and prepended (pushed) to `circuit_with_edges`. $\mathcal{O}(1)$.

## 4. Boundary Conditions and Edge Cases

The Lean 4 proofs must account for:
*   **Empty Graph ($V=\emptyset, E=\emptyset$):** `solve_cpp` immediately returns an empty circuit.
*   **Disconnected Components:** Hierholzer's algorithm operates solely on the connected component containing `start_node`. Edges in unreachable components are never pushed to `stack` and remain in `adj` indefinitely. The conservation invariant holds locally for the reachable component.
*   **Zero Odd Vertices:** If $V_{\text{odd}} = \emptyset$, the graph is natively Eulerian. $E_{dup} = \emptyset$, MWPM is skipped entirely, and `adj` is unmutated prior to the loop.
*   **Degenerate Single Edge:** An isolated graph of 2 nodes and 1 edge results in an odd-degree pair. MWPM adds a duplicate edge (deadhead), resulting in a closed walk $v_0 \to v_1 \to v_0$.

## 5. Termination Measure

To prove in Lean 4 that the `while let Some(&(v_u32, _)) = stack.last()` loop terminates, we establish a strictly monotonically decreasing measure.

*   Let $S$ be the state tuple `(RemainingEdges, StackLength)`, where `RemainingEdges` is the total count of elements across all `adj` arrays (i.e., $\sum_{v \in V} |adj[v]|$).
*   We define a **lexicographical ordering** on $S$.

**Proof of strict decrease:**
In every iteration, exactly one of two branches executes:
1.  **If `adj[v]` is not empty:** An edge is popped from `adj[v]`, and its reverse is removed from `adj[u]`. `RemainingEdges` decreases by 2. `StackLength` increases by 1. Since `RemainingEdges` decreased, the lexicographical measure $S$ strictly decreases.
2.  **If `adj[v]` is empty:** No edges are removed from `adj` (`RemainingEdges` remains constant). A node is popped from `stack`. `StackLength` decreases by 1. Therefore, the lexicographical measure $S$ strictly decreases.

Because $S \ge (0, 0)$ is bounded below, the lexicographical decrease proves that the loop terminates in $\mathcal{O}(|E_{total}|)$ iterations.
