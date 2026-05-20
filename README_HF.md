---
language:
- en
license: mit
library_name: onnx
tags:
- reinforcement-learning
- graph-neural-networks
- routing
- v2rmp
- drone-logistics
---

# 🛸 Drone RL Agent (GNN-based)

This model is a Reinforcement Learning (RL) policy designed for autonomous drone routing and Vehicle Routing Problems (VRP) within the `v2rmp` engine. It uses a **Graph Convolutional Network (GCN)** architecture to make spatial decisions based on road network topology and real-world physics.

## 🧠 Model Architecture

The agent is a Hybrid GNN-Policy network:
- **GNN Layers:** Two-layer Graph Convolutional Network (GCN) that aggregates 64-dimensional FastRP node embeddings from the local neighborhood.
- **Policy Head:** An Actor MLP that fuses graph features with the drone's dynamic meta-state.
- **Output:** A score for every node in the graph, representing the "desirability" of visiting that node next.

## 📊 State Representation

The agent "sees" the world through a 67-dimensional state vector per node:
1. **Static Features (64-dim):** FastRP embeddings capturing the geometry and importance of intersections.
2. **Dynamic Meta-state (3-dim):**
   - **Wind Vector:** Normalized wind influence.
   - **Battery SOC:** Current state of charge.
   - **Payload:** Normalized weight remaining.

## 🚀 Usage in v2rmp

This model is designed to be loaded by the `v2rmp` Rust engine using the `ort` (ONNX Runtime) solver.

### Deployment logic:
1. **Construction:** The agent performs greedy decoding to build an initial route, prioritizing nodes with favorable wind and high GNN scores.
2. **Optimization:** The resulting route is polished using 2-Opt local search.

## 🛠 Training
The model was trained using **Hugging Face Jobs** with a simulation environment that mimics the `v2rmp` elevation and energy consumption modules.

- **Optimizer:** Adam
- **Reward Function:** Efficiency-focused (Delivery Reward - Energy Penalty - Time Penalty).
- **Environment:** Mile End, Montreal road network.

## 📋 Input/Output Schema

### Inputs:
- `node_embeddings`: `[num_nodes, 64]` (float32)
- `adj_matrix`: `[num_nodes, num_nodes]` (float32)
- `meta_state`: `[1, 3]` (float32)

### Outputs:
- `node_scores`: `[num_nodes, 1]` (float32)
