# rmpca Project Roadmap

This document outlines the future goals and planned enhancements for the **rmpca** (Route Optimization TUI & Agent Engine).

## 🟢 Phase 1: VRP & Agent Core (v0.4.0 - Current)
- [x] Multi-vehicle VRP Engine (Clarke-Wright, Sweep, Local Search)
- [x] Agent-first CLI interface (`agent` command)
- [x] Resource discovery CLI (`list` command)
- [x] Fully asynchronous pipeline (Tokio)
- [x] Local OSM PBF extraction support

## 🟡 Phase 2: Advanced Routing & Constraints (v0.5.0)
- [ ] **Time Window Support (VRPTW)**: Implement time-constrained deliveries.
- [ ] **Capacity Constraints**: Enhanced vehicle capacity modeling.
- [ ] **3D Terrain-Aware Routing**: Adjust weights/penalties based on elevation changes.
- [ ] **Multi-Depot Support**: Optimize routes starting/ending at different locations.

## 🟠 Phase 3: Ecosystem & Visualization (v0.6.0)
- [ ] **Web-Based Visualization**: Export routes to interactive HTML/Leaflet maps.
- [ ] **Plugin Architecture**: Support for custom VRP solvers via dynamic loading or WASM.
- [ ] **Real-time Traffic Integration**: Optional hooks for traffic-aware edge weighting.

## 🔴 Phase 4: Enterprise & Scale (v1.0.0)
- [ ] **Clustered Processing**: Distributed optimization for massive city-scale graphs.
- [ ] **Stable C/FFI Bindings**: Allow integration into other languages (Python, Go).
- [ ] **Commercial-Grade TUI**: Advanced dashboard for fleet monitoring.

---
*Roadmap updated: 2026-05-04*
