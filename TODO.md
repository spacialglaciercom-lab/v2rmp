# TODO — Map Visualization Improvements

The current map canvas in the egui GUI renders nodes/edges as raw points and lines
using `egui::Painter`. This works for small graphs but has significant limitations
for real-world road networks (Montreal has ~35k+ nodes).

---

## MapLibre RS Feasibility Analysis

### What is maplibre-rs?

[maplibre-rs](https://github.com/maplibre/maplibre-rs) is a Rust-based map rendering
engine using **WebGPU** (via `wgpu 22.x` + `lyon` tessellation). It supports vector
tile rendering, a MapLibre Style Spec, and has recently gained GeoJSON source support
(PR #331, Feb 2026). It targets desktop (winit), mobile, and web (WASM).

### Current state (as of 2026-05)

| Aspect | Status |
|---|---|
| crates.io version | `0.0.3` (last published Feb 2023, GitHub at `0.1.0`) |
| Activity | Low — a few PRs merged in early 2026, large gaps in 2024-2025 |
| Stability badge | **Experimental** (per their own README) |
| GeoJSON sources | Added in PR #331 (Feb 2026) — minimal, not mature |
| Text/labels | Partial SDF text rendering (PR #314), no labels yet |
| Raster tiles | Basic support |
| Style spec coverage | Partial — background, fill, line, symbol layers |
| No egui integration | Uses its own winit window loop; no embeddable surface API |

### Integration blockers with v2rmp

1. **Rendering backend conflict**: v2rmp uses `eframe` with the **glow** (OpenGL)
   backend. maplibre-rs uses **wgpu** (WebGPU/Vulkan/Metal). They cannot share a
   rendering context. eframe *does* have a `wgpu` backend feature, but switching
   would require changing `Cargo.toml` and potentially breaking WASM support
   (glow is more portable for WASM today).

2. **No embeddable surface**: maplibre-rs owns its own window (`maplibre-winit`).
   There is no API to render into an existing egui surface or texture. Embedding
   it would require forking maplibre-rs to add a "render-to-texture" or
   "render-to-egui-paint-callback" pathway.

3. **GeoJSON support is nascent**: While PR #331 added `GeoJsonSource`, it's
   designed for standard GeoJSON feature collections, not for the raw graph data
   (RmpNode/RmpEdge) that v2rmp produces from `.rmp` files. We'd need to convert
   our data to GeoJSON FeatureCollection before feeding it to maplibre-rs.

4. **No custom overlay API**: maplibre-rs doesn't expose a way to draw custom
   shapes (CPP circuit, depot markers, bbox rectangles, deadhead highlights) on
   top of the base map tiles. This is core to v2rmp's visualization needs.

5. **Missing features**: No labels, no symbol layers, no collision detection,
   no raster overlays — all things a production map viewer needs.

### Alternative: `maplibre_native` (Rust bindings to C++ MapLibre Native)

- crates.io version `0.4.5`, more mature, actively maintained by the MapLibre org.
- Renders vector + raster tiles with full style spec support.
- **Same blocker**: No egui integration. Uses its own render surface (C++ with
  platform-native rendering). Would need an FFI bridge to render into an egui
   texture — significant engineering effort.
- Licensing: MapLibre Native is BSD-2, but it requires a C++ build toolchain.

### Recommendation: Phased approach

**Short term (now)**: Optimize the existing `egui::Painter` canvas. It's the
fastest path to a usable map for 5k–50k node networks. The performance items
below (viewport culling, batch rendering, LOD) will handle real-world networks.

**Medium term (if tile-based basemap needed)**: Add a **tile fetcher + static
image layer** approach. Fetch raster map tiles (from OSM/Mapbox) into an
`egui::TextureHandle`, composite the network overlay on top. No maplibre-rs
needed — just HTTP tile fetching + mercator projection.

**Long term (if interactive map needed)**: Re-evaluate maplibre-rs once it has:
- A stable 1.0 release
- A "render-to-texture" or embeddable surface API
- Mature GeoJSON + custom overlay support
- Or switch the entire GUI to a web frontend with maplibre-gl-js.

---

## Critical — Performance (egui::Painter)

- [ ] **Viewport culling**: Only draw edges whose screen-space endpoints are inside
  the visible canvas rect. Skip off-screen nodes/edges entirely. This is the single
  biggest perf win for large maps.

- [ ] **Level-of-detail (LOD)**: When zoomed out, skip drawing individual nodes
  (render edges only). When very zoomed out, draw only a reduced subset of edges.
  Consider decimating edges by importance (major roads first).

- [ ] **Avoid O(n²) circuit lookup**: The current `circuit.contains(&(i as u32))`
  call on every node is O(n) per node. Replace with a `HashSet<u32>` built once
  from `cpp_output.circuit`.

- [ ] **Batch rendering**: Instead of calling `painter.line_segment` and
  `painter.circle_filled` per edge/node, batch them into a single
  `egui::Shape` or use `painter.add()` with grouped shapes. Fewer draw calls
  = better frame time.

- [ ] **Throttle repaints**: The map canvas calls `ctx.request_repaint()` every
  frame even when nothing changed. Only request repaint when zoom/pan changes or
  new data loads.

## Important — Visual Quality

- [ ] **Edge direction arrows**: Draw small arrowheads on one-way street edges
  so the user can see which direction traffic flows. Use `oneway` field from
  `RmpEdge`.

- [ ] **Road classification colors**: Use `RmpEdge` metadata (if available) or
  edge weight to color-code roads: highways in thick yellow, residential in thin
  gray, etc.

- [ ] **Depot marker**: Draw a distinct icon (star/pin) at the depot location
  when `depot_coords` is set.

- [ ] **Bbox overlay rectangle**: Draw a semi-transparent rectangle showing the
  active bounding box filter on the map, so the user can see what area will be
  optimized.

- [ ] **Circuit animation**: Animate the CPP circuit by drawing it with a
  dashed/animated stroke so the route direction is visible. Could use
  `egui::Stroke::new(width, color)` with a time-based dash offset.

- [ ] **Deadhead edges**: Color deadhead edges (those traversed more than once,
  or matched odd-vertex pairs) differently from regular edges to show where the
  postman backtracks.

## Nice-to-have — Interaction

- [ ] **Click-to-select node/edge**: Let the user click on a node or edge to see
  its properties (lat/lon, weight, oneway status, traversal count).

- [ ] **Click-to-set bbox**: Let the user draw a bounding box on the map by
  dragging, instead of typing coordinates manually.

- [ ] **Click-to-set depot**: Let the user click on the map to set the depot
  position, snapping to the nearest node.

- [ ] **Export map as image**: Add a button to save the current map view as a PNG.

- [ ] **Legend**: Show a color legend (road types, circuit color, deadhead color).

## Medium-term — Basemap Tile Layer

If we need a real geographic basemap (streets, labels, terrain) behind the network
overlay, the simplest path is **not** maplibre-rs but a lightweight tile fetcher:

- [ ] **Tile fetcher module**: Fetch Z/X/Y raster tiles from a tile server
  (e.g., `https://tile.openstreetmap.org/{z}/{x}/{y}.png`) using `reqwest`.
  Cache tiles in `~/.cache/rmpca/tiles/`.

- [ ] **Mercator projection**: Convert lat/lon bounds to tile coordinates, fetch
  the visible tile grid, stitch them into an `egui::TextureHandle`.

- [ ] **Overlay compositing**: Draw the existing network/circuit visualization
  on top of the tile texture using `egui::Painter` with alpha blending.

- [ ] **Tile credits**: Display OSM attribution as required by the tile license.

This approach avoids the wgpu/glow conflict entirely and works within the
existing egui rendering pipeline.

## Long-term — MapLibre RS Integration (blocked)

Blocked until maplibre-rs reaches stability. Track these milestones:

- [ ] maplibre-rs publishes a stable `0.x` or `1.0` crate
- [ ] maplibre-rs adds a "render-to-texture" or embeddable surface API
- [ ] maplibre-rs GeoJSON source support is production-ready
- [ ] maplibre-rs supports custom overlay layers (for CPP circuit, depot, bbox)
- [ ] Evaluate switching eframe from `glow` to `wgpu` backend (or dual support)
- [ ] Prototype: embed maplibre-rs render output as `egui::PaintCallback`
  using wgpu texture sharing within an egui wgpu backend

## Existing Known Issues

- [ ] When `set_network` is called after bbox-filtered CPP, the map switches
  from showing the full map to showing only the filtered subset. Should consider
  keeping the full map visible but highlighting the bbox-filtered region, with
  the circuit drawn only on the subset.

- [ ] `map_solve_error` is set on error but never displayed in the UI.

- [ ] The `map_solving` bool flag exists on `GuiApp` but is never set to `true`
  (no background solving thread exists yet).

- [ ] Cursor lat/lon readout uses a simple linear projection that's inaccurate
  at high zoom. Should use inverse of `project_latlon`.
