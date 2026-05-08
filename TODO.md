# TODO — Map Visualization Improvements

The current map canvas in the egui GUI renders nodes/edges as raw points and lines
using `egui::Painter`. This works for small graphs but has significant limitations
for real-world road networks (Montreal has ~35k+ nodes). Below is a prioritized
list of what needs to be done.

## Critical — Performance

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

## Existing Known Issues

- [ ] When `set_network` is called after bbox-filtered CPP, the map switches
  from showing the full map to showing only the filtered subset. Should consider
  keeping the full map visible but highlighting the bbox-filtered region, with
  the circuit drawn only on the subset.

- [ ] `map_solve_error` is set on error but never displayed in the UI.

- [ ] The `map_solving` bool flag exists on `GuiApp` but is never set to `true`
  (no background solving thread exists yet).

- [ ] Cursor lat/lon readout (line 662-679) uses a simple linear projection
  that's inaccurate at high zoom. Should use inverse of `project_latlon`.
