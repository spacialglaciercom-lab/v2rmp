# Vector Cleaning Pipeline - Rust Implementation Design

## Overview

This document outlines the design for a Rust equivalent of the Python `vector_clean.py` module, adapted for the v2rmp codebase.

## Python Implementation Analysis

### Key Features
1. **Geometry Validation & Repair**
   - Make invalid geometries valid
   - Drop invalid features
   - Convert Polygons to LineStrings (extract rings)
   - Handle MultiLineString/MultiPolygon

2. **Graph-Based Topology Cleaning**
   - Build NetworkX MultiGraph from LineString features
   - Node deduplication with spatial proximity (STRtree + haversine)
   - Remove self-loops
   - Remove short edges (< min_length_m)
   - Remove duplicate edges (same geometry hash)
   - Merge parallel edges (same u,v pair)
   - Remove isolated nodes
   - Keep only largest N connected components

3. **Property Management**
   - Preserve feature properties on edges
   - Merge properties when combining parallel edges
   - Filter edges by required attributes

4. **Performance Optimizations**
   - Batch processing with GeoPandas for large datasets (>10k features)
   - Vectorized distance calculations
   - Spatial indexing (STRtree) for proximity queries

### Configuration Options
```python
class CleanOptions:
    makevalid: bool = True
    drop_invalid: bool = True
    remove_selfloops: bool = True
    min_length_m: float = 0.1
    node_snap_m: float = 1.0
    node_precision_decimals: int = 6
    merge_node_positions: bool = True
    dedupe_edges: bool = True
    remove_isolates: bool = True
    max_components: int = 1
    required_attrs: list[str] | None = None
    merge_parallel_edges: bool = False
    simplify_tolerance_m: float = 0.0
    include_polygons: bool = False
    include_points: bool = False
```

---

## Rust Implementation Design

### Architecture

```
src/core/clean.rs          # Main cleaning pipeline
src/core/clean/
  ├── geometry.rs          # Geometry validation & repair
  ├── graph.rs             # Graph construction & operations
  ├── spatial.rs           # Spatial indexing & proximity
  └── stats.rs             # Statistics tracking
```

### Dependencies Required

```toml
[dependencies]
# Existing
geo-types = "0.7"
geojson = "0.24"

# New for cleaning
geo = "0.28"              # Geometry operations (validation, simplification)
petgraph = "0.6"          # Graph data structures & algorithms
rstar = "0.12"            # R-tree spatial index (faster than linear search)
ordered-float = "4.2"     # Hashable floats for deduplication
```

### Core Data Structures

```rust
// Clean options (similar to Python CleanOptions)
pub struct CleanOptions {
    pub make_valid: bool,
    pub drop_invalid: bool,
    pub remove_selfloops: bool,
    pub min_length_m: f64,
    pub node_snap_m: f64,
    pub node_precision_decimals: u32,
    pub merge_node_positions: bool,
    pub dedupe_edges: bool,
    pub remove_isolates: bool,
    pub max_components: usize,
    pub required_attrs: Option<Vec<String>>,
    pub merge_parallel_edges: bool,
    pub simplify_tolerance_m: f64,
    pub include_polygons: bool,
    pub include_points: bool,
}

// Statistics (similar to Python CleanStats)
pub struct CleanStats {
    pub input_features: usize,
    pub output_features: usize,
    pub invalid_dropped: usize,
    pub selfloops_removed: usize,
    pub short_edges_removed: usize,
    pub nodes_merged: usize,
    pub duplicate_edges_removed: usize,
    pub incomplete_edges_removed: usize,
    pub parallel_edges_merged: usize,
    pub isolates_removed: usize,
    pub components_removed: usize,
}

// Node representation
#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,           // "lon,lat" rounded to precision
    pub lon: f64,
    pub lat: f64,
}

// Edge representation
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub length_m: f64,
    pub coords: Vec<[f64; 2]>,
    pub properties: serde_json::Value,
}

// Graph type
type RoadGraph = petgraph::graph::UnGraph<Node, Edge>;
```

### Implementation Phases

#### Phase 1: Core Infrastructure (Week 1)
- [ ] Add dependencies to Cargo.toml
- [ ] Create `src/core/clean.rs` module structure
- [ ] Implement `CleanOptions` and `CleanStats`
- [ ] Implement haversine distance calculation
- [ ] Implement node ID generation (rounded coordinates)

#### Phase 2: Geometry Operations (Week 1-2)
- [ ] Implement geometry validation using `geo` crate
- [ ] Implement geometry repair (make_valid equivalent)
- [ ] Implement Polygon → LineString conversion
- [ ] Implement coordinate simplification (Douglas-Peucker)
- [ ] Handle MultiLineString/MultiPolygon

#### Phase 3: Graph Construction (Week 2)
- [ ] Parse GeoJSON features into graph nodes and edges
- [ ] Build `petgraph::UnGraph` from LineString features
- [ ] Store geometry and properties on edges
- [ ] Calculate edge lengths using haversine

#### Phase 4: Spatial Indexing (Week 2-3)
- [ ] Implement R-tree spatial index using `rstar`
- [ ] Implement proximity-based node merging
- [ ] Union-find algorithm for node clustering
- [ ] Update edge endpoints after node merging

#### Phase 5: Topology Cleaning (Week 3)
- [ ] Remove self-loops
- [ ] Remove short edges (< min_length_m)
- [ ] Deduplicate edges (geometry hash)
- [ ] Merge parallel edges
- [ ] Remove isolated nodes
- [ ] Keep largest N components (BFS/DFS)

#### Phase 6: Integration (Week 3-4)
- [ ] Integrate with existing compile pipeline
- [ ] Add TUI view for clean options
- [ ] Add event handlers for clean workflow
- [ ] Export cleaned graph back to GeoJSON

#### Phase 7: Testing & Optimization (Week 4)
- [ ] Unit tests for each cleaning operation
- [ ] Integration tests with real-world data
- [ ] Performance benchmarks
- [ ] Memory optimization for large datasets

---

## Key Implementation Details

### 1. Geometry Validation (geo crate)

```rust
use geo::{LineString, Point, Polygon};
use geo::algorithm::simplify::Simplify;
use geo::algorithm::contains::Contains;

fn make_valid_linestring(line: &LineString<f64>) -> Option<LineString<f64>> {
    // Remove duplicate consecutive points
    let mut coords: Vec<_> = line.coords().collect();
    coords.dedup();
    
    if coords.len() < 2 {
        return None;
    }
    
    Some(LineString::from(coords))
}

fn simplify_linestring(line: &LineString<f64>, tolerance_m: f64) -> LineString<f64> {
    // Convert meters to degrees (approximate at equator)
    let tolerance_deg = tolerance_m / 111_320.0;
    line.simplify(&tolerance_deg)
}
```

### 2. Spatial Indexing (rstar crate)

```rust
use rstar::{RTree, AABB};
use ordered_float::OrderedFloat;

#[derive(Debug, Clone)]
struct NodePoint {
    id: String,
    lon: OrderedFloat<f64>,
    lat: OrderedFloat<f64>,
}

impl rstar::RTreeObject for NodePoint {
    type Envelope = AABB<[OrderedFloat<f64>; 2]>;
    
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.lon, self.lat])
    }
}

fn build_spatial_index(nodes: &[Node]) -> RTree<NodePoint> {
    let points: Vec<_> = nodes.iter()
        .map(|n| NodePoint {
            id: n.id.clone(),
            lon: OrderedFloat(n.lon),
            lat: OrderedFloat(n.lat),
        })
        .collect();
    
    RTree::bulk_load(points)
}

fn find_nearby_nodes(
    tree: &RTree<NodePoint>,
    lon: f64,
    lat: f64,
    radius_m: f64,
) -> Vec<String> {
    // Convert radius to degrees (latitude-aware)
    let delta_lat = radius_m / 111_320.0;
    let cos_lat = (lat.to_radians().cos()).max(0.01);
    let delta_lon = radius_m / (111_320.0 * cos_lat);
    
    let envelope = AABB::from_corners(
        [OrderedFloat(lon - delta_lon), OrderedFloat(lat - delta_lat)],
        [OrderedFloat(lon + delta_lon), OrderedFloat(lat + delta_lat)],
    );
    
    tree.locate_in_envelope(&envelope)
        .map(|p| p.id.clone())
        .collect()
}
```

### 3. Graph Operations (petgraph crate)

```rust
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::algo::connected_components;
use petgraph::visit::Dfs;

fn remove_selfloops(graph: &mut UnGraph<Node, Edge>) -> usize {
    let mut removed = 0;
    let edges: Vec<_> = graph.edge_indices().collect();
    
    for edge_idx in edges {
        if let Some((a, b)) = graph.edge_endpoints(edge_idx) {
            if a == b {
                graph.remove_edge(edge_idx);
                removed += 1;
            }
        }
    }
    
    removed
}

fn remove_short_edges(
    graph: &mut UnGraph<Node, Edge>,
    min_length_m: f64,
) -> usize {
    let mut removed = 0;
    let edges: Vec<_> = graph.edge_indices().collect();
    
    for edge_idx in edges {
        if let Some(edge) = graph.edge_weight(edge_idx) {
            if edge.length_m < min_length_m {
                graph.remove_edge(edge_idx);
                removed += 1;
            }
        }
    }
    
    removed
}

fn keep_largest_components(
    graph: &mut UnGraph<Node, Edge>,
    max_components: usize,
) -> usize {
    let num_components = connected_components(graph);
    
    if num_components <= max_components {
        return 0;
    }
    
    // Find component sizes
    let mut component_sizes: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    let mut visited = HashSet::new();
    
    for node in graph.node_indices() {
        if visited.contains(&node) {
            continue;
        }
        
        let mut component = Vec::new();
        let mut dfs = Dfs::new(&*graph, node);
        
        while let Some(n) = dfs.next(&*graph) {
            visited.insert(n);
            component.push(n);
        }
        
        component_sizes.insert(component.len(), component);
    }
    
    // Keep largest N components
    let mut sizes: Vec<_> = component_sizes.keys().copied().collect();
    sizes.sort_by(|a, b| b.cmp(a));
    
    let mut keep_nodes = HashSet::new();
    for size in sizes.iter().take(max_components) {
        if let Some(nodes) = component_sizes.get(size) {
            keep_nodes.extend(nodes);
        }
    }
    
    // Remove nodes not in largest components
    let all_nodes: Vec<_> = graph.node_indices().collect();
    let mut removed = 0;
    
    for node in all_nodes {
        if !keep_nodes.contains(&node) {
            graph.remove_node(node);
            removed += 1;
        }
    }
    
    num_components - max_components
}
```

### 4. Node Merging with Union-Find

```rust
use std::collections::HashMap;

struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
        }
    }
    
    fn find(&mut self, x: &str) -> String {
        if !self.parent.contains_key(x) {
            self.parent.insert(x.to_string(), x.to_string());
            return x.to_string();
        }
        
        let parent = self.parent[x].clone();
        if parent != x {
            let root = self.find(&parent);
            self.parent.insert(x.to_string(), root.clone());
            root
        } else {
            parent
        }
    }
    
    fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        
        if ra != rb {
            // Canonical = min by string order
            let canonical = if ra < rb { ra } else { rb };
            self.parent.insert(ra, canonical.clone());
            self.parent.insert(rb, canonical);
        }
    }
}

fn merge_nearby_nodes(
    graph: &mut UnGraph<Node, Edge>,
    node_snap_m: f64,
    decimals: u32,
) -> usize {
    let nodes: Vec<_> = graph.node_indices().collect();
    let tree = build_spatial_index(
        &nodes.iter()
            .filter_map(|&idx| graph.node_weight(idx))
            .cloned()
            .collect::<Vec<_>>()
    );
    
    let mut uf = UnionFind::new();
    
    // Find nearby nodes and union them
    for &node_idx in &nodes {
        if let Some(node) = graph.node_weight(node_idx) {
            let nearby = find_nearby_nodes(&tree, node.lon, node.lat, node_snap_m);
            
            for other_id in nearby {
                if other_id != node.id {
                    // Check actual distance
                    if let Some(other_node) = nodes.iter()
                        .filter_map(|&idx| graph.node_weight(idx))
                        .find(|n| n.id == other_id)
                    {
                        let dist_m = haversine_distance_m(
                            node.lon, node.lat,
                            other_node.lon, other_node.lat,
                        );
                        
                        if dist_m <= node_snap_m {
                            uf.union(&node.id, &other_id);
                        }
                    }
                }
            }
        }
    }
    
    // Build mapping: old_id -> canonical_id
    let mut mapping: HashMap<String, String> = HashMap::new();
    for &node_idx in &nodes {
        if let Some(node) = graph.node_weight(node_idx) {
            let canonical = uf.find(&node.id);
            mapping.insert(node.id.clone(), canonical);
        }
    }
    
    // Count merged nodes
    let merged_count = mapping.iter()
        .filter(|(k, v)| k != v)
        .count();

    // Create new graph with merged nodes
    let mut new_graph = UnGraph::new_undirected();
    let mut new_node_map: HashMap<String, NodeIndex> = HashMap::new();

    // Add canonical nodes
    for canonical_id in mapping.values().collect::<HashSet<_>>() {
        if let Some(node) = find_node_by_id(&nodes, canonical_id) {
            let new_idx = new_graph.add_node(node.clone());
            new_node_map.insert(canonical_id.clone(), new_idx);
        }
    }

    // Add edges with updated endpoints
    for edge_idx in graph.edge_indices() {
        if let (Some((old_a, old_b)), Some(edge)) = (graph.edge_endpoints(edge_idx), graph.edge_weight(edge_idx)) {
            let canonical_a = mapping[&graph.node_weight(old_a).unwrap().id];
            let canonical_b = mapping[&graph.node_weight(old_b).unwrap().id];

            if canonical_a != canonical_b {
                if let (Some(&new_a), Some(&new_b)) = (new_node_map.get(&canonical_a), new_node_map.get(&canonical_b)) {
                    new_graph.add_edge(new_a, new_b, edge.clone());
                }
            }
        }
    }
    
    *graph = new_graph;
    
    merged_count
}
```

---

## Integration with v2rmp

### 1. Add to Compile Pipeline

```rust
// In src/core/compile.rs

pub struct CompileRequest {
    pub input_geojson: String,
    pub output_rmp: String,
    pub compress: bool,
    pub road_classes: Vec<String>,
    pub clean_options: Option<CleanOptions>,  // NEW
}

pub fn run_compile(req: &CompileRequest) -> anyhow::Result<CompileResult> {
    // ... existing code ...
    
    // NEW: Clean GeoJSON before compilation
    if let Some(clean_opts) = &req.clean_options {
        let (cleaned_fc, clean_stats) = clean_geojson(&geojson, clean_opts)?;
        
        // Log cleaning stats
        println!("Cleaning stats:");
        println!("  Input features: {}", clean_stats.input_features);
        println!("  Output features: {}", clean_stats.output_features);
        println!("  Nodes merged: {}", clean_stats.nodes_merged);
        println!("  Edges removed: {}", 
            clean_stats.selfloops_removed + 
            clean_stats.short_edges_removed + 
            clean_stats.duplicate_edges_removed
        );
        
        // Use cleaned GeoJSON for compilation
        geojson = cleaned_fc;
    }
    
    // ... rest of compilation ...
}
```

### 2. Add TUI View

```rust
// In src/app.rs

pub enum View {
    Home,
    Extract,
    Compile,
    Clean,      // NEW
    Optimize,
    BrowseMaps,
    BrowseRoutes,
    FileBrowser,
    Help,
}

pub struct App {
    // ... existing fields ...
    
    // Clean state
    pub clean_options: CleanOptions,
    pub clean_status: Status,
}
```

### 3. Add UI Component

```rust
// In src/ui/clean.rs

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    // Display clean options with toggles
    // - [x] Make valid geometries
    // - [x] Remove self-loops
    // - [x] Remove short edges (< 0.1m)
    // - [x] Merge nearby nodes (< 1.0m)
    // - [x] Deduplicate edges
    // - [x] Remove isolates
    // - [x] Keep largest component
    // 
    // [Enter] Start cleaning
    // [Esc] Return to home
}
```

---

## Performance Considerations

### Memory Usage
- **Large datasets**: Stream processing for >100k features
- **Spatial index**: R-tree has O(log n) query time, O(n log n) build time
- **Graph**: petgraph uses adjacency list (efficient for sparse graphs)

### Optimization Strategies
1. **Batch processing**: Process features in chunks of 10k
2. **Parallel processing**: Use rayon for independent operations
3. **Lazy evaluation**: Only compute what's needed
4. **Memory pooling**: Reuse allocations for repeated operations

### Benchmarks (Target)
- **Small dataset** (1k features): < 100ms
- **Medium dataset** (10k features): < 1s
- **Large dataset** (100k features): < 10s
- **Very large dataset** (1M features): < 2min

---

## Testing Strategy

### Unit Tests
- Geometry validation and repair
- Distance calculations (haversine)
- Node merging (union-find)
- Edge deduplication
- Component analysis

### Integration Tests
- Full pipeline with real-world GeoJSON
- Edge cases (empty, invalid, degenerate geometries)
- Large dataset performance

### Test Data
- Small synthetic datasets (10-100 features)
- Real-world OSM extracts (1k-10k features)
- Stress test datasets (100k+ features)

---

## Future Enhancements

1. **Advanced Cleaning**
   - Topology repair (fix intersections, gaps)
   - Road network simplification (merge collinear segments)
   - Attribute validation and normalization

2. **Performance**
   - GPU acceleration for distance calculations
   - Distributed processing for very large datasets
   - Incremental cleaning (only process changed features)

3. **Visualization**
   - Before/after comparison
   - Highlight cleaned areas
   - Statistics dashboard

4. **Export Formats**
   - Support multiple output formats (GeoJSON, Shapefile, GeoPackage)
   - Preserve CRS information
   - Metadata export (cleaning report)

---

## Conclusion

This design provides a comprehensive Rust implementation of the Python vector cleaning pipeline, adapted for the v2rmp codebase. The implementation leverages Rust's performance and type safety while maintaining feature parity with the Python version.

**Estimated Implementation Time**: 3-4 weeks
**Complexity**: High (geometry operations, graph algorithms, spatial indexing)
**Dependencies**: 4 new crates (geo, petgraph, rstar, ordered-float)
**Integration Effort**: Medium (requires UI and pipeline changes)
