"""
GeoJSON/OSM cleaning pipeline before optimizer.

  POST /api/geojson/clean — validate, repair geometry, dedupe nodes/edges,
  remove self-loops/short edges/isolates, keep largest component(s); return cleaned GeoJSON + stats.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Literal

import networkx as nx
import numpy as np
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely import make_valid
from shapely.strtree import STRtree

from .geojson_ops import (
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    _extract_coords,
    _haversine_km,
    _haversine_km_vectorized,
)

router = APIRouter()

# Approx meters per degree at equator (for bbox query)
_M_PER_DEG = 111_320.0

# Use GeoPandas for geometry repair when feature count >= this (batch/vectorized path)
BATCH_GEOJSON_THRESHOLD = 10_000

# Max request body size for POST /api/geojson/clean (DoS protection)
CLEAN_MAX_BODY_BYTES = 50 * 1024 * 1024  # 50 MB


async def limit_geojson_clean_body_size(request: Request) -> None:
    """Reject request if Content-Length exceeds CLEAN_MAX_BODY_BYTES."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > CLEAN_MAX_BODY_BYTES:
                raise HTTPException(
                    413,
                    detail=f"Request body too large. Max {CLEAN_MAX_BODY_BYTES // (1024*1024)} MB for /api/geojson/clean.",
                )
        except ValueError:
            pass


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in meters."""
    return _haversine_km(lon1, lat1, lon2, lat2) * 1000.0


def _round_key(lon: float, lat: float, decimals: int = 6) -> tuple[float, float]:
    return (round(lon, decimals), round(lat, decimals))


def _node_id(lon: float, lat: float, decimals: int = 6) -> str:
    return f"{round(lon, decimals)},{round(lat, decimals)}"


# ---------------------------------------------------------------------------
# Options and stats
# ---------------------------------------------------------------------------


class CleanOptions(BaseModel):
    """Options for POST /api/geojson/clean."""

    makevalid: bool = True
    drop_invalid: bool = True
    remove_selfloops: bool = True
    min_length_m: float = 0.1
    node_snap_m: float = 1.0  # Merge nodes within this distance (m); use <1 for sub-meter precision
    node_precision_decimals: int = 6
    merge_node_positions: bool = True
    dedupe_edges: bool = True
    remove_isolates: bool = True
    max_components: int = 1
    required_attrs: list[str] | None = None
    merge_parallel_edges: bool = False
    merge_parallel_edge_properties: bool = False
    property_merge_strategy: Literal["first", "merge"] = "first"
    simplify_tolerance_m: float = 0.0
    include_polygons: bool = False
    include_points: bool = False


class CleanStats(BaseModel):
    """Counts returned after cleaning."""

    input_features: int = 0
    output_features: int = 0
    invalid_dropped: int = 0
    selfloops_removed: int = 0
    short_edges_removed: int = 0
    nodes_merged: int = 0
    duplicate_edges_removed: int = 0
    incomplete_edges_removed: int = 0
    parallel_edges_merged: int = 0
    isolates_removed: int = 0
    components_removed: int = 0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _geom_to_shapely(geom: dict[str, Any]) -> LineString | Point | Polygon | None:
    """Convert GeoJSON geometry dict to Shapely. Returns LineString, Point, Polygon, or None."""
    if not geom or not geom.get("coordinates"):
        return None
    try:
        return shape(geom)
    except Exception:
        return None


def _polygon_to_linestrings(shp: Polygon) -> list[LineString]:
    """Extract exterior and interior rings as LineStrings (no repeated closing point)."""
    lines: list[LineString] = []
    if shp.is_empty or shp.exterior is None:
        return lines
    coords = list(shp.exterior.coords)
    if len(coords) >= 2:
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    for interior in shp.interiors:
        coords = list(interior.coords)
        if len(coords) >= 2:
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) >= 2:
                lines.append(LineString(coords))
    return lines


def _simplify_coords_m(coords: list[list[float]], tolerance_m: float) -> list[list[float]]:
    """Apply Douglas-Peucker simplification; tolerance_m is approximate (equatorial)."""
    if tolerance_m <= 0 or len(coords) < 3:
        return coords
    line = LineString(coords)
    tolerance_deg = tolerance_m / _M_PER_DEG
    simplified = line.simplify(tolerance_deg, preserve_topology=True)
    if simplified.is_empty:
        return coords
    return [list(c) for c in simplified.coords]


def _make_valid_geom(geom: LineString | Point | None) -> LineString | Point | None:
    """Repair invalid geometry; return None if empty or still invalid."""
    if geom is None or geom.is_empty:
        return None
    try:
        fixed = make_valid(geom)
        if fixed is None or fixed.is_empty:
            return None
        if fixed.geom_type == "GeometryCollection":
            # Take first line-like part
            for g in fixed.geoms:
                if g.geom_type in ("LineString", "Point") and not g.is_empty:
                    return g
            return None
        return fixed
    except Exception:
        return None


def _shapely_to_geojson_geom(geom: LineString | Point) -> dict[str, Any]:
    """Convert Shapely geometry to GeoJSON dict (coordinates only for LineString/Point)."""
    if geom.geom_type == "Point":
        return {"type": "Point", "coordinates": list(geom.coords[0])}
    if geom.geom_type == "LineString":
        return {"type": "LineString", "coordinates": [list(c) for c in geom.coords]}
    return {"type": "LineString", "coordinates": []}


def _repair_geojson_geopandas(
    fc: list[dict[str, Any]],
    makevalid: bool,
    stats_dict: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Vectorized geometry repair using GeoPandas. Used when feature count >= BATCH_GEOJSON_THRESHOLD.
    Returns list of repaired feature dicts (LineString only); updates stats_dict for invalid_dropped.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame.from_features(fc)
    if gdf.empty:
        return []

    # Restrict to line types
    geom_types = gdf.geometry.type
    line_mask = geom_types.isin(("LineString", "MultiLineString"))
    gdf = gdf.loc[line_mask].copy()
    initial_line_count = len(gdf)
    if gdf.empty:
        stats_dict["invalid_dropped"] = stats_dict.get("invalid_dropped", 0) + len(fc)
        return []

    if makevalid:
        gdf["geometry"] = gdf["geometry"].apply(_make_valid_geom)
    gdf = gdf[gdf["geometry"].notnull() & ~gdf["geometry"].is_empty]
    stats_dict["invalid_dropped"] = (
        stats_dict.get("invalid_dropped", 0) + initial_line_count - len(gdf)
    )

    # Explode MultiLineString to one row per LineString
    gdf = gdf.explode(ignore_index=True)
    if gdf.empty:
        return []

    repaired: list[dict[str, Any]] = []
    for idx, row in gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        if len(geom.coords) < 2:
            continue
        props = {
            k: v for k, v in row.items()
            if k != "geometry" and not (isinstance(v, float) and np.isnan(v))
        }
        repaired.append({
            "type": "Feature",
            "geometry": _shapely_to_geojson_geom(geom),
            "properties": props,
        })
    return repaired


# ---------------------------------------------------------------------------
# GeoJSON -> Graph (with geometry + properties on edges)
# ---------------------------------------------------------------------------


def _geojson_features_to_graph(
    features: list[dict[str, Any]],
    decimals: int = 6,
) -> tuple[nx.MultiGraph, list[dict[str, Any]]]:
    """
    Build MultiGraph from LineString features. Node id = "lon,lat" (rounded).
    Each edge has: length_m, coords, properties.
    Returns (G, edge_records) where edge_records[i] = {coords, properties} for export.
    """
    G = nx.MultiGraph()
    edge_records: list[dict[str, Any]] = []

    for feat in features:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type", "")
        coords_list: list[list[list[float]]] = []
        if gtype == "LineString":
            coords_list = [geom.get("coordinates", [])]
        elif gtype == "MultiLineString":
            raw = geom.get("coordinates", [])
            # Skip empty or degenerate sub-lines in MultiLineString
            coords_list = [
                line for line in (raw or [])
                if isinstance(line, (list, tuple)) and len(line) >= 2
            ]
        else:
            continue

        props = feat.get("properties") or {}

        for line_coords in coords_list:
            if not line_coords or len(line_coords) < 2:
                continue

            coords_arr = np.array(line_coords, dtype=np.float64)
            lon1, lat1 = coords_arr[:-1, 0], coords_arr[:-1, 1]
            lon2, lat2 = coords_arr[1:, 0], coords_arr[1:, 1]
            seg_km = _haversine_km_vectorized(lon1, lat1, lon2, lat2)
            length_m = float(np.sum(seg_km) * 1000.0)

            start, end = line_coords[0], line_coords[-1]
            u = _node_id(start[0], start[1], decimals)
            v = _node_id(end[0], end[1], decimals)

            if u not in G.nodes:
                G.add_node(u, lon=round(start[0], decimals), lat=round(start[1], decimals))
            if v not in G.nodes:
                G.add_node(v, lon=round(end[0], decimals), lat=round(end[1], decimals))

            key = len(edge_records)
            G.add_edge(u, v, key=key, length_m=length_m, coords=line_coords, properties=props)
            edge_records.append({"coords": line_coords, "properties": props})

    return G, edge_records


# ---------------------------------------------------------------------------
# Graph cleaning steps
# ---------------------------------------------------------------------------


def _remove_selfloops(G: nx.MultiGraph, stats: dict[str, int]) -> None:
    selfloops = list(nx.selfloop_edges(G, keys=True))
    if selfloops:
        for u, v, k in selfloops:
            G.remove_edge(u, v, k)
        stats["selfloops_removed"] += len(selfloops)


def _remove_short_edges(
    G: nx.MultiGraph,
    min_length_m: float,
    stats: dict[str, int],
) -> None:
    to_remove = []
    for u, v, key in list(G.edges(keys=True)):
        data = G.edges[u, v, key]
        if data.get("length_m", 0) < min_length_m:
            to_remove.append((u, v, key))
    for u, v, key in to_remove:
        G.remove_edge(u, v, key)
        stats["short_edges_removed"] += 1


def _merge_duplicate_nodes(
    G: nx.MultiGraph,
    node_snap_m: float,
    stats: dict[str, int],
    decimals: int = 6,
    merge_node_positions: bool = True,
) -> None:
    """Merge nodes within node_snap_m using spatial proximity (STRtree + haversine)."""
    if G.number_of_nodes() == 0:
        return

    nodes = list(G.nodes())
    # Build points (lon, lat) for STRtree; Shapely uses (x,y) = (lon, lat)
    points = []
    for nid in nodes:
        data = G.nodes[nid]
        lon = data.get("lon", 0.0)
        lat = data.get("lat", 0.0)
        points.append(Point(lon, lat))

    tree = STRtree(points)

    # Latitude-aware bbox: at high latitudes 1 deg longitude is shorter in meters
    def _query_box(lon: float, lat: float) -> Any:
        delta_lat_deg = node_snap_m / _M_PER_DEG
        cos_lat = max(0.01, math.cos(math.radians(lat)))
        delta_lon_deg = node_snap_m / (_M_PER_DEG * cos_lat)
        return box(
            lon - delta_lon_deg, lat - delta_lat_deg,
            lon + delta_lon_deg, lat + delta_lat_deg,
        )

    # Union-find: each node -> canonical representative (smallest node id by string order)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # canonical = min by string order
            parent[ra] = parent[rb] = min(ra, rb)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    for i, nid in enumerate(nodes):
        pt = points[i]
        lon, lat = pt.x, pt.y
        env = _query_box(lon, lat)
        candidates = tree.query(env)
        if candidates is None:
            continue
        # Vectorized distance check for candidates
        j_indices = [int(j) for j in candidates if 0 <= int(j) < len(nodes)]
        j_indices = [j for j in j_indices if nodes[j] != nid]
        if not j_indices:
            continue
        other_lons = np.array([points[j].x for j in j_indices])
        other_lats = np.array([points[j].y for j in j_indices])
        dist_km = _haversine_km_vectorized(
            np.full(len(j_indices), lon),
            np.full(len(j_indices), lat),
            other_lons,
            other_lats,
        )
        dist_m = dist_km * 1000.0
        for j, d in zip(j_indices, dist_m):
            if d <= node_snap_m:
                union(nid, nodes[j])

    # Set canonical node position: average of group or first (by node id order)
    groups: dict[str, list[tuple[float, float]]] = {}
    for nid in nodes:
        root = find(nid)
        lon = G.nodes[nid]["lon"]
        lat = G.nodes[nid]["lat"]
        groups.setdefault(root, []).append((lon, lat))

    for root, pos_list in groups.items():
        if merge_node_positions and len(pos_list) > 1:
            avg_lon = sum(p[0] for p in pos_list) / len(pos_list)
            avg_lat = sum(p[1] for p in pos_list) / len(pos_list)
            G.nodes[root]["lon"] = round(avg_lon, decimals)
            G.nodes[root]["lat"] = round(avg_lat, decimals)
        else:
            G.nodes[root]["lon"] = round(pos_list[0][0], decimals)
            G.nodes[root]["lat"] = round(pos_list[0][1], decimals)

    # Relabel: old_id -> canonical_id
    mapping = {nid: find(nid) for nid in nodes}
    merged_count = sum(1 for n in nodes if mapping[n] != n)
    if merged_count > 0:
        stats["nodes_merged"] += merged_count
        nx.relabel_nodes(G, mapping, copy=False)
        # Snap edge endpoints to canonical node positions and recompute length_m
        for u, v, key in list(G.edges(keys=True)):
            data = G.edges[u, v, key]
            coords = data.get("coords")
            if not coords or len(coords) < 2:
                continue
            coords[0] = [G.nodes[u]["lon"], G.nodes[u]["lat"]]
            coords[-1] = [G.nodes[v]["lon"], G.nodes[v]["lat"]]
            coords_arr = np.array(coords, dtype=np.float64)
            lon1, lat1 = coords_arr[:-1, 0], coords_arr[:-1, 1]
            lon2, lat2 = coords_arr[1:, 0], coords_arr[1:, 1]
            seg_km = _haversine_km_vectorized(lon1, lat1, lon2, lat2)
            data["length_m"] = float(np.sum(seg_km) * 1000.0)


def _dedupe_edges(G: nx.MultiGraph, stats: dict[str, int]) -> None:
    """Remove duplicate edges: same (u, v) and same geometry hash; keep first."""
    seen: dict[tuple[str, str], set[str]] = {}
    to_remove = []
    for u, v, key in list(G.edges(keys=True)):
        data = G.edges[u, v, key]
        coords = data.get("coords", [])
        h = hashlib.sha256(str(tuple(tuple(c) for c in coords)).encode()).hexdigest()[:16]
        edge_key = (min(u, v), max(u, v))
        if edge_key not in seen:
            seen[edge_key] = set()
        if h in seen[edge_key]:
            to_remove.append((u, v, key))
            stats["duplicate_edges_removed"] += 1
        else:
            seen[edge_key].add(h)

    for u, v, key in to_remove:
        G.remove_edge(u, v, key)


def _remove_edges_missing_attrs(
    G: nx.MultiGraph,
    required_attrs: list[str] | None,
    stats: dict[str, int],
) -> None:
    if not required_attrs:
        return
    to_remove = []
    for u, v, key in list(G.edges(keys=True)):
        data = G.edges[u, v, key]
        props = data.get("properties") or {}
        for attr in required_attrs:
            if attr not in props or props[attr] is None:
                to_remove.append((u, v, key))
                stats["incomplete_edges_removed"] += 1
                break
    for u, v, key in to_remove:
        G.remove_edge(u, v, key)


def _merge_property_values(values: list[Any], key: str) -> Any:
    """Merge a list of property values: average numbers, join strings, union sets/lists."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None
    if len(non_none) == 1:
        return non_none[0]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none):
        return sum(v for v in non_none) / len(non_none)
    if all(isinstance(v, str) for v in non_none):
        seen: set[str] = set()
        out: list[str] = []
        for v in non_none:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return "; ".join(out)
    if all(isinstance(v, (list, tuple)) for v in non_none):
        result: list[Any] = []
        seen_set: set[tuple[Any, ...]] = set()
        for v in non_none:
            t = tuple(v) if isinstance(v, (list, tuple)) else (v,)
            if t not in seen_set:
                seen_set.add(t)
                result.extend(v if isinstance(v, list) else list(v))
        return result
    # Default: keep first non-None
    return non_none[0]


def _merge_parallel_edges(
    G: nx.MultiGraph,
    stats: dict[str, int],
    merge_properties: bool = False,
) -> None:
    """Collapse multi-edges between same (u,v) into one edge (first geometry, summed length)."""
    merged = 0
    for u, v in list(G.edges(keys=False)):
        keys = list(G[u][v].keys())
        if len(keys) <= 1:
            continue
        k0 = keys[0]
        data0 = G.edges[u, v, k0]
        length_m = data0.get("length_m", 0.0)
        all_props = [data0.get("properties", {})]
        for k in keys[1:]:
            data = G.edges[u, v, k]
            length_m += data.get("length_m", 0.0)
            all_props.append(data.get("properties", {}))
            G.remove_edge(u, v, k)
            merged += 1
        G.edges[u, v, k0]["length_m"] = length_m
        if merge_properties and all_props:
            all_keys = set()
            for p in all_props:
                all_keys.update(p.keys())
            merged_props: dict[str, Any] = {}
            for key in all_keys:
                vals = [p.get(key) for p in all_props if key in p and p.get(key) is not None]
                if vals:
                    merged_props[key] = _merge_property_values(vals, key)
            if merged_props:
                G.edges[u, v, k0]["properties"] = merged_props
    stats["parallel_edges_merged"] += merged


def _remove_isolates(G: nx.MultiGraph, stats: dict[str, int]) -> None:
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        stats["isolates_removed"] += len(isolates)


def _keep_largest_components(
    G: nx.MultiGraph,
    max_components: int,
    stats: dict[str, int],
) -> None:
    if max_components <= 0 or G.number_of_nodes() == 0:
        return
    comps = list(nx.connected_components(G))
    if len(comps) <= max_components:
        return
    comps_sorted = sorted(comps, key=len, reverse=True)
    keep = set()
    for c in comps_sorted[:max_components]:
        keep.update(c)
    remove = set(G.nodes) - keep
    stats["components_removed"] += len(comps) - max_components
    G.remove_nodes_from(remove)


# ---------------------------------------------------------------------------
# Graph -> GeoJSON
# ---------------------------------------------------------------------------


def _graph_to_geojson(
    G: nx.MultiGraph,
    simplify_tolerance_m: float = 0.0,
    point_features: list[dict[str, Any]] | None = None,
) -> GeoJSONFeatureCollection:
    """Rebuild FeatureCollection from graph edges (coords + properties on each edge)."""
    features: list[GeoJSONFeature] = []
    for u, v, key in G.edges(keys=True):
        data = G.edges[u, v, key]
        coords = data.get("coords")
        if not coords or len(coords) < 2:
            continue
        if simplify_tolerance_m > 0:
            coords = _simplify_coords_m(coords, simplify_tolerance_m)
        if len(coords) < 2:
            continue
        props = data.get("properties") or {}
        geom = {"type": "LineString", "coordinates": coords}
        features.append(
            GeoJSONFeature(type="Feature", geometry=geom, properties=props)
        )
    if point_features:
        for pf in point_features:
            features.append(
                GeoJSONFeature(
                    type=pf.get("type", "Feature"),
                    geometry=pf.get("geometry", {}),
                    properties=pf.get("properties") or {},
                )
            )
    return GeoJSONFeatureCollection(type="FeatureCollection", features=features)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def clean_geojson(
    geojson: dict[str, Any],
    options: CleanOptions,
) -> tuple[GeoJSONFeatureCollection, CleanStats]:
    """
    Run full cleaning pipeline: geometry repair -> graph -> topology clean -> export.
    """
    stats = CleanStats(
        input_features=0,
        output_features=0,
        invalid_dropped=0,
        selfloops_removed=0,
        short_edges_removed=0,
        nodes_merged=0,
        duplicate_edges_removed=0,
        incomplete_edges_removed=0,
        parallel_edges_merged=0,
        isolates_removed=0,
        components_removed=0,
    )
    stats_dict = stats.model_dump()

    fc = geojson.get("features") or []
    stats_dict["input_features"] = len(fc)

    # Stage 2: Geometry repair (make_valid, drop invalid)
    # Use GeoPandas batch path for large datasets when available (not when polygons/points included)
    repaired: list[dict[str, Any]] = []
    use_batch = (
        len(fc) >= BATCH_GEOJSON_THRESHOLD
        and not options.include_polygons
        and not options.include_points
    )
    if use_batch:
        try:
            repaired = _repair_geojson_geopandas(
                fc, options.makevalid, stats_dict
            )
        except ImportError:
            use_batch = False
    point_features: list[dict[str, Any]] = []
    if use_batch and options.include_points:
        for f in fc:
            geom = (f or {}).get("geometry")
            if geom and geom.get("type") == "Point":
                try:
                    pt = shape(geom)
                    if pt is not None and not pt.is_empty:
                        point_features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": list(pt.coords[0])},
                            "properties": (f or {}).get("properties") or {},
                        })
                except Exception:
                    pass
    if not use_batch:
        # Loop path: small datasets or when GeoPandas unavailable
        accepted_line = ("LineString", "MultiLineString")
        if options.include_polygons:
            accepted_line = accepted_line + ("Polygon", "MultiPolygon")
        for f in fc:
            geom = (f or {}).get("geometry")
            if not geom:
                continue
            gtype = geom.get("type", "")
            if gtype == "Point" and options.include_points:
                shp = _geom_to_shapely(geom)
                if shp is not None and not shp.is_empty:
                    point_features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": list(shp.coords[0])},
                        "properties": (f or {}).get("properties") or {},
                    })
                continue
            if gtype not in accepted_line:
                continue
            shp = _geom_to_shapely(geom)
            if shp is None:
                stats_dict["invalid_dropped"] += 1
                continue
            if options.makevalid:
                shp = _make_valid_geom(shp)
            if shp is None or shp.is_empty:
                stats_dict["invalid_dropped"] += 1
                continue
            # Normalize to LineString(s); skip empty or degenerate sub-lines in MultiLineString
            if shp.geom_type == "LineString":
                lines = [shp]
            elif shp.geom_type == "MultiLineString":
                lines = [g for g in shp.geoms if not g.is_empty and len(g.coords) >= 2]
            elif shp.geom_type == "Polygon":
                lines = _polygon_to_linestrings(shp)
            elif shp.geom_type == "MultiPolygon":
                lines = []
                for poly in shp.geoms:
                    lines.extend(_polygon_to_linestrings(poly))
            else:
                continue
            props = (f or {}).get("properties") or {}
            for line in lines:
                if line.is_empty or len(line.coords) < 2:
                    continue
                coords = [list(c) for c in line.coords]
                repaired.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": props,
                })

    if not repaired:
        return GeoJSONFeatureCollection(type="FeatureCollection", features=[]), CleanStats(**stats_dict)

    # Stage 3: Build graph
    decimals = options.node_precision_decimals
    G, _ = _geojson_features_to_graph(repaired, decimals=decimals)

    # Stages 4–11
    if options.remove_selfloops:
        _remove_selfloops(G, stats_dict)
    if options.min_length_m > 0:
        _remove_short_edges(G, options.min_length_m, stats_dict)
    if options.node_snap_m > 0:
        _merge_duplicate_nodes(
            G,
            options.node_snap_m,
            stats_dict,
            decimals=decimals,
            merge_node_positions=options.merge_node_positions,
        )
    if options.dedupe_edges:
        _dedupe_edges(G, stats_dict)
    if options.required_attrs:
        _remove_edges_missing_attrs(G, options.required_attrs, stats_dict)
    if options.merge_parallel_edges:
        merge_props = (
            options.property_merge_strategy == "merge"
            or options.merge_parallel_edge_properties
        )
        _merge_parallel_edges(G, stats_dict, merge_properties=merge_props)
    if options.remove_isolates:
        _remove_isolates(G, stats_dict)
    if options.max_components > 0:
        _keep_largest_components(G, options.max_components, stats_dict)

    out_fc = _graph_to_geojson(
        G,
        simplify_tolerance_m=options.simplify_tolerance_m,
        point_features=point_features if options.include_points else None,
    )
    stats_dict["output_features"] = len(out_fc.features)
    return out_fc, CleanStats(**stats_dict)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class CleanRequest(BaseModel):
    geojson: GeoJSONFeatureCollection
    options: CleanOptions = Field(default_factory=CleanOptions)


class CleanResponse(BaseModel):
    geojson: GeoJSONFeatureCollection
    stats: CleanStats
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings (e.g. high invalid-drop ratio)")


@router.post(
    "/api/geojson/clean",
    response_model=CleanResponse,
    dependencies=[Depends(limit_geojson_clean_body_size)],
    summary="Clean GeoJSON",
    description="Repair geometry, remove self-loops/short edges/duplicates, keep largest component. Request body limit: 50 MB.",
)
def post_geojson_clean(body: CleanRequest) -> CleanResponse:
    """Clean GeoJSON: repair geometry, remove self-loops/short edges/duplicates, keep largest component."""
    geojson_dict = body.geojson.model_dump()
    options = body.options
    cleaned_fc, clean_stats = clean_geojson(geojson_dict, options)
    warnings: list[str] = []
    n_in = clean_stats.input_features
    if n_in > 0 and clean_stats.invalid_dropped / n_in > 0.10:
        msg = (
            "Over 10% of features were dropped as invalid; consider checking CRS and geometry validity."
        )
        warnings.append(msg)
        logging.warning(
            "geojson/clean: high invalid ratio (invalid_dropped=%s, input_features=%s, ratio=%.2f). %s",
            clean_stats.invalid_dropped,
            n_in,
            clean_stats.invalid_dropped / n_in,
            msg,
        )
    return CleanResponse(geojson=cleaned_fc, stats=clean_stats, warnings=warnings)
