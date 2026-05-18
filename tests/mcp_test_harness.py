#!/usr/bin/env python3
"""
v2rmp MCP Tools Test Harness
Tests every tool exposed by rmpca-mcp-server using synthetic data
that replicates real-world situations (urban grid, rural sparse,
mountainous terrain, delivery clusters, coastal routes, etc.)
"""

import json
import os
import subprocess
import sys
import time
import math
import tempfile
import struct
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = ROOT / "target" / "release" / "rmpca-mcp-server"
TEST_DIR = ROOT / "mcp_test_data"

# ── Synthetic Data Generators ────────────────────────────────────────────

def generate_city_grid_geojson(
    min_lon: float,
    min_lat: float,
    grid_size: int = 5,
    spacing_m: float = 200.0,
    one_way_ratio: float = 0.1,
    noise_m: float = 5.0,
) -> dict:
    """
    Generate a synthetic urban road grid (Manhattan-style) as GeoJSON.
    Each block is ~200 m, with some one-way streets and noise.
    """
    features = []
    # Approx metres per degree at this latitude
    lat_rad = math.radians(min_lat)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat_rad)

    def pt(lon: float, lat: float) -> list:
        return [lon, lat]

    def add_street(coords: list, props: dict) -> None:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })

    # Horizontal streets
    for i in range(grid_size + 1):
        lat = min_lat + (i * spacing_m) / m_per_deg_lat
        for j in range(grid_size):
            lon0 = min_lon + (j * spacing_m) / m_per_deg_lon
            lon1 = min_lon + ((j + 1) * spacing_m) / m_per_deg_lon
            one_way = (i + j) % 10 == 0 and one_way_ratio > 0
            props = {
                "highway": "residential",
                "name": f"H{i} Segment {j}",
                "oneway": "yes" if one_way else "no",
            }
            add_street([pt(lon0, lat), pt(lon1, lat)], props)

    # Vertical streets
    for j in range(grid_size + 1):
        lon = min_lon + (j * spacing_m) / m_per_deg_lon
        for i in range(grid_size):
            lat0 = min_lat + (i * spacing_m) / m_per_deg_lat
            lat1 = min_lat + ((i + 1) * spacing_m) / m_per_deg_lat
            one_way = (i + j) % 7 == 0 and one_way_ratio > 0
            props = {
                "highway": "tertiary",
                "name": f"V{j} Segment {i}",
                "oneway": "yes" if one_way else "no",
            }
            add_street([pt(lon, lat0), pt(lon, lat1)], props)

    # A diagonal arterial road (realistic curved shape)
    diag_pts = []
    steps = grid_size * 4
    for k in range(steps + 1):
        t = k / steps
        lon = min_lon + t * (grid_size * spacing_m) / m_per_deg_lon
        lat = min_lat + t * (grid_size * spacing_m) / m_per_deg_lat + math.sin(t * math.pi) * 30 / m_per_deg_lat
        diag_pts.append(pt(lon, lat))
    add_street(diag_pts, {"highway": "primary", "name": "Main Diagonal", "oneway": "no"})

    return {"type": "FeatureCollection", "features": features}


def generate_rural_sparse_geojson(
    center_lon: float,
    center_lat: float,
    num_segments: int = 12,
    max_length_m: float = 3000.0,
) -> dict:
    """Sparse rural road network with long segments and dead ends."""
    features = []
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

    def pt(lon: float, lat: float) -> list:
        return [lon, lat]

    # Spine road
    spine = []
    steps = 8
    for i in range(steps + 1):
        t = i / steps
        lon = center_lon + (t - 0.5) * 5000 / m_per_deg_lon
        lat = center_lat + math.sin(t * math.pi * 0.5) * 2000 / m_per_deg_lat
        spine.append(pt(lon, lat))
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": spine},
        "properties": {"highway": "secondary", "name": "County Road 1"},
    })

    # Random feeder roads
    import random
    rng = random.Random(42)
    for seg in range(num_segments):
        base_lon = center_lon + rng.uniform(-4000, 4000) / m_per_deg_lon
        base_lat = center_lat + rng.uniform(-3000, 3000) / m_per_deg_lat
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(500, max_length_m)
        lon2 = base_lon + math.cos(angle) * length / m_per_deg_lon
        lat2 = base_lat + math.sin(angle) * length / m_per_deg_lat
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [pt(base_lon, base_lat), pt(lon2, lat2)]},
            "properties": {"highway": "unclassified", "name": f"Feeder {seg}"},
        })

    return {"type": "FeatureCollection", "features": features}


def generate_mountain_terrain_geojson(
    center_lon: float,
    center_lat: float,
) -> dict:
    """
    Mountain road network with switchbacks and steep grades.
    """
    features = []
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

    def pt(lon: float, lat: float) -> list:
        return [lon, lat]

    # Switchback mountain road
    switchback = []
    num_turns = 6
    base_lon = center_lon
    base_lat = center_lat
    for i in range(num_turns * 2 + 1):
        t = i / (num_turns * 2)
        lon = base_lon + t * 2000 / m_per_deg_lon + (i % 2) * 100 / m_per_deg_lon
        lat = base_lat + t * 1500 / m_per_deg_lat
        switchback.append(pt(lon, lat))
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": switchback},
        "properties": {"highway": "tertiary", "name": "Alpine Pass", "surface": "gravel"},
    })

    # Valley road
    valley = []
    for i in range(10):
        t = i / 9
        lon = base_lon - 500 / m_per_deg_lon + t * 3000 / m_per_deg_lon
        lat = base_lat - 800 / m_per_deg_lat + math.sin(t * math.pi) * 100 / m_per_deg_lat
        valley.append(pt(lon, lat))
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": valley},
        "properties": {"highway": "secondary", "name": "Valley Road"},
    })

    return {"type": "FeatureCollection", "features": features}


def generate_delivery_stops(
    depot_lon: float,
    depot_lat: float,
    num_stops: int = 12,
    cluster_radius_m: float = 2000.0,
    seed: int = 123,
) -> list:
    """Generate realistic delivery stops clustered around a depot."""
    import random
    rng = random.Random(seed)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(depot_lat))

    stops = [{"lat": depot_lat, "lon": depot_lon, "label": "Depot", "demand": 0.0}]
    for i in range(num_stops):
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(100, cluster_radius_m)
        lat = depot_lat + math.sin(angle) * dist / m_per_deg_lat
        lon = depot_lon + math.cos(angle) * dist / m_per_deg_lon
        demand = rng.choice([1.0, 1.0, 1.0, 2.0, 3.0, 5.0])
        stops.append({
            "lat": lat,
            "lon": lon,
            "label": f"Stop {i + 1}",
            "demand": demand,
        })
    return stops


def generate_dem_geotiff(
    path: Path,
    width: int = 256,
    height: int = 256,
    min_lon: float = -0.1,
    min_lat: float = 45.7,
    max_lon: float = 0.1,
    max_lat: float = 45.9,
    base_elevation: float = 400.0,
    mountain_height: float = 1500.0,
) -> None:
    """
    Generate a minimal GeoTIFF DEM file using GDAL Python bindings if available,
    otherwise fall back to a raw raster + world file that GDAL can still open.
    """
    try:
        from osgeo import gdal, osr
    except ImportError:
        # Fallback: create a simple Erdas Imagine (.img) or raw + worldfile
        # We use numpy to create a binary grid and a world file
        _generate_dem_numpy(path, width, height, min_lon, min_lat, max_lon, max_lat,
                            base_elevation, mountain_height)
        return

    pixel_size_x = (max_lon - min_lon) / width
    pixel_size_y = (max_lat - min_lat) / height

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), width, height, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([min_lon, pixel_size_x, 0, max_lat, 0, -pixel_size_y])

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)

    import numpy as np
    # Mountain peak in the centre
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, y)
    Z = base_elevation + mountain_height * np.exp(-(X**2 + Y**2) * 2.0)
    band.WriteArray(Z.astype(np.float32))
    band.FlushCache()
    ds = None


def _generate_dem_numpy(
    path: Path,
    width: int,
    height: int,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    base_elevation: float,
    mountain_height: float,
) -> None:
    """Fallback DEM writer using raw float32 binary + simple world file."""
    import numpy as np
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, y)
    Z = base_elevation + mountain_height * np.exp(-(X**2 + Y**2) * 2.0)

    raw_path = path.with_suffix(".dem.bin")
    Z.astype(np.float32).tofile(raw_path)

    # Create a minimal VRT that GDAL can open
    vrt_path = path.with_suffix(".dem.vrt")
    pixel_size_x = (max_lon - min_lon) / width
    pixel_size_y = (max_lat - min_lat) / height
    vrt = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <GeoTransform>{min_lon}, {pixel_size_x}, 0, {max_lat}, 0, -{pixel_size_y}</GeoTransform>
  <SRS>EPSG:4326</SRS>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>-9999</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{raw_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    vrt_path.write_text(vrt)

    # Also write a world file for reference
    wld = path.with_suffix(".dem.wld")
    wld.write_text(f"{pixel_size_x}\n0.0\n0.0\n-{pixel_size_y}\n{min_lon + pixel_size_x / 2}\n{max_lat - pixel_size_y / 2}\n")


def write_rmp_binary(path: Path, nodes: list, edges: list) -> None:
    """
    Write a minimal .rmp binary file for testing inspect_rmp / optimize.
    Format: magic (4) + version (4) + num_nodes (4) + num_edges (4)
            then nodes (lat:f64, lon:f64) and edges (from:u32, to:u32, weight:f64, oneway:u8)
    """
    buf = bytearray()
    buf.extend(b"RMP1")
    buf.extend(struct.pack("<I", 1))  # version
    buf.extend(struct.pack("<I", len(nodes)))
    buf.extend(struct.pack("<I", len(edges)))
    for lat, lon in nodes:
        buf.extend(struct.pack("<dd", lat, lon))
    for f, t, w, ow in edges:
        buf.extend(struct.pack("<II d B", f, t, w, ow))
    path.write_bytes(bytes(buf))


# ── MCP JSON-RPC Helpers ─────────────────────────────────────────────────

class MCPClient:
    def __init__(self, server_bin: Path):
        self.proc = subprocess.Popen(
            [str(server_bin)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 1
        self._init()

    def _send(self, payload: dict) -> dict:
        line = json.dumps(payload, separators=(",", ":"))
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        response_line = self.proc.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed stdout")
        return json.loads(response_line)

    def _init(self):
        req = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test-harness", "version": "1.0"}},
        }
        resp = self._send(req)
        # Notify initialized
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict) -> dict:
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        self._next_id += 1
        return self._send(req)

    def list_tools(self) -> list:
        req = {"jsonrpc": "2.0", "id": self._next_id, "method": "tools/list"}
        self._next_id += 1
        resp = self._send(req)
        return resp.get("result", {}).get("tools", [])

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


# ── Test Suite ───────────────────────────────────────────────────────────

def run_tests(test_dir: Path, server_bin: Path) -> dict:
    results = {}
    client = MCPClient(server_bin)

    # 1. list_solvers — no external data
    print("[TEST] list_solvers")
    try:
        resp = client.call_tool("list_solvers", {})
        result = _extract_content(resp)
        results["list_solvers"] = {"status": "PASS", "solvers": result.get("solvers", [])}
    except Exception as e:
        results["list_solvers"] = {"status": "FAIL", "error": str(e)}

    # 2. haversine_distance — pure math
    print("[TEST] haversine_distance")
    try:
        resp = client.call_tool("haversine_distance", {
            "from": {"lat": 48.8566, "lon": 2.3522},
            "to": {"lat": 51.5074, "lon": -0.1278},
        })
        result = _extract_content(resp)
        dist_km = result.get("distance_km", 0)
        assert 300 < dist_km < 350, f"Expected ~344 km, got {dist_km}"
        results["haversine_distance"] = {"status": "PASS", "distance_km": dist_km}
    except Exception as e:
        results["haversine_distance"] = {"status": "FAIL", "error": str(e)}

    # Create synthetic data files
    city_geojson_path = test_dir / "city_grid.geojson"
    rural_geojson_path = test_dir / "rural_sparse.geojson"
    mountain_geojson_path = test_dir / "mountain_roads.geojson"
    rmp_path = test_dir / "city_grid.rmp"
    dem_path = test_dir / "mountain.dem.tif"

    print("[DATA] Generating city grid GeoJSON ...")
    city_geojson = generate_city_grid_geojson(min_lon=-0.05, min_lat=45.75, grid_size=6)
    city_geojson_path.write_text(json.dumps(city_geojson))

    print("[DATA] Generating rural sparse GeoJSON ...")
    rural_geojson = generate_rural_sparse_geojson(center_lon=2.3, center_lat=46.5, num_segments=15)
    rural_geojson_path.write_text(json.dumps(rural_geojson))

    print("[DATA] Generating mountain roads GeoJSON ...")
    mountain_geojson = generate_mountain_terrain_geojson(center_lon=6.8, center_lat=45.8)
    mountain_geojson_path.write_text(json.dumps(mountain_geojson))

    print("[DATA] Generating synthetic .rmp binary ...")
    # Create nodes and edges from the city grid
    nodes = []
    edges = []
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(45.75))
    spacing_m = 200.0
    grid_size = 6
    # Build node index map
    node_index = {}
    def get_node_idx(lat, lon):
        key = (round(lat, 7), round(lon, 7))
        if key not in node_index:
            node_index[key] = len(nodes)
            nodes.append((lat, lon))
        return node_index[key]

    for i in range(grid_size + 1):
        for j in range(grid_size):
            lat = 45.75 + (i * spacing_m) / m_per_deg_lat
            lon0 = -0.05 + (j * spacing_m) / m_per_deg_lon
            lon1 = -0.05 + ((j + 1) * spacing_m) / m_per_deg_lon
            f = get_node_idx(lat, lon0)
            t = get_node_idx(lat, lon1)
            dist = math.sqrt((lon1 - lon0)**2 + 0**2) * m_per_deg_lon
            edges.append((f, t, dist, 0))
    for j in range(grid_size + 1):
        for i in range(grid_size):
            lon = -0.05 + (j * spacing_m) / m_per_deg_lon
            lat0 = 45.75 + (i * spacing_m) / m_per_deg_lat
            lat1 = 45.75 + ((i + 1) * spacing_m) / m_per_deg_lat
            f = get_node_idx(lat0, lon)
            t = get_node_idx(lat1, lon)
            dist = math.sqrt((lat1 - lat0)**2 + 0**2) * m_per_deg_lat
            edges.append((f, t, dist, 0))

    write_rmp_binary(rmp_path, nodes, edges)

    print("[DATA] Generating synthetic DEM ...")
    generate_dem_geotiff(dem_path, width=128, height=128,
                         min_lon=-0.1, min_lat=45.7, max_lon=0.1, max_lat=45.9)

    # 3. compile
    print("[TEST] compile (city grid)")
    try:
        resp = client.call_tool("compile", {
            "input": str(city_geojson_path),
            "output": str(test_dir / "city_grid.rmp"),
            "clean": True,
            "prune_disconnected": False,
        })
        result = _extract_content(resp)
        nc = result.get("node_count", 0)
        ec = result.get("edge_count", 0)
        assert nc > 0 and ec > 0, f"Expected nodes/edges, got {nc}/{ec}"
        results["compile_city"] = {"status": "PASS", "node_count": nc, "edge_count": ec}
    except Exception as e:
        results["compile_city"] = {"status": "FAIL", "error": str(e)}

    # 4. compile rural
    print("[TEST] compile (rural sparse)")
    try:
        resp = client.call_tool("compile", {
            "input": str(rural_geojson_path),
            "output": str(test_dir / "rural.rmp"),
            "clean": True,
            "prune_disconnected": True,
        })
        result = _extract_content(resp)
        results["compile_rural"] = {"status": "PASS", "node_count": result.get("node_count"), "edge_count": result.get("edge_count")}
    except Exception as e:
        results["compile_rural"] = {"status": "FAIL", "error": str(e)}

    # 5. clean
    print("[TEST] clean (mountain roads)")
    try:
        resp = client.call_tool("clean", {
            "input": str(mountain_geojson_path),
            "output": str(test_dir / "mountain_cleaned.geojson"),
            "min_length_m": 5.0,
            "node_snap_m": 2.0,
            "max_components": 1,
            "simplify_tolerance_m": 1.0,
        })
        result = _extract_content(resp)
        results["clean_mountain"] = {"status": "PASS", "stats": result}
    except Exception as e:
        results["clean_mountain"] = {"status": "FAIL", "error": str(e)}

    # 6. inspect_rmp
    print("[TEST] inspect_rmp")
    try:
        resp = client.call_tool("inspect_rmp", {"input": str(rmp_path)})
        result = _extract_content(resp)
        assert result.get("node_count", 0) > 0
        results["inspect_rmp"] = {"status": "PASS", "info": result}
    except Exception as e:
        results["inspect_rmp"] = {"status": "FAIL", "error": str(e)}

    # 7. optimize (CPP on city grid)
    print("[TEST] optimize (CPP mode)")
    try:
        resp = client.call_tool("optimize", {
            "input": str(rmp_path),
            "output": str(test_dir / "route_cpp.gpx"),
            "mode": "cpp",
            "depot": {"lat": 45.75, "lon": -0.05},
            "oneway_mode": "respect",
            "left_penalty": 1.0,
            "right_penalty": 0.0,
            "uturn_penalty": 5.0,
        })
        result = _extract_content(resp)
        results["optimize_cpp"] = {"status": "PASS", "distance_km": result.get("total_distance_km")}
    except Exception as e:
        results["optimize_cpp"] = {"status": "FAIL", "error": str(e)}

    # 8. vrp_solve
    print("[TEST] vrp_solve")
    stops = generate_delivery_stops(depot_lon=-0.04, depot_lat=45.76, num_stops=8)
    try:
        resp = client.call_tool("vrp_solve", {
            "stops": stops,
            "num_vehicles": 2,
            "vehicle_capacity": 20.0,
            "solver_id": "clarke_wright",
            "avg_speed_kmh": 30.0,
            "objective": "min_distance",
        })
        result = _extract_content(resp)
        routes = result.get("routes", [])
        assert len(routes) > 0, "Expected at least one route"
        results["vrp_solve"] = {"status": "PASS", "routes": len(routes), "distance_km": result.get("total_distance_km")}
    except Exception as e:
        results["vrp_solve"] = {"status": "FAIL", "error": str(e)}

    # 9. fuel_estimate
    print("[TEST] fuel_estimate")
    samples = [
        {"distance_m": 0, "elevation_m": 400},
        {"distance_m": 2000, "elevation_m": 600},
        {"distance_m": 5000, "elevation_m": 350},
        {"distance_m": 8000, "elevation_m": 450},
    ]
    try:
        resp = client.call_tool("fuel_estimate", {
            "samples": samples,
            "base_consumption": 0.08,
        })
        result = _extract_content(resp)
        assert result.get("total_fuel_l", 0) > 0
        results["fuel_estimate"] = {"status": "PASS", "fuel_l": result.get("total_fuel_l")}
    except Exception as e:
        results["fuel_estimate"] = {"status": "FAIL", "error": str(e)}

    # 10. score_route
    print("[TEST] score_route")
    try:
        routes_indices = [[0, 1, 2, 3, 0], [0, 4, 5, 6, 7, 0]]
        resp = client.call_tool("score_route", {
            "stops": stops,
            "routes": routes_indices,
            "total_distance_km": "12.50",
            "num_vehicles": 2,
            "vehicle_capacity": 20.0,
            "objective": "min_distance",
        })
        result = _extract_content(resp)
        assert result.get("overall", 0) > 0
        results["score_route"] = {"status": "PASS", "overall": result.get("overall")}
    except Exception as e:
        results["score_route"] = {"status": "FAIL", "error": str(e)}

    # 11. route_embedding
    print("[TEST] route_embedding")
    try:
        resp = client.call_tool("route_embedding", {
            "stops": stops,
            "num_vehicles": 2,
            "vehicle_capacity": 20.0,
            "objective": "min_distance",
        })
        result = _extract_content(resp)
        vec = result.get("vector", [])
        assert len(vec) > 0
        results["route_embedding"] = {"status": "PASS", "dim": len(vec)}
    except Exception as e:
        results["route_embedding"] = {"status": "FAIL", "error": str(e)}

    # 12. elevation_query (if DEM available)
    print("[TEST] elevation_query")
    # Determine actual DEM file (may be .dem.vrt)
    actual_dem = dem_path
    if not actual_dem.exists():
        actual_dem = dem_path.with_suffix(".dem.vrt")
    if actual_dem.exists():
        try:
            resp = client.call_tool("elevation_query", {
                "dem_path": str(actual_dem),
                "points": [
                    {"lon": 0.0, "lat": 45.8},
                    {"lon": -0.05, "lat": 45.75},
                    {"lon": 0.08, "lat": 45.85},
                ],
            })
            result = _extract_content(resp)
            elevations = [r.get("elevation_m") for r in result.get("results", [])]
            results["elevation_query"] = {"status": "PASS", "elevations": elevations}
        except Exception as e:
            results["elevation_query"] = {"status": "FAIL", "error": str(e)}
    else:
        results["elevation_query"] = {"status": "SKIP", "reason": f"DEM file not found at {actual_dem}"}

    # 13. elevation_profile
    print("[TEST] elevation_profile")
    if actual_dem.exists():
        route_pts = [
            {"lon": -0.08, "lat": 45.72},
            {"lon": -0.04, "lat": 45.78},
            {"lon": 0.0, "lat": 45.82},
            {"lon": 0.05, "lat": 45.88},
        ]
        try:
            resp = client.call_tool("elevation_profile", {
                "dem_path": str(actual_dem),
                "route": route_pts,
                "sample_interval_m": 250.0,
            })
            result = _extract_content(resp)
            assert result.get("num_samples", 0) > 0
            results["elevation_profile"] = {"status": "PASS", "samples": result.get("num_samples"), "ascent": result.get("total_ascent_m")}
        except Exception as e:
            results["elevation_profile"] = {"status": "FAIL", "error": str(e)}
    else:
        results["elevation_profile"] = {"status": "SKIP", "reason": "DEM file not found"}

    # 14. elevation_stats
    print("[TEST] elevation_stats")
    if actual_dem.exists():
        try:
            resp = client.call_tool("elevation_stats", {
                "dem_path": str(actual_dem),
                "bbox": {"min_lon": -0.1, "min_lat": 45.7, "max_lon": 0.1, "max_lat": 45.9},
                "grid_step": 4,
            })
            result = _extract_content(resp)
            assert result.get("pixel_count", 0) > 0
            results["elevation_stats"] = {"status": "PASS", "stats": result}
        except Exception as e:
            results["elevation_stats"] = {"status": "FAIL", "error": str(e)}
    else:
        results["elevation_stats"] = {"status": "SKIP", "reason": "DEM file not found"}

    # 15. dem_info
    print("[TEST] dem_info")
    if actual_dem.exists():
        try:
            resp = client.call_tool("dem_info", {"dem_path": str(actual_dem)})
            result = _extract_content(resp)
            assert result.get("width", 0) > 0
            results["dem_info"] = {"status": "PASS", "info": result}
        except Exception as e:
            results["dem_info"] = {"status": "FAIL", "error": str(e)}
    else:
        results["dem_info"] = {"status": "SKIP", "reason": "DEM file not found"}

    # 16. predict_solver (ML gated — may fail if ML feature disabled)
    print("[TEST] predict_solver")
    try:
        resp = client.call_tool("predict_solver", {
            "stops": stops,
            "num_vehicles": 2,
            "vehicle_capacity": 20.0,
            "objective": "min_distance",
        })
        result = _extract_content(resp)
        results["predict_solver"] = {"status": "PASS", "recommended": result.get("recommended")}
    except Exception as e:
        err = str(e)
        if "ML feature is not enabled" in err or "not enabled" in err.lower():
            results["predict_solver"] = {"status": "SKIP", "reason": "ML feature disabled in build"}
        else:
            results["predict_solver"] = {"status": "FAIL", "error": err}

    # 17. predict_quality
    print("[TEST] predict_quality")
    try:
        resp = client.call_tool("predict_quality", {
            "stops": stops,
            "num_vehicles": 2,
            "vehicle_capacity": 20.0,
            "objective": "min_distance",
        })
        result = _extract_content(resp)
        results["predict_quality"] = {"status": "PASS", "predicted_gap_pct": result.get("predicted_gap_pct")}
    except Exception as e:
        err = str(e)
        if "ML feature is not enabled" in err or "not enabled" in err.lower():
            results["predict_quality"] = {"status": "SKIP", "reason": "ML feature disabled in build"}
        else:
            results["predict_quality"] = {"status": "FAIL", "error": err}

    # 18. tune_hyperparams
    print("[TEST] tune_hyperparams")
    try:
        resp = client.call_tool("tune_hyperparams", {
            "stops": stops,
            "num_vehicles": 2,
            "vehicle_capacity": 20.0,
            "objective": "min_distance",
        })
        result = _extract_content(resp)
        results["tune_hyperparams"] = {"status": "PASS", "params": result}
    except Exception as e:
        err = str(e)
        if "ML feature is not enabled" in err or "not enabled" in err.lower():
            results["tune_hyperparams"] = {"status": "SKIP", "reason": "ML feature disabled in build"}
        else:
            results["tune_hyperparams"] = {"status": "FAIL", "error": err}

    # 19. parse_routing_query
    print("[TEST] parse_routing_query")
    try:
        resp = client.call_tool("parse_routing_query", {
            "query": "Route 30 packages with 3 vans starting at 45.5,-73.6 by 5pm",
            "use_llm": False,
        })
        result = _extract_content(resp)
        results["parse_routing_query"] = {"status": "PASS", "variant": result.get("variant"), "method": result.get("method")}
    except Exception as e:
        err = str(e)
        if "ML feature is not enabled" in err or "not enabled" in err.lower() or "NLP requires" in err:
            results["parse_routing_query"] = {"status": "SKIP", "reason": "ML feature disabled in build"}
        else:
            results["parse_routing_query"] = {"status": "FAIL", "error": err}

    # 20. pipeline
    print("[TEST] pipeline")
    try:
        resp = client.call_tool("pipeline", {
            "bbox": {"min_lon": -0.02, "min_lat": 45.74, "max_lon": 0.02, "max_lat": 45.78},
            "output_dir": str(test_dir / "pipeline_output"),
            "source": "overture",
            "depot": {"lat": 45.76, "lon": 0.0},
            "mode": "cpp",
            "prune_disconnected": True,
        })
        result = _extract_content(resp)
        # This will likely take time or fail without internet — catch gracefully
        results["pipeline"] = {"status": "PASS", "summary": {k: v for k, v in result.items() if k != "extract"}}
    except Exception as e:
        err = str(e)
        if "overture" in err.lower() or "s3" in err.lower() or "network" in err.lower() or "timed out" in err.lower():
            results["pipeline"] = {"status": "SKIP", "reason": "Requires internet / S3 access"}
        else:
            results["pipeline"] = {"status": "FAIL", "error": err}

    # 21. get_valhalla_matrix
    print("[TEST] get_valhalla_matrix")
    try:
        resp = client.call_tool("get_valhalla_matrix", {
            "locations": [
                {"lat": 48.8566, "lon": 2.3522},
                {"lat": 51.5074, "lon": -0.1278},
                {"lat": 52.5200, "lon": 13.4050},
            ],
        })
        result = _extract_content(resp)
        mat = result.get("matrix", [])
        assert len(mat) == 3
        results["get_valhalla_matrix"] = {"status": "PASS", "size": len(mat)}
    except Exception as e:
        err = str(e)
        if "valhalla" in err.lower() or "network" in err.lower() or "timed out" in err.lower() or "request" in err.lower():
            results["get_valhalla_matrix"] = {"status": "SKIP", "reason": "Requires internet / Valhalla API"}
        else:
            results["get_valhalla_matrix"] = {"status": "FAIL", "error": err}

    client.close()
    return results


def _extract_content(resp: dict) -> dict:
    result = resp.get("result", {})
    content = result.get("content", [])
    if not content:
        if result.get("isError"):
            raise RuntimeError(f"Tool error: {result}")
        return result
    text = content[0].get("text", "{}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def _print_report(results: dict) -> None:
    print("\n" + "=" * 70)
    print("v2rmp MCP Tools Test Report")
    print("=" * 70)
    passed = 0
    failed = 0
    skipped = 0
    for name, res in results.items():
        status = res.get("status", "UNKNOWN")
        if status == "PASS":
            passed += 1
            print(f"  ✅ {name:30s} PASS")
        elif status == "FAIL":
            failed += 1
            print(f"  ❌ {name:30s} FAIL — {res.get('error', 'unknown')}")
        elif status == "SKIP":
            skipped += 1
            print(f"  ⏭️  {name:30s} SKIP — {res.get('reason', 'unknown')}")
        else:
            print(f"  ⚠️  {name:30s} {status}")
    print("-" * 70)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("=" * 70)


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if not SERVER_BIN.exists():
        print(f"ERROR: Server binary not found at {SERVER_BIN}")
        print("Run: cargo build --bin rmpca-mcp-server --release")
        sys.exit(1)

    print("Starting MCP test harness ...")
    print(f"Server binary: {SERVER_BIN}")
    print(f"Test data dir: {TEST_DIR}")
    print()

    results = run_tests(TEST_DIR, SERVER_BIN)
    _print_report(results)

    report_path = TEST_DIR / "mcp_test_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nDetailed report written to: {report_path}")

    if any(r.get("status") == "FAIL" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
