#!/usr/bin/env python3
"""
v2rmp Synthetic Data Generators
================================
Produces realistic GeoJSON road networks, DEM rasters, VRP stop sets,
and .rmp binaries that mirror real-world geospatial situations.
Importable as a library: from tests.synthetic_data import *
"""

import json
import math
import random
import struct
import zlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────

M_PER_DEG_LAT = 111320.0

def m_per_deg_lon(lat: float) -> float:
    return M_PER_DEG_LAT * math.cos(math.radians(lat))

# ── Helpers ───────────────────────────────────────────────────────────────

def _pt(lon: float, lat: float) -> List[float]:
    return [lon, lat]

def _feature(coords: List[List[float]], props: Dict[str, Any], geom_type: str = "LineString") -> Dict[str, Any]:
    return {"type": "Feature", "geometry": {"type": geom_type, "coordinates": coords}, "properties": props}

def _fc(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}

# ── Road Network Generators ─────────────────────────────────────────────

def generate_city_grid(
    min_lon: float,
    min_lat: float,
    grid_size: int = 6,
    spacing_m: float = 200.0,
    one_way_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Manhattan-style urban grid with residential/tertiary/primary roads.
    Includes a diagonal arterial and some one-way streets.
    """
    rng = random.Random(seed)
    features = []
    mpx = m_per_deg_lon(min_lat)
    mp_lat = M_PER_DEG_LAT

    for i in range(grid_size + 1):
        lat = min_lat + (i * spacing_m) / mp_lat
        for j in range(grid_size):
            lon0 = min_lon + (j * spacing_m) / mpx
            lon1 = min_lon + ((j + 1) * spacing_m) / mpx
            one_way = rng.random() < one_way_ratio
            features.append(_feature(
                [_pt(lon0, lat), _pt(lon1, lat)],
                {"highway": "residential", "name": f"H{i}S{j}", "oneway": "yes" if one_way else "no"}
            ))

    for j in range(grid_size + 1):
        lon = min_lon + (j * spacing_m) / mpx
        for i in range(grid_size):
            lat0 = min_lat + (i * spacing_m) / mp_lat
            lat1 = min_lat + ((i + 1) * spacing_m) / mp_lat
            one_way = rng.random() < one_way_ratio
            features.append(_feature(
                [_pt(lon, lat0), _pt(lon, lat1)],
                {"highway": "tertiary", "name": f"V{j}S{i}", "oneway": "yes" if one_way else "no"}
            ))

    # Diagonal arterial with slight sine curve
    diag = []
    steps = grid_size * 4
    for k in range(steps + 1):
        t = k / steps
        lon = min_lon + t * (grid_size * spacing_m) / mpx
        lat = min_lat + t * (grid_size * spacing_m) / mp_lat + math.sin(t * math.pi) * 30 / mp_lat
        diag.append(_pt(lon, lat))
    features.append(_feature(diag, {"highway": "primary", "name": "Main Diagonal", "oneway": "no"}))

    return _fc(features)


def generate_rural_sparse(
    center_lon: float,
    center_lat: float,
    num_segments: int = 15,
    max_length_m: float = 3000.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Long rural roads with sparse feeders and dead ends.
    """
    rng = random.Random(seed)
    features = []
    mpx = m_per_deg_lon(center_lat)
    mp_lat = M_PER_DEG_LAT

    # Spine road
    spine = []
    steps = 8
    for i in range(steps + 1):
        t = i / steps
        lon = center_lon + (t - 0.5) * 5000 / mpx
        lat = center_lat + math.sin(t * math.pi * 0.5) * 2000 / mp_lat
        spine.append(_pt(lon, lat))
    features.append(_feature(spine, {"highway": "secondary", "name": "County Road 1"}))

    # Random feeders
    for seg in range(num_segments):
        base_lon = center_lon + rng.uniform(-4000, 4000) / mpx
        base_lat = center_lat + rng.uniform(-3000, 3000) / mp_lat
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(500, max_length_m)
        lon2 = base_lon + math.cos(angle) * length / mpx
        lat2 = base_lat + math.sin(angle) * length / mp_lat
        features.append(_feature(
            [_pt(base_lon, base_lat), _pt(lon2, lat2)],
            {"highway": "unclassified", "name": f"Feeder {seg}"}
        ))

    return _fc(features)


def generate_mountain_terrain(
    center_lon: float,
    center_lat: float,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Mountain road network with switchbacks and a valley road.
    """
    rng = random.Random(seed)
    features = []
    mpx = m_per_deg_lon(center_lat)
    mp_lat = M_PER_DEG_LAT

    # Switchback mountain road
    switchback = []
    num_turns = 6
    base_lon = center_lon
    base_lat = center_lat
    for i in range(num_turns * 2 + 1):
        t = i / (num_turns * 2)
        lon = base_lon + t * 2000 / mpx + (i % 2) * 100 / mpx
        lat = base_lat + t * 1500 / mp_lat
        switchback.append(_pt(lon, lat))
    features.append(_feature(switchback, {"highway": "tertiary", "name": "Alpine Pass", "surface": "gravel"}))

    # Valley road
    valley = []
    for i in range(10):
        t = i / 9
        lon = base_lon - 500 / mpx + t * 3000 / mpx
        lat = base_lat - 800 / mp_lat + math.sin(t * math.pi) * 100 / mp_lat
        valley.append(_pt(lon, lat))
    features.append(_feature(valley, {"highway": "secondary", "name": "Valley Road"}))

    return _fc(features)


def generate_coastal_route(
    start_lon: float,
    start_lat: float,
    length_km: float = 15.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Winding coastal highway with cliffs and bays.
    """
    rng = random.Random(seed)
    features = []
    mpx = m_per_deg_lon(start_lat)
    mp_lat = M_PER_DEG_LAT
    total_m = length_km * 1000.0
    steps = int(length_km * 10)
    pts = []
    for i in range(steps + 1):
        t = i / steps
        lon = start_lon + (t * total_m) / mpx
        # Coastal meander
        lat = start_lat + math.sin(t * math.pi * 3) * 200 / mp_lat + rng.uniform(-20, 20) / mp_lat
        pts.append(_pt(lon, lat))
    features.append(_feature(pts, {"highway": "primary", "name": "Coastal Highway 1", "oneway": "no"}))
    return _fc(features)


def generate_industrial_zone(
    center_lon: float,
    center_lat: float,
    num_warehouses: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Industrial area with large blocks, service roads, and roundabouts.
    """
    rng = random.Random(seed)
    features = []
    mpx = m_per_deg_lon(center_lat)
    mp_lat = M_PER_DEG_LAT
    block_m = 400.0

    for w in range(num_warehouses):
        wx = center_lon + rng.uniform(-1000, 1000) / mpx
        wy = center_lat + rng.uniform(-1000, 1000) / mp_lat
        # Perimeter road
        features.append(_feature([
            _pt(wx, wy), _pt(wx + block_m / mpx, wy),
            _pt(wx + block_m / mpx, wy + block_m / mp_lat),
            _pt(wx, wy + block_m / mp_lat), _pt(wx, wy)
        ], {"highway": "service", "name": f"Warehouse {w} Perimeter"}))
        # Access road
        features.append(_feature([
            _pt(wx + block_m / (2 * mpx), wy - 100 / mp_lat),
            _pt(wx + block_m / (2 * mpx), wy)
        ], {"highway": "service", "name": f"Warehouse {w} Access"}))

    return _fc(features)


# ── VRP / Delivery Stop Generators ────────────────────────────────────────

def generate_delivery_stops(
    depot_lon: float,
    depot_lat: float,
    num_stops: int = 12,
    cluster_radius_m: float = 2000.0,
    capacity_range: Tuple[float, float] = (1.0, 5.0),
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """
    Delivery stops clustered around a depot. Demands vary to simulate
    mixed package sizes.
    """
    rng = random.Random(seed)
    mpx = m_per_deg_lon(depot_lat)
    mp_lat = M_PER_DEG_LAT
    stops = [{"lat": depot_lat, "lon": depot_lon, "label": "Depot", "demand": 0.0}]
    for i in range(num_stops):
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(100, cluster_radius_m)
        lat = depot_lat + math.sin(angle) * dist / mp_lat
        lon = depot_lon + math.cos(angle) * dist / mpx
        demand = rng.uniform(*capacity_range) if isinstance(capacity_range, tuple) else float(capacity_range)
        stops.append({"lat": lat, "lon": lon, "label": f"Stop {i + 1}", "demand": round(demand, 2)})
    return stops


def generate_time_window_stops(
    depot_lon: float,
    depot_lat: float,
    num_stops: int = 10,
    cluster_radius_m: float = 3000.0,
    seed: int = 77,
) -> List[Dict[str, Any]]:
    """
    Same as delivery stops but with synthetic time-window labels
    (no actual TW constraints in the VRP model yet).
    """
    rng = random.Random(seed)
    mpx = m_per_deg_lon(depot_lat)
    mp_lat = M_PER_DEG_LAT
    stops = [{"lat": depot_lat, "lon": depot_lon, "label": "Depot", "demand": 0.0}]
    for i in range(num_stops):
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(200, cluster_radius_m)
        lat = depot_lat + math.sin(angle) * dist / mp_lat
        lon = depot_lon + math.cos(angle) * dist / mpx
        tw_start = rng.randint(8, 14)
        tw_end = tw_start + rng.randint(2, 4)
        stops.append({
            "lat": lat, "lon": lon,
            "label": f"Stop {i + 1} ({tw_start}:00-{tw_end}:00)",
            "demand": rng.choice([1.0, 1.0, 2.0, 3.0]),
        })
    return stops


# ── DEM Raster Generator ─────────────────────────────────────────────────

def generate_dem_envi(
    output_dir: Path,
    stem: str = "synthetic_dem",
    width: int = 256,
    height: int = 256,
    min_lon: float = -0.1,
    min_lat: float = 45.7,
    max_lon: float = 0.1,
    max_lat: float = 45.9,
    base_elevation: float = 400.0,
    peak_elevation: float = 1500.0,
    noise_std: float = 10.0,
    seed: int = 42,
) -> Path:
    """
    Generate a GeoTIFF-compatible DEM via ENVI raw + header.
    Returns the path to the .vrt file that GDAL can open.
    """
    rng = np.random.default_rng(seed)
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, y)

    # Gaussian mountain
    Z = base_elevation + peak_elevation * np.exp(-(X**2 + Y**2) * 2.0)
    # Add ridges
    for _ in range(3):
        rx = rng.uniform(-0.8, 0.8)
        ry = rng.uniform(-0.8, 0.8)
        ridge = peak_elevation * 0.3 * np.exp(-((X - rx)**2 + (Y - ry)**2) * 8.0)
        Z += ridge
    # Noise
    Z += rng.normal(0, noise_std, Z.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{stem}.bin"
    hdr_path = output_dir / f"{stem}.hdr"
    vrt_path = output_dir / f"{stem}.vrt"

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    Z.astype(np.float32).tofile(bin_path)

    pixel_size_x = (max_lon - min_lon) / width
    pixel_size_y = (max_lat - min_lat) / height
    hdr_path.write_text(
        f"""ENVI
samples = {width}
lines   = {height}
bands   = 1
header offset = 0
data type = 4
interleave = bsq
byte order = 0
map info = {{Geographic Lat/Lon, 1.0, 1.0, {min_lon}, {max_lat}, {pixel_size_x}, {pixel_size_y}, WGS-84, units=Degrees}}"""
    )
    vrt_path.write_text(f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <GeoTransform>{min_lon}, {pixel_size_x}, 0, {max_lat}, 0, -{pixel_size_y}</GeoTransform>
  <SRS>EPSG:4326</SRS>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>-9999</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{bin_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""
    )
    return vrt_path


# ── .rmp Binary Generator ──────────────────────────────────────────────────

def build_rmp_from_geojson(
    geojson: Dict[str, Any],
    output_path: Path,
) -> Tuple[int, int]:
    """
    Build a valid .rmp binary file from a GeoJSON FeatureCollection.
    Mimics compile.rs logic (Node dedup + edge list + CRC32).
    Returns (node_count, edge_count).
    """
    nodes: List[Tuple[float, float]] = []
    node_map: Dict[Tuple[int, int], int] = {}
    edges: List[Tuple[int, int, float, int]] = []

    def get_node(lat: float, lon: float) -> int:
        key = (round(lat, 7), round(lon, 7))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append((lat, lon))
        return node_map[key]

    for feat in geojson.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        props = feat.get("properties", {})
        oneway = 1 if str(props.get("oneway", "")).lower() in ("yes", "1", "true") else 0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            f_id = get_node(lat1, lon1)
            t_id = get_node(lat2, lon2)
            dist = math.sqrt(
                ((lon2 - lon1) * m_per_deg_lon((lat1 + lat2) / 2)) ** 2 +
                ((lat2 - lat1) * M_PER_DEG_LAT) ** 2
            )
            edges.append((f_id, t_id, dist, oneway))

    buf = bytearray()
    buf.extend(b"RMP1")
    buf.extend(struct.pack("<I", len(nodes)))
    buf.extend(struct.pack("<I", len(edges)))
    for lat, lon in nodes:
        buf.extend(struct.pack("<dd", lat, lon))
    for f, t, w, ow in edges:
        buf.extend(struct.pack("<II d B", f, t, w, ow))
    crc = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    buf.extend(struct.pack("<I", crc))
    output_path.write_bytes(bytes(buf))
    return len(nodes), len(edges)


# ── Scenario Factories ────────────────────────────────────────────────────

def create_urban_scenario(output_dir: Path, seed: int = 42) -> Dict[str, Path]:
    """Create a full urban test scenario: grid roads, .rmp, stops, DEM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson = generate_city_grid(min_lon=-0.05, min_lat=45.75, grid_size=5, seed=seed)
    gj_path = output_dir / "urban_roads.geojson"
    gj_path.write_text(json.dumps(geojson))
    rmp_path = output_dir / "urban_network.rmp"
    build_rmp_from_geojson(geojson, rmp_path)
    stops = generate_delivery_stops(depot_lon=-0.04, depot_lat=45.76, num_stops=8, seed=seed)
    stops_path = output_dir / "urban_stops.json"
    stops_path.write_text(json.dumps(stops, indent=2))
    dem_path = generate_dem_envi(output_dir, stem="urban_dem", width=128, height=128,
                                 min_lon=-0.1, min_lat=45.7, max_lon=0.1, max_lat=45.9, seed=seed)
    return {
        "geojson": gj_path,
        "rmp": rmp_path,
        "stops": stops_path,
        "dem": dem_path,
    }


def create_rural_scenario(output_dir: Path, seed: int = 42) -> Dict[str, Path]:
    """Create a rural scenario with sparse roads, long distances, small DEM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson = generate_rural_sparse(center_lon=2.3, center_lat=46.5, num_segments=12, seed=seed)
    gj_path = output_dir / "rural_roads.geojson"
    gj_path.write_text(json.dumps(geojson))
    rmp_path = output_dir / "rural_network.rmp"
    build_rmp_from_geojson(geojson, rmp_path)
    stops = generate_delivery_stops(depot_lon=2.31, depot_lat=46.51, num_stops=6, cluster_radius_m=5000.0, seed=seed)
    stops_path = output_dir / "rural_stops.json"
    stops_path.write_text(json.dumps(stops, indent=2))
    dem_path = generate_dem_envi(output_dir, stem="rural_dem", width=64, height=64,
                                 min_lon=2.2, min_lat=46.4, max_lon=2.4, max_lat=46.6,
                                 base_elevation=300.0, peak_elevation=800.0, seed=seed)
    return {
        "geojson": gj_path,
        "rmp": rmp_path,
        "stops": stops_path,
        "dem": dem_path,
    }


def create_mountain_scenario(output_dir: Path, seed: int = 42) -> Dict[str, Path]:
    """Mountain scenario: switchbacks + steep DEM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson = generate_mountain_terrain(center_lon=6.8, center_lat=45.8, seed=seed)
    gj_path = output_dir / "mountain_roads.geojson"
    gj_path.write_text(json.dumps(geojson))
    rmp_path = output_dir / "mountain_network.rmp"
    build_rmp_from_geojson(geojson, rmp_path)
    stops = generate_delivery_stops(depot_lon=6.81, depot_lat=45.81, num_stops=5, cluster_radius_m=1500.0, seed=seed)
    stops_path = output_dir / "mountain_stops.json"
    stops_path.write_text(json.dumps(stops, indent=2))
    dem_path = generate_dem_envi(output_dir, stem="mountain_dem", width=128, height=128,
                                 min_lon=6.7, min_lat=45.7, max_lon=6.9, max_lat=45.9,
                                 base_elevation=600.0, peak_elevation=2500.0, seed=seed)
    return {
        "geojson": gj_path,
        "rmp": rmp_path,
        "stops": stops_path,
        "dem": dem_path,
    }


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic v2rmp test data")
    parser.add_argument("--output-dir", type=Path, default=Path("mcp_test_data"))
    parser.add_argument("--scenario", choices=["urban", "rural", "mountain", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = {}
    if args.scenario in ("urban", "all"):
        results["urban"] = create_urban_scenario(args.output_dir / "urban", seed=args.seed)
        print(f"[urban] → {results['urban']}")
    if args.scenario in ("rural", "all"):
        results["rural"] = create_rural_scenario(args.output_dir / "rural", seed=args.seed)
        print(f"[rural] → {results['rural']}")
    if args.scenario in ("mountain", "all"):
        results["mountain"] = create_mountain_scenario(args.output_dir / "mountain", seed=args.seed)
        print(f"[mountain] → {results['mountain']}")
    print("Done.")
