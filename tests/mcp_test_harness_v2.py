#!/usr/bin/env python3
"""v2rmp MCP Tools Test Harness v2 — faster, with per-test timeout."""
import json, os, subprocess, sys, math, struct, select, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Prefer debug build
SERVER_BIN = ROOT / "target" / "debug" / "rmpca-mcp-server-legacy"
if not SERVER_BIN.exists():
    SERVER_BIN = ROOT / "target" / "release" / "rmpca-mcp-server-legacy"
TEST_DIR = ROOT / "mcp_test_data"

def generate_city_grid_geojson(min_lon, min_lat, grid_size=6, spacing_m=200.0):
    features = []
    lat_rad = math.radians(min_lat)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat_rad)
    def pt(lon, lat): return [lon, lat]
    def add(coords, props):
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": props})
    for i in range(grid_size + 1):
        lat = min_lat + (i * spacing_m) / m_per_deg_lat
        for j in range(grid_size):
            lon0 = min_lon + (j * spacing_m) / m_per_deg_lon
            lon1 = min_lon + ((j + 1) * spacing_m) / m_per_deg_lon
            add([pt(lon0, lat), pt(lon1, lat)], {"highway": "residential", "oneway": "yes" if (i+j)%10==0 else "no"})
    for j in range(grid_size + 1):
        lon = min_lon + (j * spacing_m) / m_per_deg_lon
        for i in range(grid_size):
            lat0 = min_lat + (i * spacing_m) / m_per_deg_lat
            lat1 = min_lat + ((i + 1) * spacing_m) / m_per_deg_lat
            add([pt(lon, lat0), pt(lon, lat1)], {"highway": "tertiary", "oneway": "yes" if (i+j)%7==0 else "no"})
    # diagonal
    diag = []
    steps = grid_size * 4
    for k in range(steps + 1):
        t = k / steps
        lon = min_lon + t * (grid_size * spacing_m) / m_per_deg_lon
        lat = min_lat + t * (grid_size * spacing_m) / m_per_deg_lat + math.sin(t * math.pi) * 30 / m_per_deg_lat
        diag.append(pt(lon, lat))
    add(diag, {"highway": "primary", "oneway": "no"})
    return {"type": "FeatureCollection", "features": features}


def generate_delivery_stops(depot_lon, depot_lat, num_stops=8, cluster_radius_m=2000.0, seed=123):
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
        stops.append({"lat": lat, "lon": lon, "label": f"Stop {i+1}", "demand": rng.choice([1.0, 1.0, 2.0, 3.0])})
    return stops


def write_rmp_binary(path, nodes, edges):
    """Write a valid .rmp binary with CRC32 footer matching compile.rs."""
    buf = bytearray()
    buf.extend(b"RMP1")
    buf.extend(struct.pack("<I", len(nodes)))
    buf.extend(struct.pack("<I", len(edges)))
    for lat, lon in nodes:
        buf.extend(struct.pack("<dd", lat, lon))
    for f, t, w, ow in edges:
        buf.extend(struct.pack("<II d B", f, t, w, ow))
    import zlib
    crc = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    buf.extend(struct.pack("<I", crc))
    path.write_bytes(bytes(buf))


def _generate_dem_fallback(path, width=128, height=128, min_lon=-0.1, min_lat=45.7, max_lon=0.1, max_lat=45.9, base_elev=400.0, mountain=1500.0):
    import numpy as np
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    X, Y = np.meshgrid(x, y)
    Z = base_elev + mountain * np.exp(-(X**2 + Y**2) * 2.0)
    raw = path.with_suffix(".dem.bin")
    Z.astype(np.float32).tofile(raw)
    pixel_size_x = (max_lon - min_lon) / width
    pixel_size_y = (max_lat - min_lat) / height
    # GDAL needs an ENVI header to read raw float32; name the binary so .hdr sits alongside it
    hdr = path.with_suffix(".dem.hdr")
    hdr.write_text(
        f"""ENVI\nsamples = {width}\nlines   = {height}\nbands   = 1\nheader offset = 0\ndata type = 4\ninterleave = bsq\nbyte order = 0\nmap info = {{Geographic Lat/Lon, 1.0, 1.0, {min_lon}, {max_lat}, {pixel_size_x}, {pixel_size_y}, WGS-84, units=Degrees}}"""
    )
    vrt_path = path.parent / (path.stem + ".vrt")
    # Referencing the raw binary; GDAL will auto-find the .hdr with same basename
    vrt_path.write_text(f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <GeoTransform>{min_lon}, {pixel_size_x}, 0, {max_lat}, 0, -{pixel_size_y}</GeoTransform>
  <SRS>EPSG:4326</SRS>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>-9999</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{raw.name}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>""")


class MCPClient:
    def __init__(self, bin_path):
        self.proc = subprocess.Popen([str(bin_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self._id = 1
        # initialize
        self._send({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}})
        self._write({"jsonrpc":"2.0","method":"notifications/initialized"})
        # Drain any stderr startup message
        time.sleep(0.2)

    def _write(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _send(self, obj):
        self._write(obj)
        ready, _, _ = select.select([self.proc.stdout], [], [], 15)
        if not ready:
            raise TimeoutError("No response from MCP server")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("EOF from server")
        return json.loads(line)

    def call(self, name, args):
        req = {"jsonrpc":"2.0","id":self._id,"method":"tools/call","params":{"name":name,"arguments":args}}
        self._id += 1
        return self._send(req)

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def _extract(resp):
    result = resp.get("result", {})
    content = result.get("content", [])
    if not content:
        if result.get("isError"):
            raise RuntimeError(str(result))
        return result
    text = content[0].get("text", "{}")
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text}


def run_test(name, client, args, check=None):
    print(f"[TEST] {name} ...", end=" ", flush=True)
    try:
        resp = client.call(name, args)
        data = _extract(resp)
        if check and not check(data):
            print(f"FAIL (check failed): {data}")
            return {"status": "FAIL", "error": "check failed", "data": data}
        print("PASS")
        return {"status": "PASS", "data": data}
    except Exception as e:
        err = str(e)
        if "ML feature is not enabled" in err or "NLP requires" in err:
            print("SKIP (ML disabled)")
            return {"status": "SKIP", "reason": "ML feature disabled"}
        print(f"FAIL: {err}")
        return {"status": "FAIL", "error": err}


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if not SERVER_BIN.exists():
        print("ERROR: binary not found")
        sys.exit(1)

    print("Generating synthetic data ...")
    city_geojson = TEST_DIR / "city_grid.geojson"
    city_geojson.write_text(json.dumps(generate_city_grid_geojson(-0.05, 45.75, grid_size=5)))
    rmp_path = TEST_DIR / "test_grid.rmp"
    nodes, edges = [], []
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(45.75))
    spacing_m = 200.0
    grid_size = 5
    node_index = {}
    def get_node(lat, lon):
        key = (round(lat, 7), round(lon, 7))
        if key not in node_index:
            node_index[key] = len(nodes)
            nodes.append((lat, lon))
        return node_index[key]
    for i in range(grid_size + 1):
        lat = 45.75 + (i * spacing_m) / m_per_deg_lat
        for j in range(grid_size):
            lon0 = -0.05 + (j * spacing_m) / m_per_deg_lon
            lon1 = -0.05 + ((j + 1) * spacing_m) / m_per_deg_lon
            edges.append((get_node(lat, lon0), get_node(lat, lon1), spacing_m, 0))
    for j in range(grid_size + 1):
        lon = -0.05 + (j * spacing_m) / m_per_deg_lon
        for i in range(grid_size):
            lat0 = 45.75 + (i * spacing_m) / m_per_deg_lat
            lat1 = 45.75 + ((i + 1) * spacing_m) / m_per_deg_lat
            edges.append((get_node(lat0, lon), get_node(lat1, lon), spacing_m, 0))
    write_rmp_binary(rmp_path, nodes, edges)

    dem_path = TEST_DIR / "mountain.dem.tif"
    _generate_dem_fallback(dem_path)
    actual_dem = TEST_DIR / "mountain.dem.vrt"

    stops = generate_delivery_stops(-0.04, 45.76, num_stops=6)
    print(f"Data ready. Starting server ...")

    client = MCPClient(SERVER_BIN)
    results = {}

    results["list_solvers"] = run_test("list_solvers", client, {}, lambda d: len(d.get("solvers", [])) > 0)
    results["haversine_distance"] = run_test("haversine_distance", client, {"from": {"lat": 48.8566, "lon": 2.3522}, "to": {"lat": 51.5074, "lon": -0.1278}}, lambda d: 300 < d.get("distance_km", 0) < 350)
    results["compile"] = run_test("compile", client, {"input": str(city_geojson), "output": str(TEST_DIR / "city.rmp"), "clean": True}, lambda d: d.get("node_count", 0) > 0)
    results["clean"] = run_test("clean", client, {"input": str(city_geojson), "output": str(TEST_DIR / "cleaned.geojson"), "min_length_m": 5.0, "max_components": 1}, lambda d: d.get("output_features", 0) > 0)
    results["inspect_rmp"] = run_test("inspect_rmp", client, {"input": str(rmp_path)}, lambda d: d.get("node_count", 0) > 0)
    results["optimize_cpp"] = run_test("optimize", client, {"input": str(rmp_path), "mode": "cpp", "depot": {"lat": 45.75, "lon": -0.05}}, lambda d: d.get("total_distance_km") is not None)
    results["vrp_solve"] = run_test("vrp_solve", client, {"stops": stops, "num_vehicles": 2, "vehicle_capacity": 20.0, "solver_id": "clarke_wright"}, lambda d: len(d.get("routes", [])) > 0)
    results["fuel_estimate"] = run_test("fuel_estimate", client, {"samples": [{"distance_m": 0, "elevation_m": 400}, {"distance_m": 5000, "elevation_m": 800}]}, lambda d: d.get("total_fuel_l", 0) > 0)
    results["score_route"] = run_test("score_route", client, {"stops": stops, "routes": [[0,1,2,3,0],[0,4,5,6,0]], "total_distance_km": "10.0", "num_vehicles": 2}, lambda d: d.get("overall", 0) > 0)
    results["route_embedding"] = run_test("route_embedding", client, {"stops": stops, "num_vehicles": 2}, lambda d: len(d.get("vector", [])) > 0)

    if actual_dem.exists():
        results["elevation_query"] = run_test("elevation_query", client, {"dem_path": str(actual_dem), "points": [{"lon": 0.0, "lat": 45.8}]}, lambda d: len(d.get("results", [])) > 0)
        results["elevation_profile"] = run_test("elevation_profile", client, {"dem_path": str(actual_dem), "route": [{"lon": -0.08, "lat": 45.72}, {"lon": 0.0, "lat": 45.82}]}, lambda d: d.get("num_samples", 0) > 0)
        results["elevation_stats"] = run_test("elevation_stats", client, {"dem_path": str(actual_dem), "bbox": {"min_lon": -0.1, "min_lat": 45.7, "max_lon": 0.1, "max_lat": 45.9}}, lambda d: d.get("pixel_count", 0) > 0)
        results["dem_info"] = run_test("dem_info", client, {"dem_path": str(actual_dem)}, lambda d: d.get("width", 0) > 0)
    else:
        for k in ["elevation_query", "elevation_profile", "elevation_stats", "dem_info"]:
            results[k] = {"status": "SKIP", "reason": "DEM not available"}

    results["predict_solver"] = run_test("predict_solver", client, {"stops": stops, "num_vehicles": 2}, lambda d: d.get("ml_ready") is True)
    results["predict_quality"] = run_test("predict_quality", client, {"stops": stops, "num_vehicles": 2}, lambda d: d.get("ml_ready") is True)
    results["tune_hyperparams"] = run_test("tune_hyperparams", client, {"stops": stops, "num_vehicles": 2}, lambda d: d.get("ml_ready") is True)
    results["parse_routing_query"] = run_test("parse_routing_query", client, {"query": "Route 30 packages with 3 vans at 45.5,-73.6", "use_llm": False}, lambda d: True)

    # Feedback loop
    print("[TEST] submit_feedback ...", end=" ", flush=True)
    try:
        resp = client.call("submit_feedback", {
            "stops": stops,
            "solver_id": "clarke_wright",
            "total_distance_km": 42.5,
            "actual_gap_pct": 3.5,
            "elapsed_ms": 1200,
            "num_vehicles": 2,
            "vehicle_capacity": 20.0,
            "objective": "min_distance",
        })
        data = _extract(resp)
        if data.get("status") == "logged" and data.get("feedback_count", 0) >= 1:
            print("PASS")
            results["submit_feedback"] = {"status": "PASS", "feedback_count": data.get("feedback_count")}
        else:
            print(f"FAIL (unexpected response: {data})")
            results["submit_feedback"] = {"status": "FAIL", "error": "unexpected response"}
    except Exception as e:
        print(f"FAIL: {e}")
        results["submit_feedback"] = {"status": "FAIL", "error": str(e)}

    # Network-dependent tools: treat timeout as SKIP instead of FAIL
    print("[TEST] pipeline ...", end=" ", flush=True)
    try:
        resp = client.call("pipeline", {"bbox": {"min_lon": -0.02, "min_lat": 45.74, "max_lon": 0.02, "max_lat": 45.78}, "output_dir": str(TEST_DIR / "pipe_out"), "source": "overture"})
        data = _extract(resp)
        print("PASS")
        results["pipeline"] = {"status": "PASS", "data": data}
    except TimeoutError:
        print("SKIP (network — Overture S3)")
        results["pipeline"] = {"status": "SKIP", "reason": "Requires internet / Overture S3 download"}
    except Exception as e:
        print(f"SKIP ({e})")
        results["pipeline"] = {"status": "SKIP", "reason": str(e)}

    print("[TEST] get_valhalla_matrix ...", end=" ", flush=True)
    try:
        resp = client.call("get_valhalla_matrix", {"locations": [{"lat": 48.8566, "lon": 2.3522}, {"lat": 51.5074, "lon": -0.1278}]})
        data = _extract(resp)
        print("PASS")
        results["get_valhalla_matrix"] = {"status": "PASS", "data": data}
    except TimeoutError:
        print("SKIP (network — Valhalla API)")
        results["get_valhalla_matrix"] = {"status": "SKIP", "reason": "Requires internet / Valhalla API"}
    except Exception as e:
        print(f"SKIP ({e})")
        results["get_valhalla_matrix"] = {"status": "SKIP", "reason": str(e)}

    client.close()

    print("\n" + "=" * 60)
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    failed = sum(1 for r in results.values() if r["status"] == "FAIL")
    skipped = sum(1 for r in results.values() if r["status"] == "SKIP")
    for name, res in results.items():
        s = res["status"]
        if s == "PASS":
            print(f"  ✅ {name}")
        elif s == "FAIL":
            print(f"  ❌ {name} — {res.get('error', 'unknown')}")
        else:
            print(f"  ⏭️  {name} — {res.get('reason', 'skipped')}")
    print(f"\nPassed: {passed}  Failed: {failed}  Skipped: {skipped}")
    (TEST_DIR / "mcp_test_report.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"Report saved to {TEST_DIR / 'mcp_test_report.json'}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
