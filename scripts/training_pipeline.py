# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "safetensors",
#     "numpy",
#     "requests",
# ]
# ///

import os
import sys
import json
import subprocess
import requests
import numpy as np

CVRPLIB_URL = "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/X/X-n101-k25.vrp"
BKS_URL = "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/X/X-n101-k25.sol"

def download_file(url, dest):
    if os.path.exists(dest):
        return
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)

def parse_vrp(path):
    with open(path, "r") as f:
        lines = f.readlines()
    
    coords = {}
    demands = {}
    capacity = 0
    section = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1])
        elif line.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        elif line.startswith("DEMAND_SECTION"):
            section = "demand"
            continue
        elif line.startswith("DEPOT_SECTION"):
            section = "depot"
            continue
        elif line.startswith("EOF"):
            break
        
        if section == "coords":
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                # CVRPLIB coordinates are usually Euclidean, but we'll treat them as Lat/Lon 
                # scaled to be in a reasonable range (e.g. around Montreal 45, -73)
                coords[idx] = (45.0 + float(parts[2]) / 1000.0, -73.0 + float(parts[1]) / 1000.0)
        elif section == "demand":
            parts = line.split()
            if len(parts) >= 2:
                demands[int(parts[0])] = float(parts[1])
                
    return coords, demands, capacity

def write_csv(coords, demands, path):
    with open(path, "w") as f:
        f.write("lat,lon,label,demand,type\n")
        for idx in sorted(coords.keys()):
            lat, lon = coords[idx]
            demand = demands[idx]
            role = "depot" if idx == 1 else "stop"
            f.write(f"{lat},{lon},node_{idx},{demand},{role}\n")

def run_solver(rmp_path, csv_path, solver_id):
    cmd = [
        "./target/release/rmpca",
        "--json",
        "optimize",
        "--mode", "vrp",
        "--input", rmp_path,
        "--coordinates", csv_path,
        "--solver", solver_id,
        "--vehicles", "25" # Hardcoded for this instance for now
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Solver {solver_id} failed: {e}")
        return None

def main():
    os.makedirs("data/cvrplib", exist_ok=True)
    instance_path = "data/cvrplib/X-n101-k25.vrp"
    csv_path = "data/cvrplib/X-n101-k25.csv"
    
    download_file(CVRPLIB_URL, instance_path)
    coords, demands, capacity = parse_vrp(instance_path)
    write_csv(coords, demands, csv_path)
    
    # We need an .rmp file. Let's create a dummy one if it doesn't exist.
    rmp_path = "data/cvrplib/dummy.rmp"
    if not os.path.exists(rmp_path):
        # Create a tiny GeoJSON and compile it
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[-73.0, 45.0], [-73.1, 45.1]]},
                "properties": {"highway": "residential"}
            }]
        }
        with open("data/cvrplib/dummy.geojson", "w") as f:
            json.dump(geojson, f)
        subprocess.run(["./target/release/rmpca", "compile", "--input", "data/cvrplib/dummy.geojson", "--output", rmp_path], check=True)

    solvers = ["clarke_wright", "sweep", "two_opt", "or_opt", "default"]
    results = {}
    
    for s in solvers:
        print(f"Running solver: {s}")
        res = run_solver(rmp_path, csv_path, s)
        if res:
            results[s] = {
                "distance": res["total_distance_km"],
                "time_ms": res["elapsed_ms"]
            }
            
    print("\nResults for X-n101-k25:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
