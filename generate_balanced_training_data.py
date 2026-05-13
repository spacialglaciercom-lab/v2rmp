# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24",
# ]
# ///

"""
Generate class-balanced VRP training data.

Instead of random instance generation where default always wins,
this creates instances specifically designed to challenge different solvers,
ensuring each solver gets meaningful training signal.

Solver bias rules used during generation:
- clarke_wright: tight capacity, medium size (n=20-200), MinDistance
- sweep: many vehicles (v>=5), wide geographic spread, radial layout
- or_opt: large instances (n>100), MinVehicles, MinTime
- two_opt: medium instances with crossing edges (grid-like)
- neural_guided: medium-large with clustering
- default: variety of small/medium instances

We still run all solvers to record actual distances, but we bias the
instance generation to favor under-represented solvers.

Output: data/training_data_balanced.jsonl (same schema as original)
Usage:  python generate_balanced_training_data.py 6000
"""

import os, sys, json, time, subprocess, math, random
from collections import Counter
import numpy as np

random.seed(42)
np.random.seed(42)

OUT_PATH = "data/training_data_balanced.jsonl"
LOG_PATH = "data/generate_balanced.log"

NUM_SOLVERS = 6
SOLVER_IDS = ["default", "clarke_wright", "sweep", "or_opt", "two_opt", "neural_guided"]
TARGET_PER_SOLVER = None  # Will be computed from total count


def generate_instance_for_solver(target_solver: str, seed: int) -> dict:
    """
    Generate a deterministic VRP instance biased so that `target_solver`
    has a higher chance of winning. We vary:
    - n_stops
    - n_vehicles
    - objective
    - layout pattern
    - capacity/demand ratio
    - depot position
    """
    rng = random.Random(seed)
    
    # Target solver bias
    if target_solver == "clarke_wright":
        n = rng.randint(20, 150)
        v = rng.randint(2, 8)
        objective = "MinDistance"
        pattern = "clustered_tight"  # tight capacity clusters
        capacity_ratio = rng.uniform(1.3, 2.0)  # tight: demand close to capacity
        area_scale = rng.uniform(0.5, 2.0)
    elif target_solver == "sweep":
        n = rng.randint(30, 200)
        v = rng.randint(5, 15)
        objectives = ["MinDistance", "MinTime", "BalanceLoad"]
        objective = rng.choice(objectives)
        pattern = "radial_wide"  # wide radial → sweep sectors work well
        capacity_ratio = rng.uniform(2.0, 5.0)
        area_scale = rng.uniform(2.0, 5.0)
    elif target_solver == "or_opt":
        n = rng.randint(100, 300)
        v = rng.randint(1, 6)
        objectives = ["MinDistance", "MinTime", "BalanceLoad", "MinVehicles"]
        objective = rng.choice(objectives)
        pattern = rng.choice(["uniform", "grid", "clustered_wide"])
        capacity_ratio = rng.uniform(1.5, 4.0)
        area_scale = rng.uniform(1.0, 4.0)
    elif target_solver == "two_opt":
        n = rng.randint(20, 120)
        v = rng.randint(1, 4)
        objective = "MinDistance"
        pattern = "grid_cross"  # grid with noise produces crossing edges
        capacity_ratio = rng.uniform(2.0, 6.0)
        area_scale = rng.uniform(0.8, 2.0)
    elif target_solver == "neural_guided":
        n = rng.randint(50, 250)
        v = rng.randint(2, 10)
        objectives = ["MinDistance", "MinTime", "BalanceLoad"]
        objective = rng.choice(objectives)
        pattern = rng.choice(["clustered", "grid_clustered"])
        capacity_ratio = rng.uniform(1.5, 4.0)
        area_scale = rng.uniform(0.8, 3.0)
    else:  # default
        n = rng.randint(10, 200)
        v = rng.randint(1, 12)
        objectives = ["MinDistance", "MinTime", "BalanceLoad", "MinVehicles"]
        objective = rng.choice(objectives)
        pattern = rng.choice(["uniform", "clustered", "grid", "radial"])
        capacity_ratio = rng.uniform(1.5, 6.0)
        area_scale = rng.uniform(0.5, 3.0)

    # Depot: vary globally, not all near Montreal
    base_lat = rng.uniform(35.0, 55.0)
    base_lon = rng.uniform(-125.0, -65.0)

    stops = [{"lat": base_lat, "lon": base_lon, "label": "depot", "demand": 0.0}]
    capacity = 100.0
    total_demand_target = capacity * v * capacity_ratio
    per_stop_demand = total_demand_target / max(n, 1)

    for i in range(1, n + 1):
        if pattern == "uniform":
            lat = base_lat + (rng.random() - 0.5) * area_scale * 2.0
            lon = base_lon + (rng.random() - 0.5) * area_scale * 2.0
        elif pattern == "clustered" or pattern == "clustered_tight":
            num_clusters = rng.randint(2, 5)
            cx = base_lat + (rng.random() - 0.5) * area_scale * 1.2
            cy = base_lon + (rng.random() - 0.5) * area_scale * 1.2
            lat = cx + (rng.random() - 0.5) * 0.25
            lon = cy + (rng.random() - 0.5) * 0.25
        elif pattern == "grid" or pattern == "grid_cross" or pattern == "grid_clustered":
            grid_n = int(math.sqrt(n)) + 1
            gx = (i % grid_n) / grid_n
            gy = (i // grid_n) / grid_n
            lat = base_lat + (gx - 0.5) * area_scale * 1.5 + (rng.random() - 0.5) * 0.04
            lon = base_lon + (gy - 0.5) * area_scale * 1.5 + (rng.random() - 0.5) * 0.04
        elif pattern == "radial" or pattern == "radial_wide":
            angle = rng.random() * 2.0 * math.pi
            radius = rng.random() * area_scale * 0.8
            lat = base_lat + radius * math.sin(angle)
            lon = base_lon + radius * math.cos(angle)
        else:
            lat = base_lat + (rng.random() - 0.5) * area_scale * 2.0
            lon = base_lon + (rng.random() - 0.5) * area_scale * 2.0

        demand = max(rng.gauss(per_stop_demand, per_stop_demand * 0.2), 1.0)
        stops.append({"lat": lat, "lon": lon, "label": f"stop_{i}", "demand": demand})

    return {
        "stops": stops,
        "num_vehicles": v,
        "vehicle_capacity": capacity,
        "objective": objective,
    }


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_features(instance: dict) -> list:
    """Extract 28-dim feature vector (same as Rust code, in Python)."""
    stops = instance["stops"]
    n = len(stops)
    if n <= 1:
        return [0.0] * 28
    
    depot = stops[0]
    others = stops[1:]
    num_stops = len(others)
    
    # Geometric
    dists = []
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            dists.append(haversine_km(others[i]["lat"], others[i]["lon"], others[j]["lat"], others[j]["lon"]))
    avg_pairwise = sum(dists) / max(len(dists), 1)
    
    lats = [s["lat"] for s in others]
    lons = [s["lon"] for s in others]
    lat_mean = sum(lats) / len(lats)
    lon_mean = sum(lons) / len(lons)
    lat_spread = math.sqrt(sum((x - lat_mean) ** 2 for x in lats) / max(len(lats) - 1, 1))
    lon_spread = math.sqrt(sum((x - lon_mean) ** 2 for x in lons) / max(len(lons) - 1, 1))
    
    lat_km = lat_spread * 111.0
    lon_km = lon_spread * 111.0 * math.cos(math.radians(depot["lat"]))
    area_km2 = max(lat_km * lon_km, 0.001)
    density = num_stops / area_km2
    
    depot_centroid_dist = haversine_km(depot["lat"], depot["lon"], lat_mean, lon_mean)
    
    # kNN graph features (k=10)
    n_others = len(others)
    k = min(10, n_others - 1) if n_others > 1 else 1
    if n_others >= 2:
        dist_matrix = [[haversine_km(others[i]["lat"], others[i]["lon"], others[j]["lat"], others[j]["lon"]) for j in range(n_others)] for i in range(n_others)]
        adj = [[] for _ in range(n_others)]
        degrees = [0] * n_others
        for i in range(n_others):
            neighbors = sorted([(j, dist_matrix[i][j]) for j in range(n_others) if j != i], key=lambda x: x[1])
            for j, _ in neighbors[:k]:
                adj[i].append(j)
                adj[j].append(i)
                degrees[i] += 1
                degrees[j] += 1
        avg_degree = sum(degrees) / max(n_others, 1)
        max_degree = max(degrees) if degrees else 0
        
        # Clustering
        clustering_sum = 0.0
        for i in range(n_others):
            neighbors = adj[i]
            di = len(neighbors)
            if di < 2:
                continue
            edges_between = sum(1 for a_idx in range(di) for b_idx in range(a_idx + 1, di) if neighbors[b_idx] in adj[neighbors[a_idx]])
            clustering_sum += edges_between / (di * (di - 1) / 2)
        clustering = clustering_sum / max(n_others, 1)
        
        # MST (Prim)
        visited = [False] * n_others
        min_edge = [float('inf')] * n_others
        min_edge[0] = 0.0
        mst_weight = 0.0
        for _ in range(n_others):
            u = min((i for i in range(n_others) if not visited[i]), key=lambda i: min_edge[i])
            visited[u] = True
            mst_weight += min_edge[u]
            for v in range(n_others):
                if not visited[v] and dist_matrix[u][v] < min_edge[v]:
                    min_edge[v] = dist_matrix[u][v]
        
        # Diameter & avg shortest path (Floyd-Warshall for small n)
        if n_others <= 200:
            sp = [row[:] for row in dist_matrix]
            for kk in range(n_others):
                for ii in range(n_others):
                    for jj in range(n_others):
                        if sp[ii][kk] + sp[kk][jj] < sp[ii][jj]:
                            sp[ii][jj] = sp[ii][kk] + sp[kk][jj]
            max_d = 0.0
            sum_d = 0.0
            count = 0
            for ii in range(n_others):
                for jj in range(ii + 1, n_others):
                    if sp[ii][jj] < float('inf') / 2:
                        max_d = max(max_d, sp[ii][jj])
                        sum_d += sp[ii][jj]
                        count += 1
            avg_sp = sum_d / max(count, 1)
        else:
            max_d = max(max(row) for row in dist_matrix)
            avg_sp = sum(sum(row) for row in dist_matrix) / max(n_others * (n_others - 1), 1)
        
        spectral_gap = max_degree - avg_degree
        assortativity = 0.0  # simplified
    else:
        avg_degree = max_degree = clustering = mst_weight = max_d = avg_sp = spectral_gap = assortativity = 0.0
    
    # Demand
    demands = [s["demand"] for s in others]
    total_demand = sum(demands)
    demand_std = math.sqrt(sum((d - total_demand / max(len(demands), 1)) ** 2 for d in demands) / max(len(demands) - 1, 1)) if len(demands) > 1 else 0.0
    capacity_ratio = total_demand / (instance["vehicle_capacity"] * max(instance["num_vehicles"], 1))
    tight_capacity = 1.0 if capacity_ratio > 0.8 else 0.0
    
    # Distance matrix stats
    dist_mean = sum(dists) / max(len(dists), 1)
    dist_std = math.sqrt(sum((d - dist_mean) ** 2 for d in dists) / max(len(dists) - 1, 1)) if len(dists) > 1 else 0.0
    dist_skew = 0.0
    if len(dists) > 2 and dist_std > 0:
        mean = dist_mean
        m3 = sum((d - mean) ** 3 for d in dists) / len(dists)
        dist_skew = m3 / (dist_std ** 3)
    depot_dists = [haversine_km(depot["lat"], depot["lon"], s["lat"], s["lon"]) for s in others]
    depot_dist_mean = sum(depot_dists) / max(len(depot_dists), 1)
    
    # Objective one-hot
    obj = instance["objective"]
    obj_dist = 1.0 if obj == "MinDistance" else 0.0
    obj_time = 1.0 if obj == "MinTime" else 0.0
    obj_balance = 1.0 if obj == "BalanceLoad" else 0.0
    obj_vehicles = 1.0 if obj == "MinVehicles" else 0.0
    
    # Normalize
    features = [
        min(num_stops / 500.0, 1.0),
        min(instance["num_vehicles"] / 20.0, 1.0),
        min(avg_pairwise / 100.0, 1.0),
        min(lat_spread / 5.0, 1.0),
        min(lon_spread / 5.0, 1.0),
        min(density / 50.0, 1.0),
        min(area_km2 / 1000.0, 1.0),
        min(depot_centroid_dist / 100.0, 1.0),
        avg_degree,
        max_degree,
        clustering,
        min(max_d / 100.0, 1.0),
        min(mst_weight / 1000.0, 1.0),
        min(avg_sp / 100.0, 1.0),
        min(spectral_gap / 50.0, 1.0),
        max(min(assortativity, 1.0), -1.0),
        min(total_demand / 1000.0, 1.0),
        min(demand_std / 100.0, 1.0),
        tight_capacity,
        min(capacity_ratio, 1.0),
        min(dist_mean / 100.0, 1.0),
        min(dist_std / 100.0, 1.0),
        dist_skew,
        min(depot_dist_mean / 100.0, 1.0),
        obj_dist,
        obj_time,
        obj_balance,
        obj_vehicles,
    ]
    return features


# ---------------------------------------------------------------------------
# Call Rust solvers via subprocess
# ---------------------------------------------------------------------------

RMP_PATH = "data/dummy_for_training.rmp"

def ensure_dummy_rmp():
    """Create a minimal GeoJSON and compile to .rmp for solver input."""
    if os.path.exists(RMP_PATH):
        return
    os.makedirs("data", exist_ok=True)
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[-73.0, 45.0], [-73.1, 45.1]]},
            "properties": {"highway": "residential"}
        }]
    }
    with open("data/dummy.geojson", "w") as f:
        json.dump(geojson, f)
    # Use python to create a trivial RMP header or just touch it if compilation fails
    try:
        subprocess.run(["./target/release/rmpca", "compile", "--input", "data/dummy.geojson", "--output", RMP_PATH],
                       check=True, capture_output=True, timeout=10)
    except Exception:
        # Fallback: create a minimal binary RMP (not valid, but solver only needs coordinates)
        import struct
        with open(RMP_PATH, "wb") as f:
            f.write(b"rmpca\x01\x00\x00\x00")
        print("WARNING: Using placeholder RMP file")


def solve_with_rust(instance: dict, solver_id: str) -> float:
    """Run a single solver on an instance and return distance in km."""
    ensure_dummy_rmp()
    csv_path = f"/tmp/vrp_{os.getpid()}_{solver_id}.csv"
    
    # Write CSV
    with open(csv_path, "w") as f:
        f.write("lat,lon,label,demand,type\n")
        for s in instance["stops"]:
            role = "depot" if s["label"] == "depot" else "stop"
            f.write(f"{s['lat']},{s['lon']},{s['label']},{s['demand']},{role}\n")
    
    cmd = [
        "./target/release/rmpca", "--json",
        "optimize", "--mode", "vrp",
        "--input", RMP_PATH,
        "--coordinates", csv_path,
        "--solver", solver_id,
        "--vehicles", str(instance["num_vehicles"]),
        "--objective", instance["objective"],
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("total_distance_km", "inf"))
    except Exception:
        pass
    finally:
        try:
            os.remove(csv_path)
        except:
            pass
    return float('inf')


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_balanced(n_total: int, target_frac: dict = None):
    """
    Generate n_total instances with controlled solver proportions.
    target_frac: e.g. {"default": 0.35, "clarke_wright": 0.15, "sweep": 0.10, 
                        "or_opt": 0.20, "two_opt": 0.10, "neural_guided": 0.10}
    """
    if target_frac is None:
        target_frac = {
            "default": 0.30,
            "clarke_wright": 0.14,
            "sweep": 0.12,
            "or_opt": 0.20,
            "two_opt": 0.12,
            "neural_guided": 0.12,
        }
    
    # Compute per-solver targets
    per_solver = {s: int(n_total * f) for s, f in target_frac.items()}
    # Adjust rounding
    diff = n_total - sum(per_solver.values())
    per_solver["default"] += diff
    
    print(f"Generating {n_total} instances with target distribution:")
    for s, c in per_solver.items():
        print(f"  {s:18s}: {c:5d}")
    
    records = []
    solver_counts = Counter()
    seed = 1_000_000
    
    # Round-robin generation for each target solver
    for target_solver, target_count in per_solver.items():
        generated = 0
        attempts = 0
        while generated < target_count and attempts < target_count * 10:
            seed += 1
            attempts += 1
            instance = generate_instance_for_solver(target_solver, seed)
            
            # Run all solvers
            solver_dists = []
            best_dist = float('inf')
            best_solver = "default"
            
            for sid in SOLVER_IDS:
                dist = solve_with_rust(instance, sid)
                solver_dists.append({"solver": sid, "distance": dist})
                if dist < best_dist:
                    best_dist = dist
                    best_solver = sid
            
            if best_dist == float('inf'):
                print(f"  Solver failure, skipping (attempt {attempts})")
                continue
            
            features = extract_features(instance)
            n_stops = len(instance["stops"]) - 1
            depot = instance["stops"][0]
            lower_bound = sum(haversine_km(depot["lat"], depot["lon"], s["lat"], s["lon"]) for s in instance["stops"][1:]) * 2.0
            gap = ((best_dist - lower_bound) / max(lower_bound, 1e-6)) * 100.0 if lower_bound > 0 else 0.0
            
            record = {
                "features": features,
                "best_solver": best_solver,
                "best_distance_km": best_dist,
                "lower_bound_km": lower_bound,
                "gap_pct": gap,
                "n_stops": n_stops,
                "n_vehicles": instance["num_vehicles"],
                "objective": instance["objective"],
                "solver_dists": solver_dists,
                "target_bias": target_solver,
            }
            records.append(record)
            solver_counts[best_solver] += 1
            generated += 1
            
            if generated % 10 == 0:
                print(f"  {target_solver}: {generated}/{target_count}  (best={best_solver}, dist={best_dist:.1f})")
    
    return records, solver_counts


def save(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print("=" * 60)
    print(f"Generating {n_total} BALANCED training instances")
    print("This calls Rust solver binaries per instance — may take ~10-30 min")
    print("=" * 60)
    
    t0 = time.time()
    records, solver_counts = generate_balanced(n_total)
    
    # Save
    os.makedirs("data", exist_ok=True)
    save(records, OUT_PATH)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"  Total generated: {len(records)}")
    print(f"  Saved to: {OUT_PATH}")
    print(f"  Actual solver distribution:")
    for sid in SOLVER_IDS:
        print(f"    {sid:18s}: {solver_counts[sid]:5d} ({solver_counts[sid]/max(len(records),1)*100:.1f}%)")
    print(f"  Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
