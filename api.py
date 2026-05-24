from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import subprocess
import json
import os

app = FastAPI(
    title="v2rmp API",
    description="REST API wrapper for the v2rmp Route Optimization Engine",
    version="0.4.8"
)

# --- Models ---

class BBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_str(self) -> str:
        return f"{self.min_lon},{self.min_lat},{self.max_lon},{self.max_lat}"

class ExtractRequest(BaseModel):
    bbox: str = Field(..., description="MIN_LON,MIN_LAT,MAX_LON,MAX_LAT")
    source: str = "overture"
    output: str = "network.geojson"

class CompileRequest(BaseModel):
    input: str = "network.geojson"
    output: str = "network.rmp"

class CleanRequest(BaseModel):
    input: str = "network.geojson"
    output: str = "network_clean.geojson"
    min_length_m: Optional[float] = 0.1
    snap_dist_m: Optional[float] = 1.0

class OptimizeRequest(BaseModel):
    input: str = "network.rmp"
    output: str = "route.gpx"
    oneway: str = "respect"
    left_penalty: float = 1.0
    right_penalty: float = 0.0
    uturn_penalty: float = 5.0
    depot: Optional[str] = None # "lat,lon"

class VrpRequest(BaseModel):
    input: str = "network.rmp"
    output_dir: str = "routes/"
    vehicles: int = 1
    algo: str = "greedy" # greedy, savings, local-search, simulated-annealing
    city: Optional[str] = "montreal"
    generate_stops: Optional[int] = 10
    bbox: Optional[str] = None # "MIN_LON,MIN_LAT,MAX_LON,MAX_LAT"
    is_drone: bool = False

class AgentTask(BaseModel):
    task_file: str = "agent_task.json"

# --- Helpers ---

def run_rmpca(args: List[str]) -> Dict:
    cmd = ["rmpca"] + args + ["--json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout
        try:
            # Try to parse JSON error if available
            return json.loads(error_msg)
        except:
            raise HTTPException(status_code=500, detail=f"CLI Error: {error_msg}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse CLI output as JSON")

# --- Endpoints ---

@app.post("/extract", tags=["Core"])
async def extract(req: ExtractRequest):
    """Extract road network data from Overture Maps."""
    args = ["extract", "--bbox", req.bbox, "--source", req.source, "--output", req.output]
    return run_rmpca(args)

@app.post("/compile", tags=["Core"])
async def compile_map(req: CompileRequest):
    """Compile GeoJSON into binary .rmp format."""
    args = ["compile", "--input", req.input, "--output", req.output]
    return run_rmpca(args)

@app.post("/clean", tags=["Core"])
async def clean(req: CleanRequest):
    """Clean a GeoJSON road network."""
    args = ["clean", "--input", req.input, "--output", req.output]
    if req.min_length_m: args += ["--min-length-m", str(req.min_length_m)]
    if req.snap_dist_m: args += ["--snap-dist-m", str(req.snap_dist_m)]
    return run_rmpca(args)

@app.post("/optimize", tags=["Routing"])
async def optimize(req: OptimizeRequest):
    """Optimize a route (Chinese Postman Problem)."""
    args = ["optimize", "--input", req.input, "--output", req.output, 
            "--oneway", req.oneway, "--left-penalty", str(req.left_penalty),
            "--right-penalty", str(req.right_penalty), "--uturn-penalty", str(req.uturn_penalty)]
    if req.depot: args += ["--depot", req.depot]
    return run_rmpca(args)

@app.post("/vrp", tags=["Routing"])
async def vrp(req: VrpRequest):
    """Solve Vehicle Routing Problem (VRP)."""
    args = ["vrp", "--input", req.input, "--output-dir", req.output_dir, 
            "--vehicles", str(req.vehicles), "--algo", req.algo]
    if req.city: args += ["--city", req.city]
    if req.generate_stops: args += ["--generate-stops", str(req.generate_stops)]
    if req.bbox: args += ["--bbox", req.bbox]
    if req.is_drone: args += ["--is-drone"]
    return run_rmpca(args)

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "0.4.8"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
