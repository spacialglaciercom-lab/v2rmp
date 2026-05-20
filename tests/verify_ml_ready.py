#!/usr/bin/env python3
"""Quick validation that ml_ready fields propagate correctly."""
import json
import subprocess
import select

proc = subprocess.Popen(
    ["target/quick/rmpca-mcp-server"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1, cwd="/root/v2rmp"
)

def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    ready, _, _ = select.select([proc.stdout], [], [], 10)
    if not ready:
        raise TimeoutError("No response")
    return json.loads(proc.stdout.readline())

# init
send({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}})
# notification — no response expected
proc.stdin.write(json.dumps({"jsonrpc":"2.0","method":"notifications/initialized"}) + "\n")
proc.stdin.flush()


def call_tool(name, args):
    resp = send({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":args}})
    content = resp.get("result", {}).get("content", [])
    return json.loads(content[0].get("text", "{}"))

stops = [
    {"lat": 45.76, "lon": -0.04, "label": "Depot", "demand": 0},
    {"lat": 45.761, "lon": -0.039, "label": "A", "demand": 1},
    {"lat": 45.759, "lon": -0.041, "label": "B", "demand": 2},
    {"lat": 45.762, "lon": -0.038, "label": "C", "demand": 1},
]

print("=== predict_solver ===")
res = call_tool("predict_solver", {"stops": stops, "num_vehicles": 1})
print("  ml_ready  :", res.get("ml_ready"))
print("  model_loaded:", res.get("model_loaded"))
print("  recommended:", res.get("recommended"))
assert "ml_ready" in res, "ml_ready missing from predict_solver"
assert res["ml_ready"], f"Expected ml_ready=True with model present, got {res['ml_ready']}"

print("\n=== predict_quality ===")
res = call_tool("predict_quality", {"stops": stops, "num_vehicles": 1})
print("  ml_ready  :", res.get("ml_ready"))
print("  confidence:", res.get("confidence"))
assert "ml_ready" in res, "ml_ready missing from predict_quality"
assert res["ml_ready"]

print("\n=== tune_hyperparams ===")
res = call_tool("tune_hyperparams", {"stops": stops, "num_vehicles": 1})
print("  ml_ready  :", res.get("ml_ready"))
print("  max_iter  :", res.get("max_iterations"))
assert "ml_ready" in res, "ml_ready missing from tune_hyperparams"
assert res["ml_ready"]

proc.stdin.close()
proc.wait(timeout=5)
print("\n✅ All ml_ready fields present and correct.")
