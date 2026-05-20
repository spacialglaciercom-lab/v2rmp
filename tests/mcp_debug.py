#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path
import select
"""Minimal debug script to isolate the hang."""

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "target" / "release" / "rmpca-mcp-server"

print("Launching server ...")
proc = subprocess.Popen(
    [str(SERVER)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)
print(f"PID {proc.pid}")

# Give it a moment
print("Sending initialize ...")
proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}) + "\n")
proc.stdin.flush()

# Read response with timeout
ready, _, _ = select.select([proc.stdout], [], [], 10)
if ready:
    line = proc.stdout.readline()
    print("INIT RESPONSE:", line.strip())
else:
    print("TIMEOUT waiting for initialize response")
    proc.kill()
    sys.exit(1)

print("Sending tools/list ...")
proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/list"}) + "\n")
proc.stdin.flush()

ready, _, _ = select.select([proc.stdout], [], [], 10)
if ready:
    line = proc.stdout.readline()
    print("LIST RESPONSE:", line[:200])
else:
    print("TIMEOUT waiting for tools/list")

print("Sending list_solvers ...")
proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_solvers","arguments":{}}}) + "\n")
proc.stdin.flush()

ready, _, _ = select.select([proc.stdout], [], [], 10)
if ready:
    line = proc.stdout.readline()
    print("SOLVERS RESPONSE:", line[:300])
else:
    print("TIMEOUT waiting for list_solvers")

proc.stdin.close()
proc.wait(timeout=5)
print("Done")
