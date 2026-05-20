#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = ROOT / "target" / "debug" / "rmpca-mcp"

def send(proc, msg):
    line = json.dumps(msg)
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

def recv(proc):
    line = proc.stdout.readline()
    if not line:
        return None
    return json.loads(line)

def test():
    if not SERVER_BIN.exists():
        print(f"Error: {SERVER_BIN} not found. Run 'cargo build --bin rmpca-mcp'")
        sys.exit(1)

    print(f"Starting server: {SERVER_BIN}")
    proc = subprocess.Popen(
        [str(SERVER_BIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        # 1. Initialize
        print("[1] Initializing...")
        send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        resp = recv(proc)
        print("Response:", json.dumps(resp, indent=2))

        # 2. List tools
        print("\n[2] Listing tools...")
        send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        resp = recv(proc)
        tools = resp.get("result", {}).get("tools", [])
        print(f"Found {len(tools)} tools:")
        for t in tools:
            print(f" - {t['name']}")

        # 3. Test a specific tool: v2rmp_compile (requires a file, so let's try ping or something simple if available)
        # Actually, let's just check the tools list is correct.
        
        # v2rmp_optimize, v2rmp_compile etc.
        # Since I don't want to create complex data here, I'll just check if they are in the list.
        tool_names = [t['name'] for t in tools]
        expected = ["v2rmp_compile", "v2rmp_optimize"]
        for exp in expected:
            if exp in tool_names:
                print(f"✅ Tool '{exp}' is registered.")
            else:
                print(f"❌ Tool '{exp}' is MISSING.")

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test()
