# mypy: ignore-errors
#!/usr/bin/env python3
"""
tests/check_mcp_compatibility.py — MCP server tool compatibility guard.

Parses the MCP server source (src/bin/mcp-server.rs) for tool definitions
and compares against the baseline in mcp_baseline.json. Reports:

- Added tools (new names not in baseline)
- Removed tools (names in baseline but missing from source)
- Changed tools (same name, different schema or description)

Exit 0 if source matches baseline, 1 on drift.

Usage:
    python3 tests/check_mcp_compatibility.py [--update]
    --update  rewrites mcp_baseline.json from the current source
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_SRC = ROOT / "src" / "bin" / "mcp-server.rs"
BASELINE_PATH = ROOT / "mcp_baseline.json"


def parse_mcp_tools(source_text: str) -> list[dict]:
    """Extract tool definitions from the MCP server Rust source.

    The server defines tools in a `tools/list` handler as a JSON array literal.
    We extract that array and parse it as JSON. If that fails, we fall back to
    regex-based extraction of individual tool objects.
    """
    tools = []

    # Strategy 1: find the `"tools": [` JSON block in tools/list response
    # The tools array spans from `json!({ "tools": [` to the closing `])` of the
    # `respond` call. We'll extract the inner array.
    match = re.search(
        r'"tools":\s*(\[\s*\{.*?\}\s*\])',
        source_text,
        re.DOTALL,
    )
    if match:
        try:
            tools = json.loads(match.group(1))
            if isinstance(tools, list) and len(tools) > 0:
                return tools
        except json.JSONDecodeError:
            pass

    # Strategy 2: extract per-tool JSON objects from individual
    # `{ "name": "...", ... }` blocks within tools/list.
    # Find the tools array manually.
    tools_start = source_text.find('"tools": [')
    if tools_start < 0:
        print("ERROR: Could not find tools array in MCP server source.", file=sys.stderr)
        sys.exit(2)

    # Find the matching closing bracket by tracking JSON nesting.
    depth = 0
    i = tools_start + len('"tools": [') - 1
    while i < len(source_text):
        c = source_text[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                tools_json_str = source_text[tools_start + len('"tools": '):i + 1]
                try:
                    tools = json.loads(tools_json_str)
                    return tools
                except json.JSONDecodeError:
                    break
        i += 1

    if not tools:
        print("ERROR: Failed to parse tools from MCP server source.", file=sys.stderr)
        sys.exit(2)

    return tools


def normalize_tool(tool: dict) -> dict:
    """Return a canonical dict suitable for comparison."""
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "inputSchema": tool.get("inputSchema", {}),
    }


def compare_tools(source_tools: list[dict], baseline_tools: list[dict]) -> int:
    """Compare source tools against baseline. Return count of differences."""
    source_map: dict[str, dict] = {t["name"]: t for t in source_tools}
    baseline_map: dict[str, dict] = {t["name"]: t for t in baseline_tools}

    source_names = set(source_map)
    baseline_names = set(baseline_map)

    added = source_names - baseline_names
    removed = baseline_names - source_names
    changed = []

    for name in source_names & baseline_names:
        s_norm = normalize_tool(source_map[name])
        b_norm = normalize_tool(baseline_map[name])
        # Compare using JSON-serialized forms for structural equality
        if json.dumps(s_norm, sort_keys=True) != json.dumps(b_norm, sort_keys=True):
            changed.append(name)

    diffs = len(added) + len(removed) + len(changed)

    if diffs == 0:
        print("✓ MCP server tools match baseline — no regressions detected.")
        return 0

    # Report drift
    print(f"✗ MCP server tools have drifted from baseline ({diffs} difference(s)):")
    if added:
        print(f"\n  + ADDED ({len(added)}):")
        for name in sorted(added):
            print(f"      {name}")
    if removed:
        print(f"\n  - REMOVED ({len(removed)}):")
        for name in sorted(removed):
            print(f"      {name}")
    if changed:
        print(f"\n  ~ CHANGED ({len(changed)}):")
        for name in sorted(changed):
            print(f"      {name}")

    print("\n  Run with --update to refresh the baseline.")
    return 1


def update_baseline(source_tools: list[dict]):
    """Write the current source tools to mcp_baseline.json."""
    from datetime import datetime, timezone

    baseline = {
        "description": "Baseline snapshot of v2rmp MCP server tools. Used by tests/check_mcp_compatibility.py to detect regressions.",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "server_name": "v2rmp-mcp-server",
        "protocol_version": "2024-11-05",
        "tools": source_tools,
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"✓ Updated {BASELINE_PATH} ({len(source_tools)} tools)")


def main():
    do_update = "--update" in sys.argv or "-u" in sys.argv

    if not MCP_SERVER_SRC.exists():
        print(f"ERROR: MCP server source not found: {MCP_SERVER_SRC}", file=sys.stderr)
        sys.exit(2)

    if not BASELINE_PATH.exists():
        print(f"ERROR: Baseline not found: {BASELINE_PATH}", file=sys.stderr)
        print("  Run with --update to create it from source.", file=sys.stderr)
        sys.exit(2)

    source_text = MCP_SERVER_SRC.read_text()
    source_tools = parse_mcp_tools(source_text)

    if do_update:
        update_baseline(source_tools)
        return 0

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    baseline_tools = baseline.get("tools", [])
    if not baseline_tools:
        print("ERROR: Baseline has no tools.", file=sys.stderr)
        sys.exit(2)

    return compare_tools(source_tools, baseline_tools)


if __name__ == "__main__":
    sys.exit(main())
