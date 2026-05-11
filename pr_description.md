🧪 Add tests for JSON bbox parser in rmpca-mcp-server

🎯 What: Added tests for the parse_bbox function in src/bin/rmpca-mcp-server.rs to ensure it accurately parses bounding box parameters from a JSON object. Fixed unrelated clippy issues in the same file preventing clean build.
📊 Coverage: Added scenarios for a fully valid bbox, missing bbox root object, missing internal required fields (min_lon, min_lat, max_lon, max_lat), and invalid property types.
✨ Result: Improved reliability and confidence in argument parsing for the MCP server.
