import json
import subprocess
import select
import time
import threading

class MCPClient:
    def __init__(self, bin_path):
        self.proc = subprocess.Popen([str(bin_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self._id = 1
        self.stderr_content = []
        self.stderr_thread = threading.Thread(target=self._read_stderr)
        self.stderr_thread.daemon = True
        self.stderr_thread.start()
        
        # initialize
        self._send({"jsonrpc":"2.0","id":0,"method":"initialize","params":{}})
        self._write({"jsonrpc":"2.0","method":"notifications/initialized"})
        time.sleep(0.5)

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.stderr_content.append(line)

    def _write(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _send(self, obj):
        self._write(obj)
        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], 30)
            if not ready:
                raise TimeoutError("No response from MCP server")
            line = self.proc.stdout.readline()
            if not line:
                print("Stderr output:")
                print("".join(self.stderr_content))
                raise RuntimeError("EOF from server")
            resp = json.loads(line)
            if "method" in resp and resp["method"].startswith("notifications/"):
                continue
            return resp

    def list_tools(self):
        req = {"jsonrpc":"2.0","id":self._id,"method":"tools/list","params":{}}
        self._id += 1
        return self._send(req)

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

def extract_text(resp):
    result = resp.get("result", {})
    content = result.get("content", [])
    if not content:
        return json.dumps(result)
    return content[0].get("text", "{}")

if __name__ == "__main__":
    client = MCPClient("/home/rmp/v2rmp/target/debug/rmpca-mcp")
    tools_resp = client.list_tools()
    tools = [t["name"] for t in tools_resp.get("result", {}).get("tools", [])]
    print(f"Available tools: {tools}")

    target_tool = "v2rmp_neural_optimize"

    args = {
    "model_path": "/home/rmp/v2rmp/cvrp50_model.onnx",
    "locations": [
        [45.5017, -73.5673], [45.5088, -73.5540], [45.4948, -73.5779], [45.5122, -73.5547],
        [45.5195, -73.6219], [45.4312, -73.5934], [45.5410, -73.6276], [45.5042, -73.6143],
        [45.5161, -73.5684], [45.5284, -73.5972], [45.4851, -73.5285], [45.4475, -73.6821],
        [45.5398, -73.5512], [45.5583, -73.5519], [45.6015, -73.6331], [45.5020, -73.4474],
        [45.4678, -73.7412], [45.4278, -73.8341], [45.4116, -73.6845], [45.4897, -73.6231]
    ],
    "demands": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "capacity": 10.0
    }

    print(f"Calling {target_tool} ...")
    try:
        resp = client.call(target_tool, args)
        print(extract_text(resp))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()
