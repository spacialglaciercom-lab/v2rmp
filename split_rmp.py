import struct
import zlib
import sys

def split_rmp(input_path):
    with open(input_path, "rb") as f:
        data = f.read()

    magic = data[0:4]
    if magic != b"RMP1":
        print("Invalid magic")
        return

    node_count = struct.unpack("<I", data[4:8])[0]
    edge_count = struct.unpack("<I", data[8:12])[0]

    nodes = []
    for i in range(node_count):
        offset = 12 + i * 16
        lat, lon = struct.unpack("<dd", data[offset:offset+16])
        nodes.append({"lat": lat, "lon": lon, "id": i})

    edges_start = 12 + node_count * 16
    edges = []
    for i in range(edge_count):
        offset = edges_start + i * 17
        u, v, weight, oneway = struct.unpack("<IIdB", data[offset:offset+17])
        edges.append({"u": u, "v": v, "weight": weight, "oneway": oneway})

    # Find lon bounds
    lons = [n["lon"] for n in nodes]
    min_lon = min(lons)
    max_lon = max(lons)
    mid_lon = (min_lon + max_lon) / 2.0

    print(f"BBox: {min_lon} to {max_lon}, Mid: {mid_lon}")

    def create_half(is_left):
        if is_left:
            half_nodes_indices = [i for i, n in enumerate(nodes) if n["lon"] < mid_lon]
        else:
            half_nodes_indices = [i for i, n in enumerate(nodes) if n["lon"] >= mid_lon]

        # Map old node indices to new ones
        new_node_map = {old: i for i, old in enumerate(half_nodes_indices)}
        new_nodes = [nodes[old] for old in half_nodes_indices]

        # Keep edges where both nodes are in the same half
        new_edges = []
        for e in edges:
            if e["u"] in new_node_map and e["v"] in new_node_map:
                new_edges.append({
                    "u": new_node_map[e["u"]],
                    "v": new_node_map[e["v"]],
                    "weight": e["weight"],
                    "oneway": e["oneway"]
                })

        # Pack
        header = b"RMP1" + struct.pack("<II", len(new_nodes), len(new_edges))
        node_data = b""
        for n in new_nodes:
            node_data += struct.pack("<dd", n["lat"], n["lon"])
        
        edge_data = b""
        for e in new_edges:
            edge_data += struct.pack("<IIdB", e["u"], e["v"], e["weight"], e["oneway"])
        
        payload = header + node_data + edge_data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return payload + struct.pack("<I", crc)

    with open("half1.rmp", "wb") as f:
        f.write(create_half(True))
    
    with open("half2.rmp", "wb") as f:
        f.write(create_half(False))

    print(f"Created half1.rmp and half2.rmp")

if __name__ == "__main__":
    split_rmp("../Downloads/test_filtered.rmp")
