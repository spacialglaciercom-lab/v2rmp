import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.numpy import save_file
from torch.utils.data import Dataset, DataLoader
import random

# Architecture matching src/core/ml/graph_embed.rs
class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim, has_bias=True):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=has_bias)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, x_neigh):
        return self.lin_l(x) + self.lin_r(x_neigh)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, adj_matrix):
        # adj_matrix should be normalized adjacency: D^-1 * A
        
        # Conv 1
        x_neigh1 = torch.sparse.mm(adj_matrix, x)
        h1 = F.relu(self.conv1(x, x_neigh1))
        
        # Conv 2
        x_neigh2 = torch.sparse.mm(adj_matrix, h1)
        h2 = F.relu(self.conv2(h1, x_neigh2))
        
        return h2

def load_geojson_as_line_graph(path):
    print(f"Loading {path}...")
    with open(path, "r") as f:
        data = json.load(f)
    
    features = []
    # In line-graph, road segments are nodes.
    # We need to find segments that share a coordinate.
    
    coord_to_edges = {}
    edges = []
    
    for i, feat in enumerate(data["features"]):
        if feat["geometry"]["type"] != "LineString":
            continue
            
        coords = feat["geometry"]["coordinates"]
        if not coords:
            continue
            
        # Feature extraction (matching Rust core)
        # 1. length
        # Simple distance approximation
        p1 = coords[0]
        p2 = coords[-1]
        dist_km = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 111.0
        
        # 2. oneway
        oneway = 1.0 if feat["properties"].get("oneway") in ["yes", "1", True] else 0.0
        
        f_vec = [dist_km, oneway] + [0.0]*8
        features.append(f_vec)
        
        edge_idx = len(edges)
        edges.append(edge_idx)
        
        # Map endpoints to edge index
        start_node = tuple(coords[0])
        end_node = tuple(coords[-1])
        
        for node in [start_node, end_node]:
            if node not in coord_to_edges:
                coord_to_edges[node] = []
            coord_to_edges[node].append(edge_idx)

    num_nodes = len(edges)
    print(f"  Line-graph has {num_nodes} nodes (road segments)")
    
    adj = [[] for _ in range(num_nodes)]
    for node, connected_edges in coord_to_edges.items():
        for i in range(len(connected_edges)):
            for j in range(i + 1, len(connected_edges)):
                u, v = connected_edges[i], connected_edges[j]
                adj[u].append(v)
                adj[v].append(u)
                
    # Use sparse adjacency matrix
    indices = []
    values = []
    for i in range(num_nodes):
        neighbors = adj[i]
        if neighbors:
            for n in neighbors:
                indices.append([i, n])
                values.append(1.0 / len(neighbors))
    
    if not indices:
        indices = [[0, 0]]
        values = [0.0]
        
    i_tensor = torch.LongTensor(indices).t()
    v_tensor = torch.FloatTensor(values)
    adj_sparse = torch.sparse_coo_tensor(i_tensor, v_tensor, torch.Size([num_nodes, num_nodes]))
                
    return torch.tensor(features, dtype=torch.float32), adj_sparse, adj

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    cities = ["data/montreal_clean.geojson", "data/vancouver_clean.geojson"]
    
    model = GraphSAGE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Task: Link prediction (Unsupervised)
    # Loss: -log(sigmoid(dot(u, v))) for connected u, v
    #       -log(1 - sigmoid(dot(u, n))) for random negative n
    
    for city_path in cities:
        if not os.path.exists(city_path):
            print(f"Skipping {city_path}, not found.")
            continue
            
        x, adj_matrix, adj_list = load_geojson_as_line_graph(city_path)
        x = x.to(device)
        adj_matrix = adj_matrix.to(device)
        
        num_nodes = x.size(0)
        
        print(f"Training on {city_path}...")
        for epoch in range(50): # Increased epochs
            model.train()
            optimizer.zero_grad()
            
            # Use torch.sparse.mm for sparse multiplication
            embeddings = model(x, adj_matrix)
            
            # Sample positive edges
            pos_u = []
            pos_v = []
            for u in range(num_nodes):
                if adj_list[u]:
                    v = random.choice(adj_list[u])
                    pos_u.append(u)
                    pos_v.append(v)
            
            if not pos_u:
                continue
                
            pos_u = torch.tensor(pos_u, device=device)
            pos_v = torch.tensor(pos_v, device=device)
            
            # Sample negative edges
            neg_v = torch.randint(0, num_nodes, (len(pos_u),), device=device)
            
            # Dot products
            pos_scores = (embeddings[pos_u] * embeddings[pos_v]).sum(dim=1)
            neg_scores = (embeddings[pos_u] * embeddings[neg_v]).sum(dim=1)
            
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores) + 1e-6) + 
                               torch.log(1 - torch.sigmoid(neg_scores) + 1e-6))
            
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

    # Export to safetensors
    print("Exporting models/graph_embed.safetensors...")
    state_dict = model.state_dict()
    candle_state = {}
    for k, v in state_dict.items():
        # Map PyTorch keys to Candle keys used in graph_embed.rs
        # Rust expects: conv1.lin_l.weight, conv1.lin_l.bias, conv1.lin_r.weight, etc.
        candle_state[k] = v.cpu().numpy()
        
    save_file(candle_state, "models/graph_embed.safetensors")
    print("Done!")

if __name__ == "__main__":
    train()
