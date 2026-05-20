import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 1. Advanced Policy: Graph Convolutional Network (GCN)
class GNNPolicy(nn.Module):
    def __init__(self, node_dim=64, meta_dim=3, hidden_dim=128):
        super(GNNPolicy, self).__init__()
        # GCN Layer: Aggregates neighbor embeddings
        self.conv1 = nn.Linear(node_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy Head: Combines GNN features with Dynamic Meta state
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Probability/Score for this node
        )

    def forward(self, node_embeddings, adj_matrix, meta_state):
        # node_embeddings: [N, 64]
        # adj_matrix: [N, N] (Sparse or Dense)
        # meta_state: [1, 3] (Wind, Battery, Load)
        
        # Simple Graph Convolution: H = ReLU(A * X * W)
        x = F.relu(torch.matmul(adj_matrix, self.conv1(node_embeddings)))
        x = F.relu(torch.matmul(adj_matrix, self.conv2(x)))
        
        # Combine node features with the drone's dynamic state
        # We broadcast meta_state to all nodes
        meta_expanded = meta_state.repeat(x.size(0), 1)
        combined = torch.cat([x, meta_expanded], dim=1)
        
        scores = self.actor(combined)
        return scores # Returns scores for EVERY node on the map


# 2. Implementation Strategy for ONNX
def export_gnn():
    # Use 100 nodes as a representative sample for export
    num_nodes = 100
    node_dim = 64
    meta_dim = 3
    
    model = GNNPolicy(node_dim, meta_dim)
    model.eval() # Set to evaluation mode before export
    
    # Dummy inputs
    node_embs = torch.randn(num_nodes, node_dim)
    adj = torch.eye(num_nodes) # Identity as dummy adjacency
    meta = torch.randn(1, meta_dim)
    
    onnx_path = "gnn_drone_agent.onnx"
    if os.path.exists("/data"):
        onnx_path = "/data/gnn_drone_agent.onnx"
    
    # Export to ONNX with dynamic axes for the number of nodes
    torch.onnx.export(
        model, 
        (node_embs, adj, meta), 
        onnx_path,
        input_names=['node_embeddings', 'adj_matrix', 'meta_state'],
        output_names=['node_scores'],
        dynamic_axes={
            'node_embeddings': {0: 'num_nodes'},
            'adj_matrix': {0: 'num_nodes', 1: 'num_nodes'}
        }
    )
    print(f"Exported GNN Policy to {onnx_path}")

if __name__ == "__main__":
    export_gnn()
