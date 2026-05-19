#!/usr/bin/env python3
"""
CVRP Neural Solver — Traceable Attention Model for ONNX export.

This implements the exact same architecture as train_cvrp.py but with a
fully traceable decoder that uses no data-dependent Python control flow.
The traceable version is used for ONNX export only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# Attention Components (same as train_cvrp.py)
# ============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)
    
    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_q, _ = q.shape
        N_kv = kv.shape[1]
        
        Q = self.Wq(q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N_q, self.dim)
        return self.Wo(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.mha = MultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.mha(x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class AMEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, ff_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(3, dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, locs: torch.Tensor, demands: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([locs, demands], dim=-1)  # [B, N+1, 3]
        x = self.input_proj(x)                   # [B, N+1, dim]
        
        for layer in self.layers:
            x = layer(x)
        
        graph_embed = x.mean(dim=1, keepdim=True)  # [B, 1, dim]
        return x, graph_embed


# ============================================================================
# Traceable Decoder — no Python control flow, fixed max steps
# ============================================================================

class TraceableDecoder(nn.Module):
    """
    Fully traceable autoregressive decoder.
    Uses a fixed number of decoding steps (max_steps = 2*N) and tensor masking
    instead of Python if/break. Compatible with torch.export and ONNX export.
    """
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.tanh_clip = 10.0
        
        self.context_proj = nn.Linear(dim * 3, dim)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
    
    def forward(self, node_embeddings: torch.Tensor,
                graph_embed: torch.Tensor,
                demands: torch.Tensor,
                capacity: torch.Tensor,
                max_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embeddings: [B, N+1, dim]
            graph_embed:     [B, 1, dim]
            demands:         [B, N+1]
            capacity:        [B, 1]
            max_steps:       fixed number of decoding steps
        
        Returns:
            actions: [B, max_steps]
            log_p:   [B, max_steps]
        """
        B, Np1, _ = node_embeddings.shape
        device = node_embeddings.device
        
        # Pre-compute keys
        keys = self.Wk(node_embeddings)  # [B, N+1, dim]
        
        # Initial state
        first_node_embed = node_embeddings[:, 0:1, :]  # depot
        
        # We'll unroll the loop into a sequential scan.
        # At each step t:
        #   - compute compat from current position
        #   - apply masks
        #   - get action and log_prob
        #   - update state for next step
        
        # Track: visited mask, current position, remaining capacity
        # All updated via tensor operations (no if/break)
        
        actions_list = []
        log_p_list = []
        
        # State tensors
        visited = torch.zeros(B, Np1, dtype=torch.bool, device=device)
        visited[:, 0] = True
        cur_loc = torch.zeros(B, dtype=torch.long, device=device)  # 0 = depot
        rem_cap = capacity.clone()
        
        for _ in range(max_steps):
            # Get current node embedding
            cur_embed = node_embeddings[torch.arange(B), cur_loc].unsqueeze(1)  # [B, 1, dim]
            
            # Context: [graph_embed | first_node | current_node]
            ctx = torch.cat([graph_embed, first_node_embed, cur_embed], dim=-1)  # [B, 1, 3*dim]
            context = self.context_proj(ctx).squeeze(1)  # [B, dim]
            
            # Query
            query = self.Wq(context)  # [B, dim]
            
            # Compatibility scores
            compat = (query.unsqueeze(1) * keys).sum(dim=-1) * self.scale  # [B, N+1]
            compat = self.tanh_clip * torch.tanh(compat / self.tanh_clip)
            
            # Mask visited nodes
            compat = compat.masked_fill(visited, float('-inf'))
            
            # Mask nodes exceeding capacity
            demand_exceeds = demands > rem_cap
            compat = compat.masked_fill(demand_exceeds, float('-inf'))
            
            # Always allow depot
            # (no condition on step — depot is always visitable, but we want to force
            # the model to visit customers first. In traceable mode, we just let
            # the mask handle it)
            
            # Softmax
            probs = F.softmax(compat, dim=-1)
            
            # Greedy action
            action = torch.argmax(probs, dim=-1)  # [B]
            log_prob = torch.log(probs[torch.arange(B), action] + 1e-20)
            
            actions_list.append(action)
            log_p_list.append(log_prob)
            
            # State update — fully tensorized
            visited = visited | (torch.arange(Np1, device=device).unsqueeze(0) == action.unsqueeze(1))
            cur_loc = action
            
            # Reset capacity at depot
            at_depot = (action == 0).unsqueeze(1)  # [B, 1]
            rem_cap = torch.where(at_depot, capacity, rem_cap)
            
            # Consume demand at customer
            consumed = demands[torch.arange(B), action].unsqueeze(1)  # [B, 1]
            rem_cap = rem_cap - consumed  # subtract (will be corrected above if at depot)
            rem_cap = torch.where(at_depot, capacity, rem_cap)  # fix: don't subtract at depot
        
        actions = torch.stack(actions_list, dim=-1)  # [B, max_steps]
        log_p = torch.stack(log_p_list, dim=-1)      # [B, max_steps]
        
        return actions, log_p


# ============================================================================
# Full Traceable Model
# ============================================================================

class TraceableCVRPModel(nn.Module):
    """
    CVRP solver with traceable decoder. Identical architecture to CVPRouteModel
    but decoder uses tensor-only operations for ONNX compatibility.
    """
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8,
                 num_encoder_layers: int = 6, ff_hidden_dim: int = 512):
        super().__init__()
        self.encoder = AMEncoder(embedding_dim, num_heads, num_encoder_layers, ff_hidden_dim)
        self.decoder = TraceableDecoder(embedding_dim, num_heads)
    
    def forward(self, locs: torch.Tensor, demands: torch.Tensor,
                capacity: torch.Tensor, max_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            locs:     [B, N+1, 2]
            demands:  [B, N+1, 1]
            capacity: [B, 1]
            max_steps: decoding steps
        
        Returns:
            actions: [B, max_steps]
            log_p:   [B, max_steps]
        """
        demands_flat = demands.squeeze(-1)  # [B, N+1]
        node_emb, graph_emb = self.encoder(locs, demands)
        actions, log_p = self.decoder(node_emb, graph_emb, demands_flat, capacity, max_steps)
        return actions, log_p


# ============================================================================
# ONNX Export Wrapper — matches v2rmp MCP server interface
# ============================================================================

class ONNXExportWrapper(nn.Module):
    """
    Wraps the traceable model to match the exact I/O the v2rmp MCP server expects:
    
    Inputs:
      locs:     [1, N, 2]   — customer locations only (no depot)
      demand:   [1, N, 1]   — customer demands
      capacity: [1, 1]      — vehicle capacity
    
    Outputs:
      actions: [1, max_steps] — visit sequence (includes depot returns as 0)
      log_p:   [1, max_steps] — log probabilities
    
    The depot is implicitly at (0.5, 0.5) with demand 0, automatically prepended.
    """
    
    def __init__(self, model: TraceableCVRPModel):
        super().__init__()
        self.model = model
    
    def forward(self, locs: torch.Tensor, demand: torch.Tensor,
                capacity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = locs.shape[0]
        N = locs.shape[1]
        max_steps = N * 3  # generous upper bound
        
        # Prepend depot
        depot_loc = torch.full((B, 1, 2), 0.5, device=locs.device, dtype=locs.dtype)
        full_locs = torch.cat([depot_loc, locs], dim=1)  # [1, N+1, 2]
        
        depot_demand = torch.zeros(B, 1, 1, device=demand.device, dtype=demand.dtype)
        full_demands = torch.cat([depot_demand, demand], dim=1)  # [1, N+1, 1]
        
        actions, log_p = self.model(full_locs, full_demands, capacity, max_steps)
        return actions, log_p


def export_trained_to_onnx(checkpoint_path: str, onnx_path: str,
                           onnx_problem_size: int = 50):
    """
    Load a PyTorch checkpoint from train_cvrp.py, build traceable model, export to ONNX.
    """
    import os
    from train_cvrp import CVPRouteModel, CVRPConfig
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = ckpt["config"]
    
    # Build traceable model with same architecture
    model = TraceableCVRPModel(
        embedding_dim=train_config.embedding_dim,
        num_heads=train_config.num_heads,
        num_encoder_layers=train_config.num_encoder_layers,
        ff_hidden_dim=train_config.ff_hidden_dim,
    )
    
    # Load weights from trained model
    trained_model = CVPRouteModel(train_config)
    trained_model.load_state_dict(ckpt["model_state_dict"])
    
    # Copy weights: encoder directly, decoder partially
    model.encoder.load_state_dict(trained_model.encoder.state_dict())
    # Decoder: Wq, Wk, context_proj are shared
    model.decoder.Wq.load_state_dict(trained_model.decoder.Wq.state_dict())
    model.decoder.Wk.load_state_dict(trained_model.decoder.Wk.state_dict())
    model.decoder.context_proj.load_state_dict(trained_model.decoder.context_proj.state_dict())
    
    model.eval()
    
    wrapper = ONNXExportWrapper(model)
    wrapper.eval()
    
    # Dummy inputs
    dummy_locs = torch.randn(1, onnx_problem_size, 2)
    dummy_demand = torch.ones(1, onnx_problem_size, 1)
    dummy_capacity = torch.tensor([[1.0]])
    
    print(f"Exporting ONNX model to {onnx_path} (problem_size={onnx_problem_size})...")
    
    # Export directly — the traceable model has no data-dependent control flow
    torch.onnx.export(
        wrapper,
        (dummy_locs, dummy_demand, dummy_capacity),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['locs', 'demand', 'capacity'],
        output_names=['actions', 'log_p'],
    )
    
    # Validate
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Check with onnxruntime
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(onnx_path)
    result = sess.run(None, {
        'locs': dummy_locs.numpy().astype(np.float32),
        'demand': dummy_demand.numpy().astype(np.float32),
        'capacity': dummy_capacity.numpy().astype(np.float32),
    })
    
    print(f"✅ ONNX model exported and validated: {onnx_path}")
    print(f"   actions shape: {result[0].shape}, log_p shape: {result[1].shape}")
    print(f"   First 15 actions: {result[0][0, :15]}")
    
    return onnx_path
