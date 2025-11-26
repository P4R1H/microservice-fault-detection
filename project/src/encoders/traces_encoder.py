"""
Trace encoder with GNN on service dependency graphs.

V4.2 Update: Full GCN integration with time-series to graph conversion.

Pipeline:
1. Parse traces → service dependency graph
2. Extract node features (latency, error rate, request count)
3. Extract edge features (call frequency, latency)
4. Apply GCN/GAT for graph encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Literal
import numpy as np


try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not installed. GNN encoders unavailable.")


class TraceGraphBuilder(nn.Module):
    """
    Converts trace time-series to graph format for GCN/GAT.
    
    Input: (batch, n_services, seq_len, features) - trace time series
    Output: (node_features, edge_index, batch_assignment) - graph format
    
    Node features: Learned temporal aggregation of trace features
    Edges: Learned soft adjacency matrix OR fully connected
    """
    
    def __init__(
        self,
        n_services: int,
        n_trace_features: int = 32,
        hidden_dim: int = 32,
        node_feature_dim: int = 64,
        edge_mode: Literal['learned', 'full', 'correlation'] = 'learned',
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_services = n_services
        self.node_feature_dim = node_feature_dim
        self.edge_mode = edge_mode
        
        # === Temporal aggregation to create node features ===
        # Multi-scale temporal features
        self.temporal_conv = nn.Sequential(
            # Local patterns
            nn.Conv1d(n_trace_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Larger patterns  
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Attention-based temporal pooling
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Feature projection to node_feature_dim
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, node_feature_dim),
            nn.LayerNorm(node_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # === Edge learning (if mode='learned') ===
        if edge_mode == 'learned':
            # Learn service relationship embeddings
            self.edge_src_embed = nn.Linear(node_feature_dim, hidden_dim)
            self.edge_dst_embed = nn.Linear(node_feature_dim, hidden_dim)
            self.edge_threshold = nn.Parameter(torch.tensor(0.0))  # Learnable threshold
    
    def _aggregate_temporal(self, traces: torch.Tensor) -> torch.Tensor:
        """
        Aggregate time series into per-service node features.
        
        Args:
            traces: (batch * n_services, seq_len, features)
        Returns:
            (batch * n_services, node_feature_dim)
        """
        # Temporal convolution
        x = traces.permute(0, 2, 1)  # (B*S, features, seq_len)
        x = self.temporal_conv(x)  # (B*S, hidden_dim, seq_len)
        x = x.permute(0, 2, 1)  # (B*S, seq_len, hidden_dim)
        
        # Attention-weighted pooling
        attn_scores = self.time_attention(x)  # (B*S, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        x = (x * attn_weights).sum(dim=1)  # (B*S, hidden_dim)
        
        # Project to node feature dim
        x = self.node_proj(x)  # (B*S, node_feature_dim)
        
        return x
    
    def _build_edges(
        self, 
        node_features: torch.Tensor, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge_index for the service graph.
        
        Args:
            node_features: (batch * n_services, node_feature_dim)
            batch_size: Number of samples in batch
        
        Returns:
            edge_index: (2, num_edges)
            batch: (num_nodes,) batch assignment
        """
        device = node_features.device
        n_services = self.n_services
        
        if self.edge_mode == 'full':
            # Fully connected graph (let GCN learn to ignore irrelevant edges)
            # Create edges for single graph
            src = torch.arange(n_services, device=device).repeat_interleave(n_services)
            dst = torch.arange(n_services, device=device).repeat(n_services)
            # Remove self-loops
            mask = src != dst
            single_edge_index = torch.stack([src[mask], dst[mask]])
            
            # Replicate for batch
            edge_indices = []
            for b in range(batch_size):
                offset = b * n_services
                edge_indices.append(single_edge_index + offset)
            edge_index = torch.cat(edge_indices, dim=1)
            
        elif self.edge_mode == 'learned':
            # Learn which edges exist based on node features
            # Reshape to (batch, n_services, dim)
            node_feat_reshaped = node_features.view(batch_size, n_services, -1)
            
            # Compute edge scores
            src_emb = self.edge_src_embed(node_feat_reshaped)  # (B, S, H)
            dst_emb = self.edge_dst_embed(node_feat_reshaped)  # (B, S, H)
            
            # Bilinear attention for edge scores
            edge_scores = torch.bmm(src_emb, dst_emb.transpose(1, 2))  # (B, S, S)
            edge_probs = torch.sigmoid(edge_scores - self.edge_threshold)
            
            # Convert to edge_index using top-k or threshold
            edge_indices = []
            for b in range(batch_size):
                probs = edge_probs[b]
                # Take edges above threshold (or top-k)
                # Use straight-through estimator for differentiability
                mask = probs > 0.5
                src, dst = torch.where(mask)
                # Remove self-loops
                valid = src != dst
                src, dst = src[valid], dst[valid]
                
                # If too few edges, add top-k
                if len(src) < n_services:
                    # Add strongest edges
                    flat_probs = probs.view(-1)
                    _, top_idx = torch.topk(flat_probs, min(n_services * 2, len(flat_probs)))
                    extra_src = top_idx // n_services
                    extra_dst = top_idx % n_services
                    valid = extra_src != extra_dst
                    src = torch.cat([src, extra_src[valid][:n_services]])
                    dst = torch.cat([dst, extra_dst[valid][:n_services]])
                
                offset = b * n_services
                edge_indices.append(torch.stack([src + offset, dst + offset]))
            
            edge_index = torch.cat(edge_indices, dim=1)
            
        elif self.edge_mode == 'correlation':
            # Build edges based on feature correlation (no learnable params)
            node_feat_reshaped = node_features.view(batch_size, n_services, -1)
            
            # Normalize for cosine similarity
            norm_feat = F.normalize(node_feat_reshaped, dim=-1)
            corr = torch.bmm(norm_feat, norm_feat.transpose(1, 2))  # (B, S, S)
            
            edge_indices = []
            for b in range(batch_size):
                # Top-k correlations per node (excluding self)
                c = corr[b].clone()
                c.fill_diagonal_(-float('inf'))
                _, top_k = c.topk(min(3, n_services - 1), dim=-1)  # Top 3 neighbors
                
                src = torch.arange(n_services, device=device).unsqueeze(1).expand(-1, top_k.shape[1])
                src = src.reshape(-1)
                dst = top_k.reshape(-1)
                
                offset = b * n_services
                edge_indices.append(torch.stack([src + offset, dst + offset]))
            
            edge_index = torch.cat(edge_indices, dim=1)
        
        # Batch assignment
        batch_assign = torch.arange(batch_size, device=device).repeat_interleave(n_services)
        
        return edge_index, batch_assign
    
    def forward(
        self, 
        traces: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert trace time-series to graph format.
        
        Args:
            traces: (batch, n_services, seq_len, features)
        
        Returns:
            node_features: (batch * n_services, node_feature_dim)
            edge_index: (2, num_edges)
            batch: (batch * n_services,) batch assignment
        """
        batch_size, n_services, seq_len, _ = traces.shape
        
        # Flatten batch and services
        traces_flat = traces.view(batch_size * n_services, seq_len, -1)
        
        # Aggregate temporal features
        node_features = self._aggregate_temporal(traces_flat)
        
        # Build graph structure
        edge_index, batch_assign = self._build_edges(node_features, batch_size)
        
        return node_features, edge_index, batch_assign


class TracesGCNWrapper(nn.Module):
    """
    Full GCN-based traces encoder that matches TCN encoder interface.
    
    Takes: (batch, n_services, seq_len, features)
    Returns: (batch * n_services, embed_dim) - per-service embeddings
    
    Internally:
    1. TraceGraphBuilder: time-series → graph
    2. GCNEncoder: graph → embeddings
    3. Optional: node-level output (no pooling)
    """
    
    def __init__(
        self,
        n_services: int,
        n_trace_features: int = 32,
        hidden_dim: int = 32,
        embed_dim: int = 64,
        num_gcn_layers: int = 2,
        edge_mode: Literal['learned', 'full', 'correlation'] = 'learned',
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_services = n_services
        self.embed_dim = embed_dim
        
        # Graph builder
        node_feature_dim = hidden_dim * 2  # 64 by default
        self.graph_builder = TraceGraphBuilder(
            n_services=n_services,
            n_trace_features=n_trace_features,
            hidden_dim=hidden_dim,
            node_feature_dim=node_feature_dim,
            edge_mode=edge_mode,
            dropout=dropout
        )
        
        # GCN encoder
        if not PYGEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GCN encoder")
        
        self.gcn = GCNEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim * 2,
            embedding_dim=embed_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
            pooling='none'  # We need per-node embeddings
        )
    
    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        """
        Encode traces using GCN.
        
        Args:
            traces: (batch, n_services, seq_len, features) or
                    (batch * n_services, seq_len, features)
        
        Returns:
            (batch * n_services, embed_dim) per-service embeddings
        """
        # Handle both input formats
        if traces.dim() == 3:
            # Already flattened (B*S, T, F) - need to infer batch size
            batch_size = traces.shape[0] // self.n_services
            traces = traces.view(batch_size, self.n_services, traces.shape[1], traces.shape[2])
        
        # Build graph from time-series
        node_features, edge_index, batch_assign = self.graph_builder(traces)
        
        # Apply GCN (no pooling - we want per-node embeddings)
        node_embeddings = self.gcn(node_features, edge_index, batch=None)
        
        return node_embeddings


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for service graphs.

    Features:
    - 2-3 layer GCN (more causes over-smoothing)
    - Hidden dim: 64-128
    - Dropout: 0.3-0.5
    - Memory: 5-10MB

    Reference: PyTorch Geometric v2.3+
    Paper: "Semi-supervised Classification with GCNs" (Kipf & Welling, 2017)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean'  # 'mean', 'max', 'add'
    ):
        super().__init__()

        if not PYGEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: pip install torch-geometric"
            )

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer: hidden_channels -> embedding_dim
        self.convs.append(GCNConv(hidden_channels, embedding_dim))
        self.batch_norms.append(nn.BatchNorm1d(embedding_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode service dependency graph.

        Args:
            x: (num_nodes, in_channels) node features
            edge_index: (2, num_edges) edge connectivity
            batch: (num_nodes,) batch assignment for pooling

        Returns:
            (batch_size, embedding_dim) graph embeddings
            OR (num_nodes, embedding_dim) node embeddings if batch is None
        """
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)

            # Apply activation (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # If batch is provided, pool to graph-level
        if batch is not None:
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                from torch_geometric.nn import global_max_pool
                x = global_max_pool(x, batch)
            elif self.pooling == 'add':
                from torch_geometric.nn import global_add_pool
                x = global_add_pool(x, batch)

        return x

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get node-level embeddings (for interpretability)."""
        return self.forward(x, edge_index, batch=None)


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder (upgrade from GCN).

    Use when:
    - Heterogeneous service types need different attention
    - Dynamic workloads with shifting importance
    - Need interpretable attention visualizations

    Features:
    - Multi-head attention (4-8 heads)
    - Learns edge importance dynamically
    - Interpretable attention weights
    - Memory: 10-20MB

    Reference: PyTorch Geometric v2.3+
    Paper: "Graph Attention Networks" (Veličković et al., 2018)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.4,
        pooling: str = 'mean'
    ):
        super().__init__()

        if not PYGEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: pip install torch-geometric"
            )

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # Build GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: in_channels -> hidden_channels (multi-head)
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=True  # Concatenate heads
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Output layer: single head, no concat
        self.convs.append(
            GATConv(
                hidden_channels * num_heads if num_layers > 1 else in_channels,
                embedding_dim,
                heads=1,
                dropout=dropout,
                concat=False
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(embedding_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Encode service dependency graph with attention.

        Args:
            x: (num_nodes, in_channels) node features
            edge_index: (2, num_edges) edge connectivity
            batch: (num_nodes,) batch assignment
            return_attention_weights: Return attention for visualization

        Returns:
            (batch_size, embedding_dim) graph embeddings
            Optional: (edge_index, attention_weights) if requested
        """
        attention_weights = [] if return_attention_weights else None

        # Apply GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if return_attention_weights:
                x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, attn))
            else:
                x = conv(x, edge_index)

            x = bn(x)

            # Apply activation (except last layer)
            if i < len(self.convs) - 1:
                x = F.elu(x)  # ELU often works better for GAT
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool to graph-level if batch provided
        if batch is not None:
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                from torch_geometric.nn import global_max_pool
                x = global_max_pool(x, batch)

        if return_attention_weights:
            return x, attention_weights
        return x


# Helper functions

def create_trace_encoder(
    encoder_type: str = 'gcn',
    in_channels: int = 8,
    hidden_channels: int = 64,
    embedding_dim: int = 128,
    **kwargs
) -> nn.Module:
    """Factory function to create trace encoder."""
    if encoder_type == 'gcn':
        return GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            **kwargs
        )
    elif encoder_type == 'gat':
        return GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
