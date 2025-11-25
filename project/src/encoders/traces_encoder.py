"""
Trace encoder with GNN on service dependency graphs.

Pipeline:
1. Parse traces → service dependency graph
2. Extract node features (latency, error rate, request count)
3. Extract edge features (call frequency, latency)
4. Apply GCN/GAT for graph encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not installed. GNN encoders unavailable.")


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
