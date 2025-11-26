"""
Trace encoder using TCN (Temporal Convolutional Network).

Pipeline:
1. Parse traces → per-service feature time series
2. Apply TCN for temporal pattern extraction
3. Output per-service embeddings for fusion

Note: GCN/GAT graph-based encoders were explored in V4.2 but abandoned
      as they showed worse performance than TCN on this dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class TracesTCNEncoder(nn.Module):
    """
    TCN-based trace encoder.
    
    Takes trace time-series (latency, error rate, call counts) and produces
    per-service embeddings using temporal convolutions.
    
    Input: (batch * n_services, seq_len, n_features)
    Output: (batch * n_services, embed_dim)
    """
    
    def __init__(
        self,
        n_trace_features: int = 32,
        hidden_dim: int = 32,
        embed_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Linear(n_trace_features, hidden_dim)
        
        # TCN layers with dilated convolutions
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim, hidden_dim,
                        kernel_size=3, padding=dilation, dilation=dilation
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Temporal attention pooling
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
    
    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        """
        Encode trace time-series.
        
        Args:
            traces: (batch * n_services, seq_len, n_features)
        
        Returns:
            (batch * n_services, embed_dim) per-service embeddings
        """
        # Project input
        x = self.input_proj(traces)  # (B*S, T, H)
        
        # Apply TCN
        x = x.permute(0, 2, 1)  # (B*S, H, T)
        for tcn in self.tcn_layers:
            residual = x
            x = tcn(x)
            if x.shape == residual.shape:
                x = x + residual  # Residual connection
        x = x.permute(0, 2, 1)  # (B*S, T, H)
        
        # Attention-weighted pooling over time
        attn_scores = self.time_attention(x)  # (B*S, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        x = (x * attn_weights).sum(dim=1)  # (B*S, H)
        
        # Output projection
        x = self.output_proj(x)  # (B*S, embed_dim)
        
        return x

