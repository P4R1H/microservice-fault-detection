"""
Logs encoder with Drain3 parsing and embeddings.

V4.1 Update: Proper TF-IDF based logs encoding (replaces placeholder)
V5.0 Update: Gemini LLM embeddings for semantic understanding

Pipeline options:
1. TF-IDF: Template counts → TF-IDF weighted → Temporal encoding → 64d
2. Gemini: Template counts → LLM semantic embeddings → Projection → 64d
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple


class TFIDFLogsEncoder(nn.Module):
    """
    TF-IDF weighted logs encoder for log template time series.
    
    This encoder properly processes log template counts by:
    1. Learning TF-IDF-like weights for each template
    2. Applying temporal convolution to capture patterns
    3. Projecting to embedding dimension
    
    Input: (batch, n_services, seq_len, n_log_features) - template counts per timestep
    Output: (batch * n_services, embed_dim)
    """
    
    def __init__(self,
                 n_log_features: int = 32,
                 hidden_dim: int = 48,
                 embed_dim: int = 64,
                 num_layers: int = 2,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.n_log_features = n_log_features
        self.embed_dim = embed_dim
        
        # Learnable IDF-like weights for each template
        # Templates that appear in few cases should have higher weight
        self.template_weights = nn.Parameter(torch.ones(n_log_features))
        
        # Input projection with template weighting
        self.input_proj = nn.Sequential(
            nn.Linear(n_log_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal convolution layers (TCN-style)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            layers.extend([
                # Depthwise separable convolution
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                         padding=padding, dilation=dilation, groups=min(hidden_dim, 8)),
                nn.Conv1d(hidden_dim, hidden_dim, 1),  # Pointwise
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.temporal = nn.Sequential(*layers)
        
        # Pooling and output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Error pattern detector (high-weight templates often indicate errors)
        self.error_detector = nn.Sequential(
            nn.Linear(n_log_features, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * n_services, seq_len, n_log_features) - template counts
            
        Returns:
            (batch * n_services, embed_dim)
        """
        # Apply learnable template weights (IDF-like)
        # Softmax to normalize weights, then scale
        weights = torch.softmax(self.template_weights, dim=0)
        x_weighted = x * weights.unsqueeze(0).unsqueeze(0)  # (B*S, T, F)
        
        # Extract error pattern features from raw counts
        # Use mean counts across time to detect persistent patterns
        mean_counts = x.mean(dim=1)  # (B*S, F)
        error_features = self.error_detector(mean_counts)  # (B*S, embed_dim)
        
        # Temporal encoding
        h = self.input_proj(x_weighted)  # (B*S, T, hidden)
        h = h.permute(0, 2, 1)  # (B*S, hidden, T)
        h = self.temporal(h)  # (B*S, hidden, T)
        h = self.pool(h).squeeze(-1)  # (B*S, hidden)
        
        # Output projection
        temporal_features = self.output_proj(h)  # (B*S, embed_dim)
        
        # Combine temporal and error pattern features
        output = temporal_features + 0.3 * error_features
        
        return output


class FallbackLogsEncoder(nn.Module):
    """
    Simple fallback logs encoder using learned embeddings.
    
    DEPRECATED: Use TFIDFLogsEncoder instead.
    Kept for backward compatibility with old checkpoints.
    """

    def __init__(self, embedding_dim: int = 64, n_log_features: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_log_features = n_log_features
        
        # Simple MLP on mean log counts
        self.encoder = nn.Sequential(
            nn.Linear(n_log_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * n_services, seq_len, n_log_features) or None
            
        Returns:
            (batch * n_services, embedding_dim)
        """
        if x is None or x.numel() == 0:
            # Return zeros if no input
            return torch.zeros(1, self.embedding_dim, device=next(self.parameters()).device)
        
        # Take mean across time
        x_mean = x.mean(dim=1)  # (B*S, n_log_features)
        return self.encoder(x_mean)


class LogsEncoder(nn.Module):
    """
    Main logs encoder interface.
    
    Supports multiple backends:
    - 'tfidf': TFIDFLogsEncoder (default for V4.1)
    - 'gemini': GeminiLogsEncoder (V5.0, requires API)
    - 'fallback': Simple learned embedding (deprecated)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_log_features: int = 32,
        encoder_type: str = 'tfidf',
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type
        
        if encoder_type == 'tfidf':
            self.encoder = TFIDFLogsEncoder(
                n_log_features=n_log_features,
                hidden_dim=hidden_dim,
                embed_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif encoder_type == 'fallback':
            self.encoder = FallbackLogsEncoder(
                embedding_dim=embedding_dim,
                n_log_features=n_log_features
            )
        elif encoder_type == 'gemini':
            # Will be implemented in Stage 3
            raise NotImplementedError("Gemini encoder - implement in Stage 3")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode log data.

        Args:
            x: (batch * n_services, seq_len, n_log_features) template counts

        Returns:
            (batch * n_services, embedding_dim)
        """
        return self.encoder(x)


# Factory function
def create_logs_encoder(
    embedding_dim: int = 64,
    n_log_features: int = 32,
    encoder_type: str = 'tfidf',
    **kwargs
) -> nn.Module:
    """
    Create a logs encoder.

    Args:
        embedding_dim: Output embedding dimension
        n_log_features: Number of log template features
        encoder_type: 'tfidf', 'gemini', or 'fallback'
        **kwargs: Additional arguments

    Returns:
        LogsEncoder instance
    """
    return LogsEncoder(
        embedding_dim=embedding_dim,
        n_log_features=n_log_features,
        encoder_type=encoder_type,
        **kwargs
    )
