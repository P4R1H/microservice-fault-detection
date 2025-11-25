"""
RCA Model with cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..encoders.tcn import TCNEncoder


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention for service interactions."""
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                causal_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_services, embed_dim)
            causal_weights: (batch, n_services, n_services) optional attention bias
        Returns:
            (batch, n_services, embed_dim)
        """
        # Self-attention across services
        attn_out, attn_weights = self.attention(x, x, x)
        
        # Apply causal weights as multiplicative bias if provided
        if causal_weights is not None:
            # Use causal weights to modulate attention output
            # causal_weights: (batch, n_services, n_services)
            # We weight each service's representation by its causal importance
            causal_importance = causal_weights.sum(dim=1)  # (batch, n_services) - incoming causal strength
            causal_importance = causal_importance / (causal_importance.sum(dim=1, keepdim=True) + 1e-8)
            attn_out = attn_out * (1 + causal_importance.unsqueeze(-1))  # Boost causally important services
        
        x = self.norm(x + self.dropout(attn_out))
        return x


class RCAModel(nn.Module):
    """
    Complete RCA model with TCN encoder and cross-attention.
    
    Architecture:
    1. TCN encodes each service's time series
    2. Cross-attention models service interactions
    3. Optional causal weights from PCMCI
    4. Classification head predicts root cause
    """
    
    def __init__(self,
                 n_services: int,
                 n_features: int = 64,
                 tcn_hidden: int = 128,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_attn_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.n_services = n_services
        self.embed_dim = embed_dim
        
        # Service-level TCN encoder
        self.tcn = TCNEncoder(
            in_channels=n_features,
            hidden_channels=tcn_hidden,
            out_channels=embed_dim,
            num_layers=4,
            dropout=dropout
        )
        
        # Learnable service embeddings
        self.service_embed = nn.Embedding(n_services, embed_dim)
        
        # Cross-attention layers
        self.attention_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_attn_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, 
                metrics: torch.Tensor,
                causal_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            metrics: (batch, n_services, seq_len, n_features)
            causal_weights: (batch, n_services, n_services) from PCMCI
        Returns:
            Dict with logits, probs, ranking
        """
        batch_size, n_services, seq_len, n_features = metrics.shape
        
        # Encode each service's time series
        # Reshape: (batch * n_services, seq_len, n_features)
        metrics_flat = metrics.view(batch_size * n_services, seq_len, n_features)
        service_encodings = self.tcn(metrics_flat)  # (batch * n_services, embed_dim)
        service_encodings = service_encodings.view(batch_size, n_services, -1)
        
        # Add learnable service embeddings
        service_ids = torch.arange(n_services, device=metrics.device)
        service_emb = self.service_embed(service_ids)  # (n_services, embed_dim)
        service_encodings = service_encodings + service_emb.unsqueeze(0)
        
        # Cross-attention layers
        x = service_encodings
        for attn_layer in self.attention_layers:
            x = attn_layer(x, causal_weights)
        
        # Classification: score each service
        scores = self.classifier(x).squeeze(-1)  # (batch, n_services)
        
        # Softmax probabilities
        probs = F.softmax(scores, dim=-1)
        
        # Ranking (descending by score)
        ranking = torch.argsort(scores, dim=-1, descending=True)
        
        return {
            'logits': scores,
            'probs': probs,
            'ranking': ranking
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
