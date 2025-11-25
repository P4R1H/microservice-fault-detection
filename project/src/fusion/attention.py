"""
Cross-modal attention mechanisms for multimodal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention for fusing service representations.
    
    This attention mechanism allows services to attend to each other,
    capturing dependencies and interactions in the microservice graph.
    """
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 use_causal_bias: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_causal_bias = use_causal_bias
        
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                causal_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional causal attention bias.
        
        Args:
            x: Service embeddings (batch, n_services, embed_dim)
            causal_weights: Causal adjacency matrix from PCMCI (batch, n_services, n_services)
            
        Returns:
            Updated embeddings (batch, n_services, embed_dim)
        """
        # Self-attention across services
        attn_out, attn_weights = self.attention(x, x, x)
        
        # Apply causal weights as multiplicative modulation
        if causal_weights is not None and self.use_causal_bias:
            # Compute importance from causal weights
            causal_importance = self._compute_causal_importance(causal_weights)
            attn_out = attn_out * (1 + causal_importance.unsqueeze(-1))
        
        # Residual connection + normalization
        x = self.norm(x + self.dropout(attn_out))
        
        # Feed-forward + residual
        x = self.ffn_norm(x + self.ffn(x))
        
        return x
    
    def _compute_causal_importance(self, 
                                    causal_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute service importance from causal weights.
        
        Services that have many outgoing causal relationships (cause many effects)
        are considered more important as potential root causes.
        
        Args:
            causal_weights: (batch, n_services, n_services)
            
        Returns:
            importance: (batch, n_services) normalized importance scores
        """
        # Sum outgoing causal strength for each service
        outgoing = causal_weights.sum(dim=2)  # (batch, n_services)
        
        # Normalize
        importance = outgoing / (outgoing.sum(dim=1, keepdim=True) + 1e-8)
        
        return importance


class MultiHeadCrossAttention(nn.Module):
    """
    Enhanced cross-attention with separate query, key, value projections
    and explicit attention to causal structure.
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Causal weight projection
        self.causal_proj = nn.Linear(1, num_heads, bias=False)
        
    def forward(self,
                x: torch.Tensor,
                causal_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, n_services, embed_dim)
            causal_weights: (batch, n_services, n_services)
            
        Returns:
            output: (batch, n_services, embed_dim)
            attention: (batch, num_heads, n_services, n_services)
        """
        batch_size, n_services, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, n_services, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_services, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_services, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Add causal bias if provided
        if causal_weights is not None:
            # Project causal weights to per-head biases
            causal_bias = self.causal_proj(causal_weights.unsqueeze(-1))  # (batch, n, n, num_heads)
            causal_bias = causal_bias.permute(0, 3, 1, 2)  # (batch, num_heads, n, n)
            attn_scores = attn_scores + causal_bias
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, n_services, self.embed_dim)
        out = self.out_proj(out)
        
        # Residual + norm
        out = self.norm(x + self.dropout(out))
        
        return out, attn_weights


class CausalGraphAttention(nn.Module):
    """
    Graph attention that explicitly uses the causal graph structure.
    
    This attention mechanism treats the causal weights as a weighted
    adjacency matrix for message passing.
    """
    
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Message and update networks
        self.message_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.update_net = nn.GRU(embed_dim, embed_dim, batch_first=True)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                causal_weights: torch.Tensor) -> torch.Tensor:
        """
        Graph attention using causal structure.
        
        Args:
            x: (batch, n_services, embed_dim)
            causal_weights: (batch, n_services, n_services) - must be provided
            
        Returns:
            (batch, n_services, embed_dim)
        """
        batch_size, n_services, _ = x.shape
        
        # Expand for pairwise combinations
        x_i = x.unsqueeze(2).expand(-1, -1, n_services, -1)  # (batch, n, n, embed)
        x_j = x.unsqueeze(1).expand(-1, n_services, -1, -1)  # (batch, n, n, embed)
        
        # Compute messages: concat source and target features
        messages = torch.cat([x_i, x_j], dim=-1)  # (batch, n, n, embed*2)
        messages = self.message_net(messages)  # (batch, n, n, embed)
        
        # Weight messages by causal strength
        weighted_messages = messages * causal_weights.unsqueeze(-1)
        
        # Aggregate incoming messages
        aggregated = weighted_messages.sum(dim=2)  # (batch, n, embed)
        
        # Update node representations
        aggregated = aggregated.unsqueeze(1)  # (batch, 1, n, embed) - treat n as seq
        x_flat = x.unsqueeze(1)  # (batch, 1, n, embed)
        
        # GRU update
        batch_n = batch_size * n_services
        aggregated_flat = aggregated.view(batch_n, 1, self.embed_dim)
        h0 = x.view(batch_n, self.embed_dim).unsqueeze(0)
        
        _, h_new = self.update_net(aggregated_flat, h0)
        updated = h_new.squeeze(0).view(batch_size, n_services, self.embed_dim)
        
        # Residual + norm
        out = self.norm(x + self.dropout(updated))
        
        return out
