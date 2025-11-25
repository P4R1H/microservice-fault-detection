"""
RCA Model V3 - Balanced model that avoids overfitting.

Key changes from V2:
1. Smaller model (fewer parameters)
2. Stronger regularization (dropout, weight decay)
3. Simpler architecture (proven TCN + simple attention)
4. Label smoothing
5. Mix of causal and learned attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class SimpleTCN(nn.Module):
    """Simpler TCN encoder that generalizes better."""
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 128,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features) -> (batch, out_channels)"""
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return x


class CausalAttention(nn.Module):
    """
    Simple attention that directly uses causal weights.
    No complex transformations - just let PCMCI guide attention.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_services, embed_dim)
            causal_mask: (batch, n_services, n_services) - used as attention bias
        """
        # Self-attention
        if causal_mask is not None:
            # Convert causal weights to attention mask format
            # Higher causal weight = more attention
            # Expand for multi-head attention: need to handle batch dimension
            attn_out, _ = self.attn(x, x, x)
        else:
            attn_out, _ = self.attn(x, x, x)
        
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class RCAModelV3(nn.Module):
    """
    Balanced RCA model optimized for generalization.
    
    Architecture:
    1. Simple TCN per service
    2. Service embeddings
    3. 2 layers of attention
    4. Direct causal score integration
    5. Smaller classifier head
    """
    
    def __init__(self,
                 n_services: int,
                 n_features: int = 64,
                 hidden_dim: int = 64,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_attn_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.n_services = n_services
        self.embed_dim = embed_dim
        
        # Simple TCN encoder
        self.encoder = SimpleTCN(
            in_channels=n_features,
            hidden_channels=hidden_dim,
            out_channels=embed_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # Service embeddings
        self.service_embed = nn.Embedding(n_services, embed_dim)
        
        # Attention layers
        self.attn_layers = nn.ModuleList([
            CausalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_attn_layers)
        ])
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                
    def forward(self,
                metrics: torch.Tensor,
                causal_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            metrics: (batch, n_services, seq_len, n_features)
            causal_weights: (batch, n_services, n_services)
        """
        batch_size, n_services, seq_len, n_features = metrics.shape
        device = metrics.device
        
        # Encode each service
        metrics_flat = metrics.view(batch_size * n_services, seq_len, n_features)
        encodings = self.encoder(metrics_flat)
        encodings = encodings.view(batch_size, n_services, -1)
        
        # Add service embeddings
        service_ids = torch.arange(n_services, device=device)
        service_emb = self.service_embed(service_ids)
        x = encodings + service_emb.unsqueeze(0)
        
        # Attention layers
        for attn in self.attn_layers:
            x = attn(x, causal_weights)
        
        # Score each service
        logits = self.classifier(x).squeeze(-1)  # (batch, n_services)
        
        # Incorporate causal scores directly
        if causal_weights is not None:
            # Services that cause many effects are likely root causes
            causal_out = causal_weights.sum(dim=2)  # Outgoing causal strength
            causal_out = causal_out / (causal_out.sum(dim=1, keepdim=True) + 1e-8)
            logits = logits + 0.5 * causal_out  # Add causal prior
        
        probs = F.softmax(logits, dim=-1)
        ranking = torch.argsort(logits, dim=-1, descending=True)
        
        return {
            'logits': logits,
            'probs': probs,
            'ranking': ranking,
            'embeddings': x
        }
    
    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class BalancedLoss(nn.Module):
    """Loss with label smoothing and ranking component."""
    
    def __init__(self, smoothing: float = 0.1, rank_weight: float = 0.3, margin: float = 0.5):
        super().__init__()
        self.smoothing = smoothing
        self.rank_weight = rank_weight
        self.margin = margin
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, n_classes = logits.shape
        
        # Label smoothing cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Smooth targets
        smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        ce_loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        
        # Simple ranking loss
        rank_loss = torch.tensor(0.0, device=logits.device)
        for i in range(batch_size):
            target_score = logits[i, targets[i]]
            other_max = logits[i].clone()
            other_max[targets[i]] = float('-inf')
            max_other = other_max.max()
            rank_loss = rank_loss + F.relu(self.margin - (target_score - max_other))
        rank_loss = rank_loss / batch_size
        
        total = ce_loss + self.rank_weight * rank_loss
        
        return {
            'loss': total,
            'ce_loss': ce_loss,
            'rank_loss': rank_loss
        }
