"""
RCA Model V2 - Improved architecture for root cause analysis.

Key improvements over V1:
1. Multi-scale TCN encoding with attention fusion
2. Causal-aware graph attention (PCMCI as explicit edge weights)
3. Contrastive + ranking loss for better discrimination
4. Anomaly detection branch for confidence estimation
5. Feature-level and service-level attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class MultiScaleTCN(nn.Module):
    """
    Multi-scale TCN that captures patterns at different temporal resolutions.
    Uses parallel branches with different kernel sizes.
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 256,
                 num_scales: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.scales = nn.ModuleList()
        kernel_sizes = [3, 5, 7][:num_scales]
        
        for ks in kernel_sizes:
            scale_branch = nn.Sequential(
                # First conv
                nn.Conv1d(in_channels, hidden_channels, ks, padding=ks//2),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                # Second conv with dilation
                nn.Conv1d(hidden_channels, hidden_channels, ks, padding=(ks//2)*2, dilation=2),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                # Third conv with more dilation
                nn.Conv1d(hidden_channels, hidden_channels, ks, padding=(ks//2)*4, dilation=4),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
            )
            self.scales.append(scale_branch)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels * num_scales, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
        # Temporal attention for weighted pooling
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(out_channels, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 1, 1)
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, out_channels)
        """
        # (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Multi-scale encoding
        scale_outputs = []
        for scale in self.scales:
            out = scale(x)
            scale_outputs.append(out)
        
        # Concatenate scales
        multi_scale = torch.cat(scale_outputs, dim=1)
        
        # Fuse scales
        fused = self.fusion(multi_scale)  # (batch, out_channels, seq_len)
        
        # Temporal attention pooling
        attn_weights = self.temporal_attn(fused)  # (batch, 1, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled = (fused * attn_weights).sum(dim=-1)  # (batch, out_channels)
        
        return pooled


class FeatureAttention(nn.Module):
    """
    Attention over input features to focus on anomalous metrics.
    """
    
    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, seq_len, features) with feature attention applied
        """
        # Global feature representation
        x_mean = x.mean(dim=1)  # (batch, features)
        attn = self.attn(x_mean)  # (batch, features)
        return x * attn.unsqueeze(1)


class CausalGraphAttention(nn.Module):
    """
    Graph attention that explicitly uses PCMCI causal weights as edge weights.
    
    Unlike standard attention, this directly incorporates discovered causal
    relationships into message passing.
    """
    
    def __init__(self, 
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 causal_weight: float = 0.5):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal_weight = causal_weight
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable causal weight transformation
        self.causal_transform = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward
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
        Args:
            x: (batch, n_services, embed_dim)
            causal_weights: (batch, n_services, n_services) from PCMCI
        Returns:
            (batch, n_services, embed_dim)
        """
        batch_size, n_services, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, n_services, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, n_services, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, n_services, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Incorporate causal weights
        if causal_weights is not None:
            # Transform causal weights to per-head bias
            causal_bias = self.causal_transform(causal_weights.unsqueeze(-1))  # (batch, n, n, heads)
            causal_bias = causal_bias.permute(0, 3, 1, 2)  # (batch, heads, n, n)
            
            # Blend learned attention with causal structure
            # Higher causal_weight = more influence from PCMCI
            attn_scores = (1 - self.causal_weight) * attn_scores + self.causal_weight * causal_bias * 5.0
        
        # Softmax + dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply to values
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_services, self.embed_dim)
        out = self.out_proj(out)
        
        # Residual + norm
        x = self.norm(x + self.dropout(out))
        
        # FFN + residual
        x = self.ffn_norm(x + self.ffn(x))
        
        return x


class AnomalyScorer(nn.Module):
    """
    Branch that scores how anomalous each service's metrics are.
    This provides an additional signal beyond learned features.
    """
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_services, embed_dim)
        Returns:
            (batch, n_services) anomaly scores
        """
        return self.scorer(x).squeeze(-1)


class RCAModelV2(nn.Module):
    """
    Improved RCA model with:
    1. Multi-scale temporal encoding
    2. Feature attention
    3. Causal graph attention
    4. Anomaly scoring branch
    5. Ranking-aware output
    """
    
    def __init__(self,
                 n_services: int,
                 n_features: int = 64,
                 hidden_dim: int = 128,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_attn_layers: int = 4,
                 dropout: float = 0.2,
                 causal_weight: float = 0.3):
        super().__init__()
        
        self.n_services = n_services
        self.embed_dim = embed_dim
        
        # Feature attention (pre-encoding)
        self.feature_attn = FeatureAttention(n_features, hidden_dim // 2)
        
        # Multi-scale TCN encoder
        self.encoder = MultiScaleTCN(
            in_channels=n_features,
            hidden_channels=hidden_dim,
            out_channels=embed_dim,
            num_scales=3,
            dropout=dropout
        )
        
        # Learnable service embeddings
        self.service_embed = nn.Embedding(n_services, embed_dim)
        
        # Service position encoding (different from identity)
        self.position_embed = nn.Parameter(torch.randn(1, n_services, embed_dim) * 0.02)
        
        # Causal graph attention layers
        self.attention_layers = nn.ModuleList([
            CausalGraphAttention(embed_dim, num_heads, dropout, causal_weight)
            for _ in range(num_attn_layers)
        ])
        
        # Anomaly scoring branch
        self.anomaly_scorer = AnomalyScorer(embed_dim)
        
        # Global context aggregation
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Final classifier combining multiple signals
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),  # service_repr + global + anomaly
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Causal score head (directly from causal weights)
        self.causal_scorer = nn.Sequential(
            nn.Linear(n_services, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
    def forward(self,
                metrics: torch.Tensor,
                causal_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            metrics: (batch, n_services, seq_len, n_features)
            causal_weights: (batch, n_services, n_services)
        Returns:
            Dict with logits, probs, ranking, anomaly_scores, causal_scores
        """
        batch_size, n_services, seq_len, n_features = metrics.shape
        device = metrics.device
        
        # Feature attention per service
        metrics_flat = metrics.view(batch_size * n_services, seq_len, n_features)
        metrics_attended = self.feature_attn(metrics_flat)
        
        # Encode each service
        service_encodings = self.encoder(metrics_attended)  # (batch*n, embed_dim)
        service_encodings = service_encodings.view(batch_size, n_services, -1)
        
        # Add service embeddings and positional encoding
        service_ids = torch.arange(n_services, device=device)
        service_emb = self.service_embed(service_ids)
        service_encodings = service_encodings + service_emb.unsqueeze(0) + self.position_embed
        
        # Causal graph attention layers
        x = service_encodings
        for attn_layer in self.attention_layers:
            x = attn_layer(x, causal_weights)
        
        # Anomaly scores
        anomaly_scores = self.anomaly_scorer(x)  # (batch, n_services)
        
        # Global context
        global_query = self.global_query.expand(batch_size, -1, -1)
        global_ctx, _ = self.global_attn(global_query, x, x)  # (batch, 1, embed_dim)
        global_ctx = global_ctx.expand(-1, n_services, -1)
        
        # Causal score (how much each service causes others)
        causal_scores = torch.zeros(batch_size, n_services, device=device)
        if causal_weights is not None:
            # Outgoing causal influence
            causal_out = causal_weights.sum(dim=2)  # (batch, n_services)
            causal_scores = self.causal_scorer(causal_weights).squeeze(-1)  # Use full matrix
        
        # Combine all signals for final prediction
        combined = torch.cat([
            x,                              # Service representations
            global_ctx,                     # Global context
            anomaly_scores.unsqueeze(-1)    # Anomaly scores
        ], dim=-1)  # (batch, n_services, embed_dim*2 + 1)
        
        # Final scores
        logits = self.classifier(combined).squeeze(-1)  # (batch, n_services)
        
        # Blend with causal scores
        if causal_weights is not None:
            logits = logits + 0.2 * causal_scores
        
        # Probabilities and ranking
        probs = F.softmax(logits, dim=-1)
        ranking = torch.argsort(logits, dim=-1, descending=True)
        
        return {
            'logits': logits,
            'probs': probs,
            'ranking': ranking,
            'anomaly_scores': anomaly_scores,
            'causal_scores': causal_scores,
            'embeddings': x
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class RankingLoss(nn.Module):
    """
    Combined loss for RCA:
    1. Cross-entropy for classification
    2. Pairwise ranking loss (root cause should rank above others)
    3. Contrastive loss (root cause embedding should be distinct)
    """
    
    def __init__(self, 
                 ce_weight: float = 1.0,
                 rank_weight: float = 0.5,
                 margin: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.rank_weight = rank_weight
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (batch, n_services)
            targets: (batch,) indices
            embeddings: (batch, n_services, embed_dim) optional
        Returns:
            Dict with total loss and components
        """
        batch_size, n_services = logits.shape
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Pairwise ranking loss
        # For each sample, the root cause should score higher than all others
        rank_loss = torch.tensor(0.0, device=logits.device)
        for i in range(batch_size):
            target_idx = int(targets[i].item())
            target_score = logits[i, target_idx]
            
            # Scores of non-root-cause services
            mask = torch.ones(n_services, dtype=torch.bool, device=logits.device)
            mask[target_idx] = False
            other_scores = logits[i, mask]
            
            # Margin ranking: target should be higher than others by margin
            violations = F.relu(self.margin - (target_score - other_scores))
            rank_loss = rank_loss + violations.mean()
        
        rank_loss = rank_loss / batch_size
        
        # Total loss
        total_loss = self.ce_weight * ce_loss + self.rank_weight * rank_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'rank_loss': rank_loss
        }


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
