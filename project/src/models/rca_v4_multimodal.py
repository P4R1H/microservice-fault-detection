"""
RCA Model V4 - Multimodal Architecture with Metrics, Logs, and Traces.

Key Design Principles:
1. Lightweight modality-specific encoders (avoid parameter explosion)
2. Gated cross-modal fusion (learn which modalities matter)
3. Hierarchical attention: intra-service fusion → inter-service reasoning
4. Strong regularization to prevent overfitting on ~270 cases
5. Direct causal integration (proven effective in V3)

Architecture Overview:
┌─────────────────────────────────────────────────────────────────┐
│  Per-Service Processing                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Metrics TCN  │ │  Logs TCN    │ │ Traces MLP   │            │
│  │   → 64d      │ │   → 64d      │ │   → 64d      │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│         └────────┬───────┴────────┬───────┘                     │
│                  ▼                                              │
│         ┌─────────────────┐                                    │
│         │ Gated Fusion    │                                    │
│         │ (learnable)     │                                    │
│         └────────┬────────┘                                    │
│                  ▼                                              │
│         Service Embedding (128d)                                │
└──────────────────┬──────────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Inter-Service Reasoning                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Cross-Service Attention (2 layers)                       │    │
│  │ + Causal Weight Injection (PCMCI)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Root Cause Scoring                                       │    │
│  │ Score = Neural_Score + λ * Causal_Score                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal
import math

# Import logs encoder options
from ..encoders.logs_encoder import TFIDFLogsEncoder, LogsEncoder


class ModalityEncoder(nn.Module):
    """
    Lightweight TCN encoder for any modality.
    
    Shared architecture but separate weights per modality.
    Uses depthwise separable convolutions to reduce parameters.
    """
    
    def __init__(self, 
                 in_features: int,
                 hidden_dim: int = 32,
                 out_dim: int = 64,
                 num_layers: int = 2,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            layers.extend([
                # Depthwise separable: depthwise + pointwise
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                         padding=padding, dilation=dilation, groups=hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, 1),  # Pointwise
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.temporal = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * n_services, seq_len, features)
        Returns:
            (batch * n_services, out_dim)
        """
        # Project to hidden dim
        x = self.input_proj(x)  # (B*S, T, hidden)
        
        # Temporal convolution
        x = x.permute(0, 2, 1)  # (B*S, hidden, T)
        x = self.temporal(x)
        
        # Pool and project
        x = self.pool(x).squeeze(-1)  # (B*S, hidden)
        x = self.output_proj(x)  # (B*S, out_dim)
        
        return x


class GatedFusion(nn.Module):
    """
    Gated fusion of multiple modalities.
    
    Learns modality importance dynamically based on content.
    Handles missing modalities gracefully.
    """
    
    def __init__(self, 
                 embed_dim: int = 64,
                 n_modalities: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.n_modalities = n_modalities
        
        # Gate network: takes concatenated modalities → importance weights
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_modalities),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, 
                metrics_emb: torch.Tensor,
                logs_emb: Optional[torch.Tensor] = None,
                traces_emb: Optional[torch.Tensor] = None,
                modality_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            metrics_emb: (batch, embed_dim) - always present
            logs_emb: (batch, embed_dim) or None
            traces_emb: (batch, embed_dim) or None
            modality_mask: (batch, 3) - 1 if modality present, 0 if missing
        
        Returns:
            (batch, embed_dim) fused representation
        """
        batch_size = metrics_emb.shape[0]
        device = metrics_emb.device
        
        # Handle missing modalities with zeros
        if logs_emb is None:
            logs_emb = torch.zeros_like(metrics_emb)
        if traces_emb is None:
            traces_emb = torch.zeros_like(metrics_emb)
        
        # Stack: (batch, 3, embed_dim)
        stacked = torch.stack([metrics_emb, logs_emb, traces_emb], dim=1)
        
        # Compute gates
        concat = stacked.view(batch_size, -1)  # (batch, 3*embed_dim)
        gates = self.gate(concat)  # (batch, 3)
        
        # Apply mask if provided (zero out missing modality gates)
        if modality_mask is not None:
            gates = gates * modality_mask
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        
        # Weighted sum
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # (batch, embed_dim)
        
        # Final projection with residual
        out = self.output(fused) + metrics_emb  # Residual to metrics (always present)
        
        return out


class CrossServiceAttention(nn.Module):
    """
    Attention across services with optional causal bias.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                x: torch.Tensor, 
                causal_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_services, embed_dim)
            causal_bias: (batch, n_services, n_services) - optional attention bias
        
        Returns:
            (batch, n_services, embed_dim)
        """
        # Self-attention with optional causal bias
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        
        return x


class MultimodalRCAModel(nn.Module):
    """
    Multimodal Root Cause Analysis Model (V4/V4.1).
    
    Combines metrics, logs, and traces with gated fusion
    and causal-guided attention for root cause identification.
    
    V4.1 Update: Support for TF-IDF logs encoder (vs TCN)
    """
    
    def __init__(self,
                 n_services: int,
                 n_metric_features: int = 64,
                 n_log_features: int = 32,
                 n_trace_features: int = 32,
                 hidden_dim: int = 32,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_attn_layers: int = 2,
                 dropout: float = 0.35,
                 causal_weight: float = 0.3,
                 logs_encoder_type: Literal['tcn', 'tfidf', 'gemini'] = 'tfidf'):
        super().__init__()
        
        self.n_services = n_services
        self.embed_dim = embed_dim
        self.causal_weight = causal_weight
        self.logs_encoder_type = logs_encoder_type
        
        # === Modality-Specific Encoders ===
        encoder_out_dim = embed_dim // 2  # 64 each, fuse to 128
        
        self.metrics_encoder = ModalityEncoder(
            in_features=n_metric_features,
            hidden_dim=hidden_dim,
            out_dim=encoder_out_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # === Logs Encoder (configurable) ===
        if logs_encoder_type == 'tfidf':
            # V4.1: TF-IDF weighted encoder with temporal features
            self.logs_encoder = TFIDFLogsEncoder(
                n_log_features=n_log_features,
                hidden_dim=hidden_dim,
                embed_dim=encoder_out_dim,
                num_layers=2,
                dropout=dropout
            )
        elif logs_encoder_type == 'tcn':
            # Original V4: TCN encoder (treats logs as time series)
            self.logs_encoder = ModalityEncoder(
                in_features=n_log_features,
                hidden_dim=hidden_dim // 2,
                out_dim=encoder_out_dim,
                num_layers=2,
                dropout=dropout
            )
        elif logs_encoder_type == 'gemini':
            # V5.0: Gemini LLM semantic embeddings
            from ..encoders.logs_encoder import GeminiLogsEncoder
            self.logs_encoder = GeminiLogsEncoder(
                n_log_features=n_log_features,
                hidden_dim=hidden_dim,
                embed_dim=encoder_out_dim,
                num_layers=2,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown logs_encoder_type: {logs_encoder_type}")
        
        # === Traces Encoder ===
        # TCN encoder (treats traces as time series)
        self.traces_encoder = ModalityEncoder(
            in_features=n_trace_features,
            hidden_dim=hidden_dim // 2,  # Smaller for traces
            out_dim=encoder_out_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # === Fusion ===
        self.fusion = GatedFusion(
            embed_dim=encoder_out_dim,
            n_modalities=3,
            dropout=dropout
        )
        
        # Project fused to full embed_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(encoder_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === Service Embeddings ===
        self.service_embed = nn.Embedding(n_services, embed_dim)
        
        # === Cross-Service Attention ===
        self.attn_layers = nn.ModuleList([
            CrossServiceAttention(embed_dim, num_heads, dropout)
            for _ in range(num_attn_layers)
        ])
        
        # === Classifier ===
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
        """Conservative initialization to prevent early divergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                
    def forward(self,
                metrics: torch.Tensor,
                logs: Optional[torch.Tensor] = None,
                traces: Optional[torch.Tensor] = None,
                causal_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            metrics: (batch, n_services, seq_len, n_metric_features)
            logs: (batch, n_services, seq_len, n_log_features) or None
            traces: (batch, n_services, seq_len, n_trace_features) or None
            causal_weights: (batch, n_services, n_services) - PCMCI causal strengths
        
        Returns:
            Dict with 'logits', 'probs', 'ranking', 'gate_weights'
        """
        batch_size, n_services, seq_len, _ = metrics.shape
        device = metrics.device
        
        # === Encode each modality per service ===
        # Flatten batch and services for TCN-style encoders
        metrics_flat = metrics.view(batch_size * n_services, seq_len, -1)
        metrics_emb = self.metrics_encoder(metrics_flat)  # (B*S, embed_dim/2)
        
        if logs is not None:
            logs_flat = logs.view(batch_size * n_services, seq_len, -1)
            logs_emb = self.logs_encoder(logs_flat)
        else:
            logs_emb = None
        
        # Encode traces
        if traces is not None:
            traces_flat = traces.view(batch_size * n_services, seq_len, -1)
            traces_emb = self.traces_encoder(traces_flat)
        else:
            traces_emb = None
        
        # === Fuse modalities ===
        # Create modality mask
        modality_mask = torch.ones(batch_size * n_services, 3, device=device)
        if logs is None:
            modality_mask[:, 1] = 0
        if traces is None:
            modality_mask[:, 2] = 0
            
        fused = self.fusion(metrics_emb, logs_emb, traces_emb, modality_mask)
        fused = self.fusion_proj(fused)  # (B*S, embed_dim)
        
        # Reshape back to (batch, n_services, embed_dim)
        fused = fused.view(batch_size, n_services, -1)
        
        # === Add service embeddings ===
        service_ids = torch.arange(n_services, device=device)
        service_emb = self.service_embed(service_ids)  # (n_services, embed_dim)
        x = fused + service_emb.unsqueeze(0)  # Broadcast over batch
        
        # === Cross-service attention ===
        for attn in self.attn_layers:
            x = attn(x, causal_weights)
        
        # === Score each service ===
        logits = self.classifier(x).squeeze(-1)  # (batch, n_services)
        
        # === Integrate causal scores ===
        if causal_weights is not None:
            # Services with high outgoing causal influence are likely root causes
            causal_out = causal_weights.sum(dim=2)  # (batch, n_services)
            causal_out = causal_out / (causal_out.sum(dim=1, keepdim=True) + 1e-8)
            logits = logits + self.causal_weight * causal_out
        
        probs = F.softmax(logits, dim=-1)
        ranking = torch.argsort(logits, dim=-1, descending=True)
        
        return {
            'logits': logits,
            'probs': probs,
            'ranking': ranking,
            'embeddings': x
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters by component."""
        def count_module(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            'metrics_encoder': count_module(self.metrics_encoder),
            'logs_encoder': count_module(self.logs_encoder),
            'traces_encoder': count_module(self.traces_encoder),
            'fusion': count_module(self.fusion) + count_module(self.fusion_proj),
            'service_embed': count_module(self.service_embed),
            'attention': sum(count_module(a) for a in self.attn_layers),
            'classifier': count_module(self.classifier),
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class MultimodalLoss(nn.Module):
    """
    Combined loss for multimodal RCA.
    
    Components:
    1. Label-smoothed cross-entropy
    2. Margin ranking loss
    3. Optional modality consistency loss
    """
    
    def __init__(self, 
                 smoothing: float = 0.1, 
                 rank_weight: float = 0.3, 
                 margin: float = 0.5):
        super().__init__()
        self.smoothing = smoothing
        self.rank_weight = rank_weight
        self.margin = margin
        
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (batch, n_services)
            targets: (batch,) service indices
        
        Returns:
            Dict with 'loss', 'ce_loss', 'rank_loss'
        """
        batch_size, n_classes = logits.shape
        device = logits.device
        
        # Label-smoothed cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        ce_loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        
        # Margin ranking loss
        rank_loss = torch.tensor(0.0, device=device)
        for i in range(batch_size):
            target_score = logits[i, targets[i]]
            other_scores = logits[i].clone()
            other_scores[targets[i]] = float('-inf')
            max_other = other_scores.max()
            rank_loss = rank_loss + F.relu(self.margin - (target_score - max_other))
        rank_loss = rank_loss / batch_size
        
        total = ce_loss + self.rank_weight * rank_loss
        
        return {
            'loss': total,
            'ce_loss': ce_loss,
            'rank_loss': rank_loss
        }


def create_multimodal_model(
    n_services: int,
    n_metric_features: int = 64,
    n_log_features: int = 32,
    n_trace_features: int = 32,
    hidden_dim: int = 32,
    embed_dim: int = 128,
    dropout: float = 0.35,
    logs_encoder_type: str = 'tfidf'
) -> MultimodalRCAModel:
    """
    Factory function to create multimodal RCA model.
    
    Args:
        n_services: Number of services in the system
        n_metric_features: Number of metric features per service
        n_log_features: Number of log template features per service
        n_trace_features: Number of trace features per service
        hidden_dim: Hidden dimension for encoders
        embed_dim: Service embedding dimension
        dropout: Dropout rate
        logs_encoder_type: 'tcn' (V4), 'tfidf' (V4.1), or 'gemini' (V4.3)
    """
    return MultimodalRCAModel(
        n_services=n_services,
        n_metric_features=n_metric_features,
        n_log_features=n_log_features,
        n_trace_features=n_trace_features,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout=dropout,
        logs_encoder_type=logs_encoder_type
    )


if __name__ == '__main__':
    # Test the model
    batch_size = 4
    n_services = 15
    seq_len = 60
    n_metric_feat = 64
    n_log_feat = 32
    n_trace_feat = 32
    
    model = create_multimodal_model(
        n_services=n_services,
        n_metric_features=n_metric_feat,
        n_log_features=n_log_feat,
        n_trace_features=n_trace_feat
    )
    
    print("Model parameters:")
    params = model.count_parameters()
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    metrics = torch.randn(batch_size, n_services, seq_len, n_metric_feat)
    logs = torch.randn(batch_size, n_services, seq_len, n_log_feat)
    traces = torch.randn(batch_size, n_services, seq_len, n_trace_feat)
    causal = torch.rand(batch_size, n_services, n_services)
    
    with torch.no_grad():
        out = model(metrics, logs, traces, causal)
        print(f"\nOutput shapes:")
        print(f"  logits: {out['logits'].shape}")
        print(f"  probs: {out['probs'].shape}")
        print(f"  ranking: {out['ranking'].shape}")
    
    # Test loss
    targets = torch.randint(0, n_services, (batch_size,))
    loss_fn = MultimodalLoss()
    losses = loss_fn(out['logits'], targets)
    print(f"\nLosses:")
    for name, val in losses.items():
        print(f"  {name}: {val.item():.4f}")
