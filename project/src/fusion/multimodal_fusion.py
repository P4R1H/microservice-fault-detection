"""
Multimodal fusion with cross-modal attention.

Architecture:
1. Separate encoders per modality (Chronos/TCN, Logs, GCN)
2. Cross-modal attention for dynamic importance weighting
3. Combined representation for RCA

Reference:
- "FAMOS" (ICSE 2025): Gaussian-attention multimodal fusion
- "MULAN" (WWW 2024): Log-tailored LM with contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Learns to weight different modalities based on:
    - Inter-modality relationships
    - Task-specific importance
    - Dynamic contextual relevance

    Features:
    - Multi-head attention (8 heads typical)
    - Scaled dot-product attention
    - Dropout for regularization
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-modal attention.

        Args:
            query: (batch, seq_len, embed_dim) query modality
            key: (batch, seq_len, embed_dim) key modality
            value: (batch, seq_len, embed_dim) value modality
            return_attention: Return attention weights for visualization

        Returns:
            (batch, seq_len, embed_dim) attended output
            Optional: attention weights
        """
        attn_output, attn_weights = self.multihead_attn(
            query, key, value,
            need_weights=return_attention
        )

        if return_attention:
            return attn_output, attn_weights
        return attn_output, None


class MultimodalFusion(nn.Module):
    """
    Complete multimodal fusion pipeline.

    Architecture:
    ```
    Metrics → Encoder → 256-dim ─┐
                                   │
    Logs → Encoder → 256-dim ──────┼→ Cross-Attention → 512-dim → RCA Head
                                   │
    Traces → Encoder → 128-dim ───┘
    ```

    Fusion Strategies:
    - Early: Concatenate raw features (baseline)
    - Late: Separate predictions + ensemble (baseline)
    - Intermediate: This approach (SOTA)
    """

    def __init__(
        self,
        metrics_encoder: nn.Module,
        logs_encoder: nn.Module,
        traces_encoder: nn.Module,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_strategy: str = 'intermediate'
    ):
        super().__init__()

        self.metrics_encoder = metrics_encoder
        self.logs_encoder = logs_encoder
        self.traces_encoder = traces_encoder
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == 'intermediate':
            # Cross-modal attention
            self.cross_attention = CrossModalAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout
            )

            # Projection layers to common dimension
            self.metrics_proj = nn.Linear(256, fusion_dim)
            self.logs_proj = nn.Linear(256, fusion_dim)
            self.traces_proj = nn.Linear(128, fusion_dim)

        elif fusion_strategy == 'early':
            # Simple concatenation
            total_dim = 256 + 256 + 128
            self.fusion_layer = nn.Linear(total_dim, fusion_dim)

        elif fusion_strategy == 'late':
            # Separate heads
            self.metrics_head = nn.Linear(256, fusion_dim)
            self.logs_head = nn.Linear(256, fusion_dim)
            self.traces_head = nn.Linear(128, fusion_dim)

        # Final layers
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        metrics: Optional[torch.Tensor] = None,
        logs: Optional[torch.Tensor] = None,
        traces: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal_weights: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multimodal data.

        Args:
            metrics: (batch, seq_len, features) metrics tensor
            logs: (batch, log_embedding_dim) log embeddings
            traces: (node_features, edge_index) graph data
            causal_weights: (batch,) causal scores for weighting
            return_attention: Return attention weights for visualization

        Returns:
            Dictionary with:
            - 'fused': (batch, fusion_dim) fused representation
            - 'attention': Attention weights if requested
            - 'modality_embeddings': Individual embeddings
        """
        batch_size = None
        embeddings = []
        modality_names = []
        modality_embeddings = {}

        # Encode metrics
        if metrics is not None:
            metrics_emb = self.metrics_encoder(metrics)  # (batch, 256)
            batch_size = metrics_emb.size(0)
            embeddings.append(metrics_emb)
            modality_names.append('metrics')
            modality_embeddings['metrics'] = metrics_emb

        # Encode logs
        if logs is not None:
            if isinstance(logs, dict):
                logs_emb = self.logs_encoder(**logs)
            else:
                logs_emb = self.logs_encoder(logs)  # (batch, 256)

            if batch_size is None:
                batch_size = logs_emb.size(0)
            embeddings.append(logs_emb)
            modality_names.append('logs')
            modality_embeddings['logs'] = logs_emb

        # Encode traces
        if traces is not None:
            node_features, edge_index = traces
            if hasattr(self.traces_encoder, 'get_node_embeddings'):
                # GCN/GAT returns node embeddings
                node_emb = self.traces_encoder.get_node_embeddings(node_features, edge_index)
                # Pool to graph-level
                traces_emb = node_emb.mean(dim=0, keepdim=True)  # (1, 128)
                # Expand to batch size
                if batch_size is not None:
                    traces_emb = traces_emb.expand(batch_size, -1)
            else:
                traces_emb = self.traces_encoder(node_features, edge_index)  # (batch, 128)

            if batch_size is None:
                batch_size = traces_emb.size(0)
            embeddings.append(traces_emb)
            modality_names.append('traces')
            modality_embeddings['traces'] = traces_emb

        # Check if we have at least one modality
        if len(embeddings) == 0:
            raise ValueError("At least one modality (metrics, logs, or traces) must be provided")

        # Apply fusion strategy
        if self.fusion_strategy == 'intermediate':
            return self._intermediate_fusion(
                embeddings,
                modality_names,
                modality_embeddings,
                causal_weights,
                return_attention
            )
        elif self.fusion_strategy == 'early':
            return self._early_fusion(embeddings, modality_embeddings)
        elif self.fusion_strategy == 'late':
            return self._late_fusion(embeddings, modality_names, modality_embeddings)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _intermediate_fusion(
        self,
        embeddings: list,
        modality_names: list,
        modality_embeddings: dict,
        causal_weights: Optional[torch.Tensor],
        return_attention: bool
    ) -> Dict[str, torch.Tensor]:
        """Intermediate fusion with cross-modal attention."""
        # Project all embeddings to fusion_dim
        projected = []
        for i, (emb, name) in enumerate(zip(embeddings, modality_names)):
            if name == 'metrics':
                projected.append(self.metrics_proj(emb))
            elif name == 'logs':
                projected.append(self.logs_proj(emb))
            elif name == 'traces':
                projected.append(self.traces_proj(emb))

        # Stack for attention: (batch, num_modalities, fusion_dim)
        stacked = torch.stack(projected, dim=1)

        # Apply self-attention across modalities
        attended, attn_weights = self.cross_attention(
            query=stacked,
            key=stacked,
            value=stacked,
            return_attention=return_attention
        )

        # Apply causal weighting if provided
        if causal_weights is not None:
            # causal_weights: (batch,) → (batch, 1, 1)
            causal_weights = causal_weights.unsqueeze(1).unsqueeze(2)
            attended = attended * causal_weights

        # Aggregate across modalities (mean pooling)
        fused = attended.mean(dim=1)  # (batch, fusion_dim)

        # Normalize and dropout
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)

        result = {
            'fused': fused,
            'modality_embeddings': modality_embeddings
        }

        if return_attention:
            result['attention'] = attn_weights

        return result

    def _early_fusion(
        self,
        embeddings: list,
        modality_embeddings: dict
    ) -> Dict[str, torch.Tensor]:
        """Early fusion: Simple concatenation."""
        # Concatenate all embeddings
        concatenated = torch.cat(embeddings, dim=1)  # (batch, total_dim)

        # Project to fusion_dim
        fused = self.fusion_layer(concatenated)  # (batch, fusion_dim)

        # Normalize and dropout
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)

        return {
            'fused': fused,
            'modality_embeddings': modality_embeddings
        }

    def _late_fusion(
        self,
        embeddings: list,
        modality_names: list,
        modality_embeddings: dict
    ) -> Dict[str, torch.Tensor]:
        """Late fusion: Average predictions from each modality."""
        # Project each modality separately
        projected = []
        for emb, name in zip(embeddings, modality_names):
            if name == 'metrics':
                projected.append(self.metrics_head(emb))
            elif name == 'logs':
                projected.append(self.logs_head(emb))
            elif name == 'traces':
                projected.append(self.traces_head(emb))

        # Average across modalities
        fused = torch.stack(projected, dim=0).mean(dim=0)  # (batch, fusion_dim)

        # Normalize and dropout
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)

        return {
            'fused': fused,
            'modality_embeddings': modality_embeddings
        }


class ModalityDropout(nn.Module):
    """
    Dropout entire modalities during training for robustness.

    Helps the model learn to handle:
    - Missing modalities during inference
    - Unreliable modalities
    - Partial data availability

    Reference: "Multimodal Dropout" (NeurIPS 2018)
    """

    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(
        self,
        metrics: Optional[torch.Tensor] = None,
        logs: Optional[torch.Tensor] = None,
        traces: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple]]:
        """
        Randomly drop entire modalities during training.

        Args:
            metrics, logs, traces: Modality inputs

        Returns:
            Same inputs, but some may be None if dropped
        """
        if not self.training:
            return metrics, logs, traces

        # Keep originals for potential restore
        original_metrics = metrics
        original_logs = logs
        original_traces = traces

        # Count available modalities
        available = []
        if metrics is not None:
            available.append('metrics')
        if logs is not None:
            available.append('logs')
        if traces is not None:
            available.append('traces')

        # If only one modality available, don't drop it
        if len(available) <= 1:
            return metrics, logs, traces

        # Randomly drop each modality
        if metrics is not None and torch.rand(1).item() < self.dropout_rate:
            metrics = None

        if logs is not None and torch.rand(1).item() < self.dropout_rate:
            logs = None

        if traces is not None and torch.rand(1).item() < self.dropout_rate:
            traces = None

        # Ensure at least one modality remains
        if metrics is None and logs is None and traces is None:
            # Randomly restore one of the originally available modalities
            restore_choice = available[torch.randint(0, len(available), (1,)).item()]
            if restore_choice == 'metrics':
                metrics = original_metrics
            elif restore_choice == 'logs':
                logs = original_logs
            elif restore_choice == 'traces':
                traces = original_traces

        return metrics, logs, traces


def create_multimodal_fusion(
    metrics_encoder: nn.Module,
    logs_encoder: nn.Module,
    traces_encoder: nn.Module,
    fusion_strategy: str = 'intermediate',
    fusion_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    use_modality_dropout: bool = True,
    modality_dropout_rate: float = 0.1
) -> nn.Module:
    """
    Factory function to create multimodal fusion module.

    Args:
        metrics_encoder: Encoder for metrics (Chronos/TCN)
        logs_encoder: Encoder for logs (TF-IDF/BERT)
        traces_encoder: Encoder for traces (GCN/GAT)
        fusion_strategy: 'intermediate', 'early', or 'late'
        fusion_dim: Fusion embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_modality_dropout: Enable modality dropout
        modality_dropout_rate: Rate for modality dropout

    Returns:
        Configured MultimodalFusion module
    """
    fusion = MultimodalFusion(
        metrics_encoder=metrics_encoder,
        logs_encoder=logs_encoder,
        traces_encoder=traces_encoder,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        dropout=dropout,
        fusion_strategy=fusion_strategy
    )

    if use_modality_dropout:
        # Wrap with modality dropout
        class FusionWithModalityDropout(nn.Module):
            def __init__(self, fusion, dropout_rate):
                super().__init__()
                self.fusion = fusion
                self.modality_dropout = ModalityDropout(dropout_rate)

            def forward(self, metrics=None, logs=None, traces=None, **kwargs):
                metrics, logs, traces = self.modality_dropout(metrics, logs, traces)
                return self.fusion(metrics, logs, traces, **kwargs)

        return FusionWithModalityDropout(fusion, modality_dropout_rate)

    return fusion
