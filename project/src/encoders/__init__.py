"""
Encoder module for multimodal data encoding.

This module provides encoders for each modality:
- Metrics: Chronos-Bolt-Tiny (foundation model) or TCN (trained)
- Logs: Drain3 + embeddings
- Traces: GCN/GAT on service dependency graphs
"""

from .tcn import TCNEncoder, TemporalBlock

__all__ = [
    'TCNEncoder',
    'TemporalBlock'
]
