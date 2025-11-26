"""
Encoder module for multimodal data encoding.

This module provides encoders for each modality:
- Metrics: Chronos-Bolt-Tiny (foundation model) or TCN (trained)
- Logs: TF-IDF weighted encoding (proven best performer)
- Traces: TCN-based trace encoder
"""

from .tcn import TCNEncoder, TemporalBlock
from .logs_encoder import TFIDFLogsEncoder
from .metrics_encoder import ChronosEncoder
from .traces_encoder import TracesTCNEncoder

__all__ = [
    # Core encoders
    'TCNEncoder',
    'TemporalBlock',
    # Logs
    'TFIDFLogsEncoder',
    # Metrics
    'ChronosEncoder',
    # Traces
    'TracesTCNEncoder',
]
