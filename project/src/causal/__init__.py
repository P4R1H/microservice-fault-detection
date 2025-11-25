"""
Causal discovery module for fault propagation analysis.

Implements PCMCI/PCMCIplus from tigramite library for
time series causal discovery.
"""

from .pcmci import (
    compute_pcmci_weights,
    precompute_causal_weights
)

__all__ = [
    'compute_pcmci_weights',
    'precompute_causal_weights'
]
