"""
Causal discovery module for fault propagation analysis.

Implements PCMCI/PCMCIplus from tigramite library for
time series causal discovery.
"""

from .pcmci import (
    PCMCIDiscovery,
    GrangerLassoRCA,
    discover_causal_relations,
    visualize_causal_graph,
    analyze_causal_paths,
    compute_causal_strength
)

__all__ = [
    'PCMCIDiscovery',
    'GrangerLassoRCA',
    'discover_causal_relations',
    'visualize_causal_graph',
    'analyze_causal_paths',
    'compute_causal_strength'
]
