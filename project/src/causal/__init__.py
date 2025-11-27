"""
Causal discovery module for fault propagation analysis.

Implements:
1. PCMCI/PCMCIplus from tigramite library for time series causal discovery
2. LLM-based causal prior generation for domain knowledge injection
"""

from .pcmci import (
    compute_pcmci_weights,
    precompute_causal_weights,
    CausalWeightComputer
)

from .llm_prior import (
    LLMCausalPrior,
    CausalWeightManager,
    get_system_type
)

__all__ = [
    # PCMCI (statistical)
    'compute_pcmci_weights',
    'precompute_causal_weights',
    'CausalWeightComputer',
    # LLM Prior (domain knowledge)
    'LLMCausalPrior',
    'CausalWeightManager',
    'get_system_type'
]
