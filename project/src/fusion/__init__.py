"""
Multimodal fusion module.

Implements intermediate fusion with cross-modal attention
for combining metrics, logs, and traces.
"""

from .attention import (
    CrossModalAttention,
    MultiHeadCrossAttention,
    CausalGraphAttention
)

__all__ = [
    'CrossModalAttention',
    'MultiHeadCrossAttention',
    'CausalGraphAttention'
]
