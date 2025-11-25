"""
Multimodal fusion module.

Implements intermediate fusion with cross-modal attention
for combining metrics, logs, and traces.
"""

from .multimodal_fusion import (
    MultimodalFusion,
    CrossModalAttention,
    ModalityDropout,
    create_multimodal_fusion
)

__all__ = [
    'MultimodalFusion',
    'CrossModalAttention',
    'ModalityDropout',
    'create_multimodal_fusion'
]
