"""
RCA models module.

Current model: MultimodalRCAModel V4 with configurable encoders.
"""

from .rca_v4_multimodal import (
    MultimodalRCAModel,
    MultimodalLoss,
    create_multimodal_model
)

__all__ = [
    'MultimodalRCAModel',
    'MultimodalLoss',
    'create_multimodal_model'
]
