"""
Evaluation module for RCA performance metrics.

Implements:
- AC@k: Accuracy at top-k
- Avg@k: Position-weighted accuracy
- MRR: Mean reciprocal rank
- Statistical significance testing
"""

from .metrics import compute_ac_at_k, compute_avg_at_k, compute_mrr

__all__ = ['compute_ac_at_k', 'compute_avg_at_k', 'compute_mrr']
