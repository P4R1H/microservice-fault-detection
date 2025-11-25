"""
Evaluation module for RCA performance metrics.

Implements:
- AC@k: Accuracy at top-k
- Avg@k: Position-weighted accuracy
- MRR: Mean reciprocal rank
- Statistical significance testing
"""

from .metrics import (
    compute_ac_at_k,
    compute_avg_at_k,
    compute_mrr,
    compute_all_metrics,
    RCAEvaluator,
    evaluate_predictions,
    paired_ttest,
    compute_cohens_d
)

__all__ = [
    'compute_ac_at_k',
    'compute_avg_at_k',
    'compute_mrr',
    'compute_all_metrics',
    'RCAEvaluator',
    'evaluate_predictions',
    'paired_ttest',
    'compute_cohens_d'
]
