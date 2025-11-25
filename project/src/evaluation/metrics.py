"""
RCA evaluation metrics matching RCAEval standards.

Reference: RCAEval paper (WWW'25, ASE 2024)
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats


def compute_ac_at_k(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = 5
) -> float:
    """
    Compute Accuracy@k: Is ground truth in top-k predictions?

    Args:
        predictions: List of ranked service lists per case
        ground_truth: List of true root cause services
        k: Top-k to consider

    Returns:
        AC@k score [0, 1]
    """
    correct = 0
    for pred, gt in zip(predictions, ground_truth):
        if gt in pred[:k]:
            correct += 1
    return correct / len(predictions)


def compute_avg_at_k(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = 5
) -> float:
    """
    Compute Avg@k: Position-weighted accuracy.

    Score = 1/rank if found in top-k, else 0

    Args:
        predictions: List of ranked service lists per case
        ground_truth: List of true root cause services
        k: Top-k to consider

    Returns:
        Avg@k score [0, 1]
    """
    total_score = 0
    for pred, gt in zip(predictions, ground_truth):
        try:
            rank = pred[:k].index(gt) + 1  # 1-indexed
            total_score += 1.0 / rank
        except ValueError:
            # Not in top-k
            total_score += 0
    return total_score / len(predictions)


def compute_mrr(
    predictions: List[List[str]],
    ground_truth: List[str]
) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = average(1/rank) across all cases

    Args:
        predictions: List of ranked service lists per case
        ground_truth: List of true root cause services

    Returns:
        MRR score [0, 1]
    """
    reciprocal_ranks = []
    for pred, gt in zip(predictions, ground_truth):
        try:
            rank = pred.index(gt) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # Not found at all
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)


def compute_all_metrics(
    predictions: List[List[str]],
    ground_truth: List[str]
) -> Dict[str, float]:
    """
    Compute all RCA metrics.

    Returns:
        Dictionary with AC@1, AC@3, AC@5, Avg@5, MRR
    """
    return {
        'AC@1': compute_ac_at_k(predictions, ground_truth, k=1),
        'AC@3': compute_ac_at_k(predictions, ground_truth, k=3),
        'AC@5': compute_ac_at_k(predictions, ground_truth, k=5),
        'Avg@5': compute_avg_at_k(predictions, ground_truth, k=5),
        'MRR': compute_mrr(predictions, ground_truth)
    }


def paired_ttest(
    method1_scores: List[float],
    method2_scores: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Paired t-test for statistical significance.

    Args:
        method1_scores: Scores from method 1 (per case)
        method2_scores: Scores from method 2 (per case)
        alpha: Significance level

    Returns:
        Dictionary with t-statistic, p-value, significant
    """
    t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': compute_cohens_d(method1_scores, method2_scores)
    }


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.

    Interpretation:
    - 0.2: Small effect
    - 0.5: Medium effect
    - 0.8: Large effect
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std


# TODO: Add confidence intervals
# TODO: Add per-fault-type metrics
# TODO: Add per-system metrics
