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


class RCAEvaluator:
    """
    Evaluator for Root Cause Analysis predictions.
    
    Handles both single-case and batch evaluation with support for:
    - AC@k (Accuracy at top-k)
    - Avg@k (Position-weighted accuracy)
    - MRR (Mean Reciprocal Rank)
    - Per-fault-type and per-system breakdown
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of k values for AC@k and Avg@k metrics
        """
        self.k_values = k_values or [1, 3, 5]
        self.results_history = []
    
    def evaluate_single_case(
        self,
        predicted_ranking: List[Tuple[str, float]],
        ground_truth: str,
        metadata: Dict = None
    ) -> Dict:
        """
        Evaluate a single failure case.
        
        Args:
            predicted_ranking: List of (service_name, score) tuples, sorted by score descending
            ground_truth: True root cause service name
            metadata: Optional metadata (fault_type, system, etc.)
            
        Returns:
            Dictionary with per-case metrics
        """
        # Extract service names in ranked order
        if predicted_ranking and isinstance(predicted_ranking[0], tuple):
            ranked_services = [s[0] for s in predicted_ranking]
        else:
            ranked_services = predicted_ranking
        
        result = {
            'ground_truth': ground_truth,
            'predicted_ranking': ranked_services[:10],  # Store top 10
            'metadata': metadata or {}
        }
        
        # Compute rank of ground truth
        try:
            rank = ranked_services.index(ground_truth) + 1  # 1-indexed
            result['rank'] = rank
            result['found'] = True
        except ValueError:
            result['rank'] = len(ranked_services) + 1
            result['found'] = False
        
        # Compute AC@k for each k
        for k in self.k_values:
            result[f'AC@{k}'] = 1.0 if result['rank'] <= k else 0.0
        
        # Compute reciprocal rank
        result['reciprocal_rank'] = 1.0 / result['rank'] if result['found'] else 0.0
        
        return result
    
    def aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """
        Aggregate results from multiple cases.
        
        Args:
            results: List of single-case result dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        if not results:
            return {f'AC@{k}': 0.0 for k in self.k_values}
        
        aggregated = {}
        n_cases = len(results)
        
        # AC@k metrics
        for k in self.k_values:
            ac_k = sum(r.get(f'AC@{k}', 0.0) for r in results) / n_cases
            aggregated[f'AC@{k}'] = ac_k
        
        # Avg@k metrics (position-weighted)
        for k in self.k_values:
            avg_k = sum(
                1.0 / r['rank'] if r.get('rank', float('inf')) <= k else 0.0
                for r in results
            ) / n_cases
            aggregated[f'Avg@{k}'] = avg_k
        
        # MRR
        aggregated['MRR'] = sum(r.get('reciprocal_rank', 0.0) for r in results) / n_cases
        
        # Additional stats
        aggregated['n_cases'] = n_cases
        aggregated['n_found'] = sum(1 for r in results if r.get('found', False))
        aggregated['coverage'] = aggregated['n_found'] / n_cases
        
        return aggregated
    
    def evaluate_batch(
        self,
        predictions: List[List[str]],
        ground_truths: List[str],
        metadata_list: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: List of ranked service lists per case
            ground_truths: List of true root cause services
            metadata_list: Optional list of metadata dicts
            
        Returns:
            Aggregated metrics dictionary
        """
        if metadata_list is None:
            metadata_list = [{}] * len(predictions)
        
        results = []
        for pred, gt, meta in zip(predictions, ground_truths, metadata_list):
            # Convert to (service, score) format if needed
            if pred and not isinstance(pred[0], tuple):
                pred = [(s, 1.0 - i/len(pred)) for i, s in enumerate(pred)]
            
            result = self.evaluate_single_case(pred, gt, meta)
            results.append(result)
        
        return self.aggregate_results(results)
    
    def evaluate_by_category(
        self,
        results: List[Dict],
        category_key: str = 'fault_type'
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics broken down by category.
        
        Args:
            results: List of single-case results with metadata
            category_key: Key in metadata to group by (e.g., 'fault_type', 'system')
            
        Returns:
            Dictionary mapping category -> metrics
        """
        # Group results by category
        categories = {}
        for r in results:
            cat = r.get('metadata', {}).get(category_key, 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        # Aggregate per category
        category_metrics = {}
        for cat, cat_results in categories.items():
            category_metrics[cat] = self.aggregate_results(cat_results)
        
        return category_metrics
    
    def reset(self):
        """Clear results history."""
        self.results_history = []


# Convenience function for quick evaluation
def evaluate_predictions(
    predictions: List[List[str]],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Quick evaluation of predictions.
    
    Args:
        predictions: List of ranked service lists per case
        ground_truths: List of true root cause services
        
    Returns:
        Dictionary with AC@1, AC@3, AC@5, Avg@5, MRR
    """
    evaluator = RCAEvaluator()
    return evaluator.evaluate_batch(predictions, ground_truths)
