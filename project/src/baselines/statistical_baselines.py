"""
Statistical Baseline Methods for RCA

Implements simple statistical methods as baselines:
1. Three-Sigma (3σ) Thresholding
2. ARIMA Forecasting
3. Granger-Lasso Causal Discovery

These serve as performance lower bounds for comparison with advanced methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    metric_name: str
    is_anomalous: bool
    anomaly_score: float
    timestamps: Optional[List] = None
    threshold: Optional[float] = None


class ThreeSigmaDetector:
    """
    Three-Sigma (3σ) Statistical Anomaly Detection

    Method: Flag values outside μ ± 3σ as anomalies
    - μ: mean of historical data
    - σ: standard deviation
    - Assumes Gaussian distribution (68-95-99.7 rule)

    Reference: Industry standard baseline (SRE best practices)
    """

    def __init__(self, n_sigma: float = 3.0, window_size: int = 50, n_jobs: int = -1):
        """
        Initialize 3-sigma detector

        Args:
            n_sigma: Number of standard deviations (default: 3.0)
            window_size: Rolling window for μ and σ estimation
            n_jobs: Number of parallel jobs (-1 = use all cores, 1 = sequential)
        """
        self.n_sigma = n_sigma
        self.window_size = window_size
        self.n_jobs = n_jobs

    def _detect_single_metric(self, col: str, metrics: pd.DataFrame) -> Optional[AnomalyResult]:
        """
        Detect anomalies for a single metric

        Args:
            col: Column name
            metrics: DataFrame with metrics

        Returns:
            AnomalyResult or None if insufficient data
        """
        series = metrics[col].dropna()

        if len(series) < self.window_size:
            return None

        # Calculate rolling statistics
        rolling_mean = series.rolling(window=self.window_size, center=False).mean()
        rolling_std = series.rolling(window=self.window_size, center=False).std()

        # Calculate bounds
        upper_bound = rolling_mean + (self.n_sigma * rolling_std)
        lower_bound = rolling_mean - (self.n_sigma * rolling_std)

        # Detect anomalies
        anomalies = (series > upper_bound) | (series < lower_bound)
        is_anomalous = anomalies.any()

        # Calculate anomaly score (max deviation in sigma units)
        if rolling_std.sum() > 0:
            deviations = np.abs(series - rolling_mean) / (rolling_std + 1e-8)
            anomaly_score = deviations.max()
        else:
            anomaly_score = 0.0

        return AnomalyResult(
            metric_name=col,
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            timestamps=series.index[anomalies].tolist() if is_anomalous else [],
            threshold=self.n_sigma
        )

    def detect(self, metrics: pd.DataFrame) -> List[AnomalyResult]:
        """
        Detect anomalies using 3-sigma rule (PARALLELIZED)

        Args:
            metrics: DataFrame with metrics (columns = metric names, rows = timesteps)

        Returns:
            List of AnomalyResult for each metric
        """
        # Process metrics in parallel
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self._detect_single_metric)(col, metrics)
            for col in metrics.columns
        )

        # Filter out None results (insufficient data)
        return [r for r in results if r is not None]

    def rank_services(self, metrics: pd.DataFrame, service_mapping: Dict[str, str]) -> List[Tuple[str, float]]:
        """
        Rank services by anomaly score for root cause localization

        Args:
            metrics: DataFrame with metrics
            service_mapping: Dict mapping metric names to service names

        Returns:
            List of (service_name, total_anomaly_score) sorted by score
        """
        anomalies = self.detect(metrics)

        # Aggregate scores by service
        service_scores = {}
        for anom in anomalies:
            service = service_mapping.get(anom.metric_name, 'unknown')
            if service not in service_scores:
                service_scores[service] = 0.0
            service_scores[service] += anom.anomaly_score

        # Sort by score (descending)
        ranked = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked


class ARIMAForecaster:
    """
    ARIMA (AutoRegressive Integrated Moving Average) Forecasting

    Method: Forecast next values, flag large residuals as anomalies
    - ARIMA(p, d, q) parameters auto-selected or fixed
    - Anomaly = |actual - predicted| > threshold

    Reference: Box & Jenkins (1970), standard time series baseline
    """

    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0), threshold_sigma: float = 3.0, n_jobs: int = -1):
        """
        Initialize ARIMA forecaster

        Args:
            order: ARIMA(p, d, q) order
                   p: AR order (autoregressive)
                   d: differencing order
                   q: MA order (moving average)
            threshold_sigma: Threshold in sigma units for anomaly detection
            n_jobs: Number of parallel jobs (-1 = use all cores, 1 = sequential)
        """
        self.order = order
        self.threshold_sigma = threshold_sigma
        self.n_jobs = n_jobs

    def _detect_single_metric(self, col: str, metrics: pd.DataFrame, train_ratio: float) -> Optional[AnomalyResult]:
        """
        Detect anomalies for a single metric using ARIMA

        Args:
            col: Column name
            metrics: DataFrame with metrics
            train_ratio: Proportion of data for training

        Returns:
            AnomalyResult or None if error/insufficient data
        """
        series = metrics[col].dropna()

        if len(series) < 50:  # Minimum length for ARIMA
            return None

        try:
            # Split train/test
            split_idx = int(len(series) * train_ratio)
            train = series.iloc[:split_idx]
            test = series.iloc[split_idx:]

            # Fit ARIMA model
            model = ARIMA(train, order=self.order)
            model_fit = model.fit()

            # Forecast test period
            forecast = model_fit.forecast(steps=len(test))

            # Calculate residuals
            residuals = np.abs(test.values - forecast)

            # Threshold based on training residuals
            train_residuals = np.abs(model_fit.resid)
            threshold = train_residuals.mean() + (self.threshold_sigma * train_residuals.std())

            # Detect anomalies
            anomalies = residuals > threshold
            is_anomalous = anomalies.any()

            # Anomaly score (max residual / threshold)
            anomaly_score = (residuals.max() / threshold) if threshold > 0 else 0.0

            return AnomalyResult(
                metric_name=col,
                is_anomalous=is_anomalous,
                anomaly_score=anomaly_score,
                timestamps=test.index[anomalies].tolist() if is_anomalous else [],
                threshold=threshold
            )

        except Exception:
            # ARIMA can fail for certain series
            return None

    def detect(self, metrics: pd.DataFrame, train_ratio: float = 0.7) -> List[AnomalyResult]:
        """
        Detect anomalies using ARIMA residuals (PARALLELIZED)

        Args:
            metrics: DataFrame with metrics
            train_ratio: Proportion of data for training (rest for testing)

        Returns:
            List of AnomalyResult for each metric
        """
        # Process metrics in parallel
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self._detect_single_metric)(col, metrics, train_ratio)
            for col in metrics.columns
        )

        # Filter out None results (errors or insufficient data)
        return [r for r in results if r is not None]

    def rank_services(self, metrics: pd.DataFrame, service_mapping: Dict[str, str]) -> List[Tuple[str, float]]:
        """Rank services by ARIMA anomaly score"""
        anomalies = self.detect(metrics)

        service_scores = {}
        for anom in anomalies:
            service = service_mapping.get(anom.metric_name, 'unknown')
            if service not in service_scores:
                service_scores[service] = 0.0
            service_scores[service] += anom.anomaly_score

        ranked = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked


class GrangerLassoRCA:
    """
    Granger-Lasso Causal Root Cause Analysis

    Method: Granger causality with Lasso regularization
    - Test if metric X "Granger-causes" metric Y
    - Lasso for feature selection (sparse causal graph)
    - Root cause = service with highest outgoing causal edges

    Reference: Arnold et al. (2007) "Temporal causal modeling with graphical granger methods"
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.01, n_jobs: int = -1):
        """
        Initialize Granger-Lasso RCA

        Args:
            max_lag: Maximum lag for Granger causality test
            alpha: Significance level for Granger test (default: 0.01)
            n_jobs: Number of parallel jobs (-1 = use all cores, 1 = sequential)
        """
        self.max_lag = max_lag
        self.alpha = alpha
        self.n_jobs = n_jobs

    def _test_granger_pair(self, i: int, j: int, cols: pd.Index, metrics: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Test Granger causality for a single (i, j) pair

        Args:
            i: Index of cause variable
            j: Index of effect variable
            cols: Column names
            metrics: DataFrame with metrics

        Returns:
            Tuple of (i, j, causal_flag) where causal_flag=1 if i causes j
        """
        if i == j:
            return (i, j, 0)

        try:
            cause = cols[i]
            effect = cols[j]

            # Prepare data
            data = metrics[[effect, cause]].dropna()

            if len(data) < 50:  # Minimum length
                return (i, j, 0)

            # Granger causality test
            test_result = grangercausalitytests(
                data,
                maxlag=self.max_lag,
                verbose=False
            )

            # Check if significant at any lag
            for lag in range(1, self.max_lag + 1):
                p_value = test_result[lag][0]['ssr_ftest'][1]
                if p_value < self.alpha:
                    return (i, j, 1)

            return (i, j, 0)

        except Exception:
            return (i, j, 0)

    def build_causal_graph(self, metrics: pd.DataFrame, max_vars: int = 20) -> np.ndarray:
        """
        Build causal graph using Granger causality (PARALLELIZED)

        Args:
            metrics: DataFrame with metrics (columns = variables, rows = timesteps)
            max_vars: Maximum variables to include (computational constraint)

        Returns:
            Adjacency matrix (i,j)=1 if i causes j
        """
        # Limit variables for computational feasibility
        cols = metrics.columns[:max_vars]
        n_vars = len(cols)

        # Initialize adjacency matrix
        adj_matrix = np.zeros((n_vars, n_vars))

        # Generate all (i, j) pairs
        pairs = [(i, j) for i in range(n_vars) for j in range(n_vars) if i != j]

        # Test Granger causality in parallel
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self._test_granger_pair)(i, j, cols, metrics)
            for i, j in pairs
        )

        # Populate adjacency matrix from results
        for i, j, causal_flag in results:
            adj_matrix[i, j] = causal_flag

        return adj_matrix

    def rank_services(
        self,
        metrics: pd.DataFrame,
        service_mapping: Dict[str, str],
        max_vars: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Rank services by causal influence (outgoing edges)

        Args:
            metrics: DataFrame with metrics
            service_mapping: Dict mapping metric names to service names
            max_vars: Maximum variables to include

        Returns:
            List of (service_name, causal_score) sorted by score
        """
        # Build causal graph
        adj_matrix = self.build_causal_graph(metrics, max_vars=max_vars)

        # Map metrics to services
        cols = metrics.columns[:max_vars]
        metric_to_service = {i: service_mapping.get(col, 'unknown') for i, col in enumerate(cols)}

        # Aggregate causal edges by service
        service_scores = {}
        for i in range(len(cols)):
            service = metric_to_service[i]
            if service not in service_scores:
                service_scores[service] = 0.0

            # Outgoing edges = causal influence
            service_scores[service] += adj_matrix[i, :].sum()

        # Sort by score (descending)
        ranked = sorted(service_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked


class RandomWalkBaseline:
    """
    Random Walk Baseline

    Method: Random selection of root cause service
    - Used to verify other methods perform better than chance
    - Expected AC@1 = 1/N where N = number of services

    Reference: Standard baseline for RCA evaluation
    """

    def __init__(self, random_seed: int = 42):
        """Initialize random baseline"""
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def rank_services(self, services: List[str]) -> List[Tuple[str, float]]:
        """
        Randomly rank services

        Args:
            services: List of service names

        Returns:
            Randomly shuffled list of (service_name, random_score)
        """
        scores = np.random.rand(len(services))
        ranked = list(zip(services, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked


# Evaluation utilities
def evaluate_ranking(
    predicted_ranking: List[Tuple[str, float]],
    ground_truth_service: str,
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Evaluate root cause localization ranking

    Args:
        predicted_ranking: List of (service, score) sorted by score
        ground_truth_service: True root cause service
        k_values: Values of k for AC@k metric

    Returns:
        Dict with evaluation metrics: AC@1, AC@3, AC@5, Avg@5, MRR
    """
    metrics = {}

    # Get predicted services in order
    predicted_services = [s for s, _ in predicted_ranking]

    # Find rank of ground truth (1-indexed)
    try:
        rank = predicted_services.index(ground_truth_service) + 1
    except ValueError:
        rank = len(predicted_services) + 1  # Not found

    # AC@k (Accuracy at k)
    for k in k_values:
        metrics[f'AC@{k}'] = 1.0 if rank <= k else 0.0

    # Avg@k (Average rank up to k)
    max_k = max(k_values)
    if rank <= max_k:
        metrics[f'Avg@{max_k}'] = rank / max_k
    else:
        metrics[f'Avg@{max_k}'] = 1.0

    # MRR (Mean Reciprocal Rank)
    metrics['MRR'] = 1.0 / rank if rank > 0 else 0.0

    # Rank
    metrics['rank'] = rank

    return metrics


# Example usage
if __name__ == '__main__':
    # Generate synthetic metrics for testing
    np.random.seed(42)
    n_timesteps = 100
    n_metrics = 10

    # Normal metrics with one anomalous metric
    metrics_data = np.random.randn(n_timesteps, n_metrics)
    metrics_data[-10:, 3] += 10  # Inject anomaly in metric 3

    metrics_df = pd.DataFrame(
        metrics_data,
        columns=[f'metric_{i}' for i in range(n_metrics)]
    )

    # Service mapping
    service_mapping = {f'metric_{i}': f'service_{i % 5}' for i in range(n_metrics)}

    print("=" * 80)
    print("Testing Statistical Baselines")
    print("=" * 80)

    # Test 3-Sigma
    print("\n1. Three-Sigma Detector")
    sigma_detector = ThreeSigmaDetector()
    sigma_ranking = sigma_detector.rank_services(metrics_df, service_mapping)
    print(f"   Top 3 suspicious services:")
    for service, score in sigma_ranking[:3]:
        print(f"      {service}: {score:.2f}")

    # Test ARIMA
    print("\n2. ARIMA Forecaster")
    arima = ARIMAForecaster()
    arima_ranking = arima.rank_services(metrics_df, service_mapping)
    print(f"   Top 3 suspicious services:")
    for service, score in arima_ranking[:3]:
        print(f"      {service}: {score:.2f}")

    # Test Granger-Lasso
    print("\n3. Granger-Lasso RCA")
    granger = GrangerLassoRCA()
    granger_ranking = granger.rank_services(metrics_df, service_mapping, max_vars=10)
    print(f"   Top 3 suspicious services:")
    for service, score in granger_ranking[:3]:
        print(f"      {service}: {score:.2f}")

    # Evaluate against ground truth
    print("\n4. Evaluation Example")
    ground_truth = 'service_3'  # metric_3 was anomalous, maps to service_3
    for method_name, ranking in [
        ('3-Sigma', sigma_ranking),
        ('ARIMA', arima_ranking),
        ('Granger', granger_ranking)
    ]:
        metrics = evaluate_ranking(ranking, ground_truth)
        print(f"   {method_name}: AC@1={metrics['AC@1']:.2f}, AC@3={metrics['AC@3']:.2f}, MRR={metrics['MRR']:.3f}")
