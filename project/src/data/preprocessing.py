"""
Data preprocessing utilities for multimodal data.

This module handles preprocessing for:
- Metrics: Normalization, windowing, feature engineering
- Logs: Template extraction, temporal alignment
- Traces: Service graph construction, feature extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


class MetricsPreprocessor:
    """
    Preprocessing for time series metrics data.

    Features:
    - Normalization (z-score, min-max)
    - Sliding window creation
    - Service-level aggregation
    - Missing value imputation
    - Outlier handling
    """

    def __init__(
        self,
        window_size: int = 12,  # 1 hour at 5-min granularity
        normalization: str = 'zscore',  # 'zscore', 'minmax', or 'none'
        fill_method: str = 'forward',  # 'forward', 'backward', 'interpolate'
        clip_outliers: bool = True,
        outlier_std: float = 5.0  # Clip beyond N std devs
    ):
        self.window_size = window_size
        self.normalization = normalization
        self.fill_method = fill_method
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std

        # Will be fitted on training data
        self.scaler = None
        if normalization == 'zscore':
            self.scaler = StandardScaler()
        elif normalization == 'minmax':
            self.scaler = MinMaxScaler()

    def fit(self, df: pd.DataFrame) -> 'MetricsPreprocessor':
        """
        Fit scaler on training data.

        Args:
            df: Training metrics dataframe (time, features)

        Returns:
            self
        """
        if self.scaler is not None:
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.scaler.fit(df[numeric_cols].values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform metrics data.

        Pipeline:
        1. Handle missing values
        2. Clip outliers (if enabled)
        3. Normalize (if enabled)

        Args:
            df: Metrics dataframe (time, features)

        Returns:
            Preprocessed dataframe
        """
        df = df.copy()

        # 1. Handle missing values
        df = self._fill_missing(df)

        # 2. Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 3. Clip outliers
        if self.clip_outliers:
            df = self._clip_outliers(df, numeric_cols)

        # 4. Normalize
        if self.scaler is not None:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols].values)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values."""
        if self.fill_method == 'forward':
            return df.fillna(method='ffill').fillna(method='bfill')
        elif self.fill_method == 'backward':
            return df.fillna(method='bfill').fillna(method='ffill')
        elif self.fill_method == 'interpolate':
            return df.interpolate(method='linear', limit_direction='both')
        else:
            return df.fillna(0)

    def _clip_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Clip extreme outliers beyond N standard deviations."""
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Avoid division by zero
                lower = mean - self.outlier_std * std
                upper = mean + self.outlier_std * std
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def create_windows(
        self,
        df: pd.DataFrame,
        stride: int = 1,
        drop_last: bool = False
    ) -> np.ndarray:
        """
        Create sliding windows for temporal context.

        Args:
            df: Preprocessed metrics dataframe
            stride: Step size between windows (default 1 = no overlap)
            drop_last: Drop incomplete last window

        Returns:
            (n_windows, window_size, n_features) array
        """
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        values = df[numeric_cols].values

        n_samples, n_features = values.shape

        # Calculate number of windows
        if drop_last:
            n_windows = (n_samples - self.window_size) // stride + 1
        else:
            n_windows = (n_samples - self.window_size + stride) // stride

        windows = []
        for i in range(0, n_windows * stride, stride):
            if i + self.window_size <= n_samples:
                windows.append(values[i:i+self.window_size])
            elif not drop_last:
                # Pad incomplete window
                window = values[i:]
                pad_width = ((0, self.window_size - len(window)), (0, 0))
                window = np.pad(window, pad_width, mode='edge')
                windows.append(window)

        if len(windows) == 0:
            return np.empty((0, self.window_size, n_features))

        return np.array(windows)

    def aggregate_by_service(
        self,
        df: pd.DataFrame,
        service_col: str = 'service_name',
        agg_funcs: List[str] = ['mean', 'max', 'std']
    ) -> pd.DataFrame:
        """
        Aggregate metrics by service.

        Args:
            df: Metrics dataframe with service identifier
            service_col: Column name for service
            agg_funcs: Aggregation functions

        Returns:
            Aggregated dataframe (services, features * agg_funcs)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        agg_dict = {col: agg_funcs for col in numeric_cols}
        aggregated = df.groupby(service_col).agg(agg_dict)

        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]

        return aggregated.reset_index()


class LogsPreprocessor:
    """
    Preprocessing for log data.

    Features:
    - Template extraction (Drain3)
    - Temporal alignment with metrics (1-min windows)
    - Error pattern extraction
    - Template embedding
    """

    def __init__(
        self,
        window_size: str = '1min',
        drain_config: Optional[Dict] = None
    ):
        self.window_size = window_size
        self.drain_config = drain_config or {
            'similarity_threshold': 0.5,
            'depth': 4,
            'max_children': 100
        }
        self.drain_parser = None

    def parse_logs(self, log_df: pd.DataFrame, log_col: str = 'content') -> pd.DataFrame:
        """
        Parse logs to extract templates.

        Args:
            log_df: Dataframe with 'timestamp' and log content column
            log_col: Name of column containing log messages

        Returns:
            Dataframe with additional 'template_id' and 'template' columns
        """
        try:
            from drain3 import TemplateMiner
            from drain3.template_miner_config import TemplateMinerConfig
        except ImportError:
            raise ImportError(
                "drain3 not installed. Install with: pip install drain3"
            )

        # Initialize Drain3 if not already done
        if self.drain_parser is None:
            config = TemplateMinerConfig()
            config.load(self.drain_config)
            config.profiling_enabled = False
            self.drain_parser = TemplateMiner(config=config)

        parsed = []
        for idx, row in log_df.iterrows():
            log_message = row[log_col]
            result = self.drain_parser.add_log_message(log_message)

            parsed.append({
                'timestamp': row.get('timestamp'),
                'template_id': result['cluster_id'],
                'template': result['template_mined'],
                'original': log_message
            })

        return pd.DataFrame(parsed)

    def temporal_align(
        self,
        log_df: pd.DataFrame,
        window_size: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate logs into time windows.

        Args:
            log_df: Parsed logs with 'timestamp' and 'template_id'
            window_size: Time window (e.g., '1min', '5min')

        Returns:
            Dataframe with (timestamp_window, template_counts, error_rate, etc.)
        """
        window = window_size or self.window_size

        if 'timestamp' not in log_df.columns:
            raise ValueError("log_df must have 'timestamp' column")

        # Ensure timestamp is datetime
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

        # Set timestamp as index for resampling
        log_df = log_df.set_index('timestamp')

        # Count logs per window
        log_counts = log_df.resample(window).size().rename('log_count')

        # Count unique templates per window
        template_counts = log_df.groupby(pd.Grouper(freq=window))['template_id'].nunique()
        template_counts = template_counts.rename('unique_templates')

        # Combine
        aligned = pd.DataFrame({
            'log_count': log_counts,
            'unique_templates': template_counts
        }).fillna(0)

        return aligned.reset_index()


class TracesPreprocessor:
    """
    Preprocessing for distributed traces.

    Features:
    - Service dependency graph construction
    - Node feature extraction (latency, error rate)
    - Edge feature extraction (call frequency)
    - Temporal aggregation
    """

    def __init__(self):
        pass

    def build_service_graph(
        self,
        traces_df: pd.DataFrame,
        parent_col: str = 'parent_service',
        child_col: str = 'service_name'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Build service dependency graph from traces.

        Args:
            traces_df: Traces with parent-child service relationships
            parent_col: Column name for parent service
            child_col: Column name for child service

        Returns:
            edge_index: (2, num_edges) array of edges
            service_mapping: Dict mapping service names to indices
        """
        # Get unique services
        all_services = set(traces_df[parent_col].dropna()) | set(traces_df[child_col].dropna())
        all_services = sorted(list(all_services - {None, '', 'root'}))

        # Create service to index mapping
        service_mapping = {service: idx for idx, service in enumerate(all_services)}

        # Build edge list
        edges = []
        for _, row in traces_df.iterrows():
            parent = row[parent_col]
            child = row[child_col]

            if parent in service_mapping and child in service_mapping:
                edges.append([service_mapping[parent], service_mapping[child]])

        if len(edges) == 0:
            # Return empty graph
            return np.array([[], []], dtype=np.int64), service_mapping

        # Convert to numpy array and transpose for PyG format
        edge_index = np.array(edges, dtype=np.int64).T

        return edge_index, service_mapping

    def extract_node_features(
        self,
        traces_df: pd.DataFrame,
        service_col: str = 'service_name',
        latency_col: str = 'duration',
        error_col: Optional[str] = 'error'
    ) -> pd.DataFrame:
        """
        Extract node-level features from traces.

        Features:
        - avg_latency, p50, p90, p99
        - error_rate
        - request_count
        - dependency_count

        Args:
            traces_df: Traces dataframe
            service_col: Service identifier column
            latency_col: Latency/duration column
            error_col: Error indicator column (optional)

        Returns:
            Dataframe with (service, features)
        """
        features = []

        for service in traces_df[service_col].unique():
            service_traces = traces_df[traces_df[service_col] == service]

            latencies = service_traces[latency_col].dropna()

            feat_dict = {
                'service': service,
                'avg_latency': latencies.mean() if len(latencies) > 0 else 0,
                'p50_latency': latencies.quantile(0.5) if len(latencies) > 0 else 0,
                'p90_latency': latencies.quantile(0.9) if len(latencies) > 0 else 0,
                'p99_latency': latencies.quantile(0.99) if len(latencies) > 0 else 0,
                'request_count': len(service_traces),
            }

            # Error rate if column exists
            if error_col and error_col in service_traces.columns:
                error_count = service_traces[error_col].sum()
                feat_dict['error_rate'] = error_count / len(service_traces) if len(service_traces) > 0 else 0
            else:
                feat_dict['error_rate'] = 0

            features.append(feat_dict)

        return pd.DataFrame(features)

    def extract_edge_features(
        self,
        traces_df: pd.DataFrame,
        parent_col: str = 'parent_service',
        child_col: str = 'service_name',
        latency_col: str = 'duration',
        error_col: Optional[str] = 'error'
    ) -> pd.DataFrame:
        """
        Extract edge-level features from traces.

        Features per edge:
        - call_frequency
        - avg_latency
        - error_rate

        Args:
            traces_df: Traces dataframe
            parent_col: Parent service column
            child_col: Child service column
            latency_col: Latency column
            error_col: Error indicator (optional)

        Returns:
            Dataframe with (parent, child, features)
        """
        edge_features = []

        # Group by parent-child pairs
        grouped = traces_df.groupby([parent_col, child_col])

        for (parent, child), group in grouped:
            if parent is None or child is None or parent == '' or child == '':
                continue

            latencies = group[latency_col].dropna()

            feat_dict = {
                'parent': parent,
                'child': child,
                'call_frequency': len(group),
                'avg_latency': latencies.mean() if len(latencies) > 0 else 0,
            }

            if error_col and error_col in group.columns:
                error_count = group[error_col].sum()
                feat_dict['error_rate'] = error_count / len(group) if len(group) > 0 else 0
            else:
                feat_dict['error_rate'] = 0

            edge_features.append(feat_dict)

        return pd.DataFrame(edge_features)


# Helper functions for preprocessing pipeline

def preprocess_failure_case(
    failure_case,
    metrics_preprocessor: MetricsPreprocessor,
    logs_preprocessor: Optional[LogsPreprocessor] = None,
    traces_preprocessor: Optional[TracesPreprocessor] = None,
    load_modalities: Dict[str, bool] = None
) -> Dict[str, any]:
    """
    Preprocess a single failure case with all modalities.

    Args:
        failure_case: FailureCase object from data loader
        metrics_preprocessor: Fitted MetricsPreprocessor
        logs_preprocessor: LogsPreprocessor (optional)
        traces_preprocessor: TracesPreprocessor (optional)
        load_modalities: Which modalities to load

    Returns:
        Dictionary with preprocessed data
    """
    if load_modalities is None:
        load_modalities = {'metrics': True, 'logs': False, 'traces': False}

    # Load data
    failure_case.load_data(**load_modalities)

    result = {
        'case_id': failure_case.case_id,
        'system': failure_case.system,
        'fault_type': failure_case.fault_type,
        'ground_truth': failure_case.root_cause_service
    }

    # Preprocess metrics
    if load_modalities.get('metrics') and failure_case.metrics is not None:
        metrics_df = metrics_preprocessor.transform(failure_case.metrics)
        result['metrics'] = metrics_df
        result['metrics_windows'] = metrics_preprocessor.create_windows(metrics_df)

    # Preprocess logs
    if load_modalities.get('logs') and failure_case.logs is not None and logs_preprocessor:
        parsed_logs = logs_preprocessor.parse_logs(failure_case.logs)
        aligned_logs = logs_preprocessor.temporal_align(parsed_logs)
        result['logs'] = aligned_logs

    # Preprocess traces
    if load_modalities.get('traces') and failure_case.traces is not None and traces_preprocessor:
        edge_index, service_mapping = traces_preprocessor.build_service_graph(failure_case.traces)
        node_features = traces_preprocessor.extract_node_features(failure_case.traces)
        edge_features = traces_preprocessor.extract_edge_features(failure_case.traces)

        result['edge_index'] = edge_index
        result['service_mapping'] = service_mapping
        result['node_features'] = node_features
        result['edge_features'] = edge_features

    # Unload to save memory
    failure_case.unload_data()

    return result
