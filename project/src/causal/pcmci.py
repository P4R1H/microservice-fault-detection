"""
PCMCI causal discovery for fault propagation.

Uses tigramite library for temporal causal discovery on metrics data.

Reference:
- tigramite library (JMLR 2024)
- "Causal discovery for time series" (Science Advances 2019)
- ASE 2024: "Root Cause Analysis based on Causal Inference: How Far Are We?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import networkx as nx
import warnings

# Try to import tigramite
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests import ParCorr, GPDC
    TIGRAMITE_AVAILABLE = True
    _TIGRAMITE_IMPORT_ERROR = None
except ImportError as import_err:
    TIGRAMITE_AVAILABLE = False
    _TIGRAMITE_IMPORT_ERROR = import_err
    warnings.warn(
        "tigramite not installed. PCMCI causal discovery unavailable. "
        "Install with: pip install tigramite"
    )


class PCMCIDiscovery:
    """
    PCMCI causal discovery wrapper.

    Features:
    - Two-stage procedure (PC1 + MCI)
    - Handles autocorrelation explicitly
    - Detection power >80% in high-dimensional cases
    - Fast: Minutes for 10-50K datapoints

    Hyperparameters (from literature):
    - tau_max: 3-5 (fault propagation window in 5-min intervals)
    - pc_alpha: 0.1-0.2 (liberal parent discovery)
    - alpha_level: 0.01-0.05 (conservative final edges)
    """

    def __init__(
        self,
        tau_max: int = 5,
        pc_alpha: float = 0.15,
        alpha_level: float = 0.05,
        independence_test: str = 'parcorr',  # 'parcorr' or 'gpdc'
        verbosity: int = 0
    ):
        if not TIGRAMITE_AVAILABLE:
            raise ImportError(
                "tigramite not installed. Install with: pip install tigramite"
            ) from _TIGRAMITE_IMPORT_ERROR

        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.alpha_level = alpha_level
        self.independence_test = independence_test
        self.verbosity = verbosity

        # PCMCI will be initialized in discover_graph (needs data)
        self.pcmci = None
        self.last_results = None

    def discover_graph(
        self,
        data: np.ndarray,
        var_names: Optional[List[str]] = None,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Discover causal graph from time series data.

        Args:
            data: (n_timesteps, n_variables) time series array
            var_names: Variable names for interpretability
            mask: (n_timesteps, n_variables) boolean mask for missing values

        Returns:
            Dictionary with:
            - 'graph': Adjacency matrix (n_vars, n_vars, tau_max+1)
            - 'val_matrix': P-values
            - 'causal_graph': NetworkX DiGraph
            - 'summary': Human-readable summary
        """
        if not TIGRAMITE_AVAILABLE:
            raise ImportError("tigramite not installed") from _TIGRAMITE_IMPORT_ERROR

        # Validate input
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        n_timesteps, n_vars = data.shape

        if var_names is None:
            var_names = [f"var_{i}" for i in range(n_vars)]

        # Prepare data for tigramite
        dataframe = pp.DataFrame(
            data,
            datatime=np.arange(n_timesteps),
            var_names=var_names,
            mask=mask
        )

        # Initialize independence test
        if self.independence_test == 'parcorr':
            cond_ind_test = ParCorr(significance='analytic', verbosity=self.verbosity)
        elif self.independence_test == 'gpdc':
            cond_ind_test = GPDC(significance='analytic', verbosity=self.verbosity)
        else:
            raise ValueError(f"Unknown independence test: {self.independence_test}")

        # Initialize PCMCI
        self.pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=self.verbosity
        )

        # Run PCMCI algorithm
        results = self.pcmci.run_pcmci(
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
            alpha_level=self.alpha_level
        )

        self.last_results = results

        # Extract causal graph
        graph = results['graph']  # (n_vars, n_vars, tau_max+1)
        val_matrix = results['val_matrix']  # P-values
        p_matrix = results['p_matrix']  # Test statistics

        # Build NetworkX DiGraph for easier analysis
        causal_graph = self._build_networkx_graph(graph, var_names, val_matrix)

        # Generate summary
        summary = self._generate_summary(graph, var_names, val_matrix)

        return {
            'graph': graph,
            'val_matrix': val_matrix,
            'p_matrix': p_matrix,
            'causal_graph': causal_graph,
            'summary': summary,
            'var_names': var_names
        }

    def _build_networkx_graph(
        self,
        graph: np.ndarray,
        var_names: List[str],
        val_matrix: np.ndarray
    ) -> nx.DiGraph:
        """
        Convert PCMCI graph to NetworkX DiGraph.

        Args:
            graph: (n_vars, n_vars, tau_max+1) adjacency matrix
            var_names: Variable names
            val_matrix: P-values

        Returns:
            NetworkX DiGraph with edges and p-values
        """
        G = nx.DiGraph()

        # Add nodes
        for var_name in var_names:
            G.add_node(var_name)

        n_vars = graph.shape[0]

        # Add edges (only contemporaneous and lagged edges)
        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(self.tau_max + 1):
                    # graph[i, j, tau] indicates edge from j at time t-tau to i at time t
                    if graph[i, j, tau] != "" and graph[i, j, tau] != "":
                        # Extract edge type (-->, <--, o-o, etc.)
                        edge_type = graph[i, j, tau]

                        # Only add directed edges (-->)
                        if '-->' in edge_type or 'x-x' in edge_type:
                            # Edge from j to i with lag tau
                            G.add_edge(
                                var_names[j],
                                var_names[i],
                                lag=tau,
                                pval=val_matrix[i, j, tau] if val_matrix is not None else None,
                                edge_type=edge_type
                            )

        return G

    def _generate_summary(
        self,
        graph: np.ndarray,
        var_names: List[str],
        val_matrix: np.ndarray
    ) -> str:
        """Generate human-readable summary of causal discovery."""
        n_vars = graph.shape[0]
        n_edges = 0

        summary_lines = [
            "PCMCI Causal Discovery Summary",
            "=" * 50,
            f"Variables: {n_vars}",
            f"Max lag: {self.tau_max}",
            f"PC alpha: {self.pc_alpha}",
            f"Alpha level: {self.alpha_level}",
            "",
            "Discovered Causal Edges:",
            "-" * 50
        ]

        # Count and list edges
        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(self.tau_max + 1):
                    edge_type = graph[i, j, tau]
                    if '-->' in str(edge_type) or 'x-x' in str(edge_type):
                        n_edges += 1
                        pval = val_matrix[i, j, tau] if val_matrix is not None else 0.0
                        lag_str = f"(t-{tau})" if tau > 0 else "(t)"
                        summary_lines.append(
                            f"{var_names[j]}{lag_str} --> {var_names[i]}(t) "
                            f"[p={pval:.4f}]"
                        )

        summary_lines.insert(8, f"Total edges: {n_edges}")
        summary_lines.append("")
        summary_lines.append("=" * 50)

        return "\n".join(summary_lines)

    def integrate_with_services(
        self,
        causal_graph: nx.DiGraph,
        service_mapping: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Integrate causal graph with service-level RCA.

        Args:
            causal_graph: PCMCI discovered graph
            service_mapping: Mapping from services to metrics
                            e.g., {'service-a': ['cpu', 'memory'], ...}

        Returns:
            Service-level causal scores (higher = more likely root cause)
        """
        # Build reverse mapping: metric -> service
        metric_to_service = {}
        for service, metrics in service_mapping.items():
            for metric in metrics:
                metric_to_service[metric] = service

        # Initialize service scores
        service_scores = {service: 0.0 for service in service_mapping.keys()}

        # Compute out-degree (number of outgoing causal edges) for each metric
        metric_out_degree = {}
        for node in causal_graph.nodes():
            metric_out_degree[node] = causal_graph.out_degree(node)

        # Compute in-degree (number of incoming causal edges) for each metric
        metric_in_degree = {}
        for node in causal_graph.nodes():
            metric_in_degree[node] = causal_graph.in_degree(node)

        # Aggregate to service level
        # Root cause services have high out-degree (cause many effects)
        # and low in-degree (not caused by others)
        for metric, out_deg in metric_out_degree.items():
            if metric in metric_to_service:
                service = metric_to_service[metric]
                in_deg = metric_in_degree[metric]

                # Causal score = out_degree - 0.5 * in_degree
                # (Penalize being an effect, reward being a cause)
                causal_score = out_deg - 0.5 * in_deg
                service_scores[service] += causal_score

        # Normalize scores to [0, 1]
        max_score = max(service_scores.values()) if service_scores else 1.0
        if max_score > 0:
            service_scores = {
                service: score / max_score
                for service, score in service_scores.items()
            }

        return service_scores


class GrangerLassoRCA:
    """
    Granger-Lasso baseline for comparison.

    Faster than PCMCI but less powerful:
    - Linear relationships only
    - No conditional independence testing
    - Good for quick baseline

    Reference: causal-learn library
    """

    def __init__(self, max_lag: int = 5, alpha: float = 0.01):
        self.max_lag = max_lag
        self.alpha = alpha

    def discover_graph(
        self,
        data: np.ndarray,
        var_names: Optional[List[str]] = None
    ) -> nx.DiGraph:
        """
        Discover Granger causal graph using Lasso regression.

        Args:
            data: (n_timesteps, n_variables) time series array
            var_names: Variable names

        Returns:
            NetworkX DiGraph with causal edges
        """
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        import warnings as warn

        n_timesteps, n_vars = data.shape

        if var_names is None:
            var_names = [f"var_{i}" for i in range(n_vars)]

        # Limit features for computational efficiency
        if n_vars > 100:
            print(f"  ⚠ Warning: {n_vars} variables detected. Granger-Lasso may be slow.")
            print(f"  Consider using PCMCI instead for large-scale causal discovery.")

        # Standardize data
        scaler = StandardScaler()
        data_std = scaler.fit_transform(data)

        # Build causal graph
        G = nx.DiGraph()
        for var_name in var_names:
            G.add_node(var_name)

        # Suppress sklearn convergence warnings (we use high max_iter)
        with warn.catch_warnings():
            warn.filterwarnings('ignore', category=UserWarning)
            warn.filterwarnings('ignore', category=FutureWarning)
            warn.simplefilter('ignore')

            # For each variable, fit Lasso regression
            for target_idx in range(n_vars):
                # Create lagged features
                X_lagged = []
                for lag in range(1, self.max_lag + 1):
                    # Slice to ensure all arrays have same length (n - max_lag)
                    X_lagged.append(data_std[self.max_lag - lag : -lag, :])

                # Stack lagged features: current + lag-1 + lag-2 + ... + lag-max_lag
                X = np.hstack([data_std[self.max_lag:, :]] + X_lagged[::-1])
                y = data_std[self.max_lag:, target_idx]

                # Fit Lasso with increased max_iter to avoid convergence warnings
                model = Lasso(alpha=self.alpha, max_iter=50000, tol=1e-4)
                model.fit(X, y)

                # Extract non-zero coefficients
                coefs = model.coef_
                for lag in range(self.max_lag + 1):
                    for source_idx in range(n_vars):
                        coef_idx = source_idx + lag * n_vars
                        if abs(coefs[coef_idx]) > 1e-6:
                            # Add edge from source to target
                            G.add_edge(
                                var_names[source_idx],
                                var_names[target_idx],
                                lag=lag,
                                weight=abs(coefs[coef_idx])
                            )

        return G


# Helper functions

def discover_causal_relations(
    metrics_data: np.ndarray,
    var_names: Optional[List[str]] = None,
    method: str = 'pcmci',
    **kwargs
) -> Dict:
    """
    Convenience function to discover causal relations.

    Args:
        metrics_data: (n_timesteps, n_variables) time series
        var_names: Variable names
        method: 'pcmci' or 'granger'
        **kwargs: Additional arguments for discovery method

    Returns:
        Dictionary with causal graph and summary
    """
    if method == 'pcmci':
        discoverer = PCMCIDiscovery(**kwargs)
        return discoverer.discover_graph(metrics_data, var_names)
    elif method == 'granger':
        discoverer = GrangerLassoRCA(**kwargs)
        causal_graph = discoverer.discover_graph(metrics_data, var_names)
        return {
            'causal_graph': causal_graph,
            'summary': f"Granger-Lasso: {causal_graph.number_of_edges()} edges"
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_causal_graph(
    causal_graph: nx.DiGraph,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    node_color: str = 'lightblue',
    edge_color: str = 'gray',
    show_edge_labels: bool = True
):
    """
    Visualize causal graph with matplotlib.

    Args:
        causal_graph: NetworkX DiGraph
        output_path: Path to save figure (optional)
        figsize: Figure size
        node_color: Node color
        edge_color: Edge color
        show_edge_labels: Show edge labels (lag, p-value)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Cannot visualize.")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    try:
        pos = nx.spring_layout(causal_graph, k=2, iterations=50)
    except:
        pos = nx.circular_layout(causal_graph)

    # Draw nodes
    nx.draw_networkx_nodes(
        causal_graph,
        pos,
        node_color=node_color,
        node_size=2000,
        alpha=0.8,
        ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(
        causal_graph,
        pos,
        edge_color=edge_color,
        arrows=True,
        arrowsize=20,
        width=2,
        alpha=0.6,
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        causal_graph,
        pos,
        font_size=10,
        font_weight='bold',
        ax=ax
    )

    # Draw edge labels
    if show_edge_labels:
        edge_labels = {}
        for u, v, data in causal_graph.edges(data=True):
            lag = data.get('lag', 0)
            pval = data.get('pval', None)
            if pval is not None:
                edge_labels[(u, v)] = f"lag={lag}\np={pval:.3f}"
            else:
                edge_labels[(u, v)] = f"lag={lag}"

        nx.draw_networkx_edge_labels(
            causal_graph,
            pos,
            edge_labels,
            font_size=8,
            ax=ax
        )

    ax.set_title("Causal Graph", fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Causal graph saved to {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_causal_paths(
    causal_graph: nx.DiGraph,
    source: str,
    target: str,
    max_paths: int = 5
) -> List[List[str]]:
    """
    Find causal paths from source to target.

    Args:
        causal_graph: NetworkX DiGraph
        source: Source node
        target: Target node
        max_paths: Maximum number of paths to return

    Returns:
        List of paths (each path is a list of nodes)
    """
    try:
        paths = list(nx.all_simple_paths(causal_graph, source, target))
        # Sort by path length
        paths.sort(key=len)
        return paths[:max_paths]
    except nx.NetworkXNoPath:
        return []


def compute_causal_strength(
    causal_graph: nx.DiGraph,
    source: str,
    target: str
) -> float:
    """
    Compute causal strength from source to target.

    Uses sum of inverse path lengths (shorter paths = stronger causality).

    Args:
        causal_graph: NetworkX DiGraph
        source: Source node
        target: Target node

    Returns:
        Causal strength score
    """
    paths = analyze_causal_paths(causal_graph, source, target, max_paths=10)

    if not paths:
        return 0.0

    # Sum of 1 / path_length for all paths
    strength = sum(1.0 / len(path) for path in paths)

    return strength
