"""
Visualization Utilities for Multimodal RCA

Publication-quality visualizations for:
- Metrics: Time series, heatmaps, correlation matrices
- Logs: Template distributions, error patterns, timelines
- Traces: Service dependency graphs, call latency distributions
- Results: Confusion matrices, ranking performance, ablation studies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import networkx as nx
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class MetricsVisualizer:
    """Visualize time series metrics data"""

    def __init__(self, figsize: Tuple[int, int] = (15, 8), dpi: int = 300):
        """
        Initialize metrics visualizer

        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def plot_time_series(
        self,
        metrics: pd.DataFrame,
        highlight_anomalies: Optional[List[int]] = None,
        ground_truth_service: Optional[str] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot time series for multiple metrics

        Args:
            metrics: DataFrame with metrics (columns = metrics, rows = timesteps)
            highlight_anomalies: List of timestep indices to highlight
            ground_truth_service: Root cause service name (for title)
            save_path: Path to save figure
        """
        n_metrics = min(len(metrics.columns), 9)  # Max 9 subplots
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for i, col in enumerate(metrics.columns[:n_metrics]):
            ax = axes[i]
            series = metrics[col]

            # Plot time series
            ax.plot(series.index, series.values, linewidth=1.5, label=col)

            # Highlight anomaly region
            if highlight_anomalies:
                for idx in highlight_anomalies:
                    ax.axvspan(idx-2, idx+2, alpha=0.3, color='red')

            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        if ground_truth_service:
            fig.suptitle(f'Metrics Time Series - Root Cause: {ground_truth_service}',
                        fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_correlation_matrix(
        self,
        metrics: pd.DataFrame,
        method: str = 'pearson',
        save_path: Optional[Path] = None
    ):
        """
        Plot correlation matrix for metrics

        Args:
            metrics: DataFrame with metrics
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_path: Path to save figure
        """
        # Calculate correlation
        corr = metrics.corr(method=method)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            corr,
            cmap='coolwarm',
            center=0,
            annot=False,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_anomaly_heatmap(
        self,
        metrics: pd.DataFrame,
        anomaly_scores: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """
        Plot heatmap of anomaly scores

        Args:
            metrics: DataFrame with metrics
            anomaly_scores: Array of shape (timesteps, metrics) with anomaly scores
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            anomaly_scores.T,
            cmap='YlOrRd',
            xticklabels=metrics.index[::10],  # Show every 10th timestep
            yticklabels=metrics.columns,
            cbar_kws={'label': 'Anomaly Score'},
            ax=ax
        )

        ax.set_title('Anomaly Score Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Metric')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class LogsVisualizer:
    """Visualize log data and patterns"""

    def __init__(self, figsize: Tuple[int, int] = (15, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("whitegrid")

    def plot_log_template_distribution(
        self,
        templates: List[str],
        counts: List[int],
        top_n: int = 20,
        save_path: Optional[Path] = None
    ):
        """
        Plot distribution of log templates

        Args:
            templates: List of log template strings
            counts: List of occurrence counts
            top_n: Number of top templates to show
            save_path: Path to save figure
        """
        # Get top N templates
        template_counts = list(zip(templates, counts))
        template_counts.sort(key=lambda x: x[1], reverse=True)
        top_templates = template_counts[:top_n]

        templates, counts = zip(*top_templates)

        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=self.figsize)

        y_pos = np.arange(len(templates))
        ax.barh(y_pos, counts, color='steelblue')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[:60] + '...' if len(t) > 60 else t for t in templates], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Occurrence Count', fontsize=12)
        ax.set_title(f'Top {top_n} Log Templates', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_log_level_distribution(
        self,
        level_counts: Dict[str, int],
        save_path: Optional[Path] = None
    ):
        """
        Plot distribution of log levels

        Args:
            level_counts: Dict mapping log level to count
            save_path: Path to save figure
        """
        levels = list(level_counts.keys())
        counts = list(level_counts.values())

        # Define colors for log levels
        level_colors = {
            'DEBUG': '#4CAF50',
            'INFO': '#2196F3',
            'WARNING': '#FF9800',
            'ERROR': '#F44336',
            'CRITICAL': '#9C27B0'
        }
        colors = [level_colors.get(level, 'gray') for level in levels]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(levels, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Log Level', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Log Level Distribution', fontsize=14, fontweight='bold')

        # Add count labels on bars
        for i, (level, count) in enumerate(zip(levels, counts)):
            ax.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_log_timeline(
        self,
        timestamps: pd.Series,
        levels: pd.Series,
        save_path: Optional[Path] = None
    ):
        """
        Plot log timeline showing log volume over time

        Args:
            timestamps: Series of log timestamps
            levels: Series of log levels
            save_path: Path to save figure
        """
        # Create time bins
        log_df = pd.DataFrame({'timestamp': timestamps, 'level': levels})
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

        # Group by time bins
        log_df['time_bin'] = log_df['timestamp'].dt.floor('1min')
        timeline = log_df.groupby(['time_bin', 'level']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=self.figsize)

        timeline.plot(kind='area', stacked=True, ax=ax, alpha=0.7)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Log Count', fontsize=12)
        ax.set_title('Log Volume Timeline', fontsize=14, fontweight='bold')
        ax.legend(title='Log Level', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class TracesVisualizer:
    """Visualize distributed traces and service dependencies"""

    def __init__(self, figsize: Tuple[int, int] = (15, 12), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi

    def plot_service_dependency_graph(
        self,
        adjacency_matrix: np.ndarray,
        service_names: List[str],
        root_cause_service: Optional[str] = None,
        node_scores: Optional[Dict[str, float]] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot service dependency graph

        Args:
            adjacency_matrix: Adjacency matrix (i,j)=1 if i calls j
            service_names: List of service names
            root_cause_service: True root cause service (highlighted in red)
            node_scores: Optional dict of service -> anomaly score
            save_path: Path to save figure
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for service in service_names:
            G.add_node(service)

        # Add edges
        for i, src in enumerate(service_names):
            for j, dst in enumerate(service_names):
                if adjacency_matrix[i, j] > 0:
                    G.add_edge(src, dst, weight=adjacency_matrix[i, j])

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Node colors
        node_colors = []
        for node in G.nodes():
            if node == root_cause_service:
                node_colors.append('#F44336')  # Red for root cause
            elif node_scores and node in node_scores:
                # Color by anomaly score
                score = node_scores[node]
                intensity = min(score / max(node_scores.values()), 1.0)
                node_colors.append(plt.cm.YlOrRd(intensity))
            else:
                node_colors.append('#90CAF9')  # Light blue default

        # Node sizes (by degree)
        node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=20,
            width=2,
            ax=ax
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        ax.set_title('Service Dependency Graph', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_latency_distribution(
        self,
        latencies: Dict[str, List[float]],
        save_path: Optional[Path] = None
    ):
        """
        Plot latency distribution for services

        Args:
            latencies: Dict mapping service name to list of latencies
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data for violin plot
        services = list(latencies.keys())
        data = [latencies[s] for s in services]

        # Violin plot
        parts = ax.violinplot(data, positions=range(len(services)), showmeans=True, showmedians=True)

        ax.set_xticks(range(len(services)))
        ax.set_xticklabels(services, rotation=45, ha='right')
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Service Latency Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class ResultsVisualizer:
    """Visualize model results and comparisons"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("whitegrid")

    def plot_method_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['AC@1', 'AC@3', 'AC@5', 'MRR'],
        save_path: Optional[Path] = None
    ):
        """
        Plot comparison of different methods

        Args:
            results: Dict mapping method name to dict of metrics
            metrics: List of metric names to plot
            save_path: Path to save figure
        """
        # Prepare data
        methods = list(results.keys())
        data = []

        for metric in metrics:
            for method in methods:
                value = results[method].get(metric, 0.0)
                data.append({
                    'Method': method,
                    'Metric': metric,
                    'Value': value
                })

        df = pd.DataFrame(data)

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(metrics))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            values = [method_data[method_data['Metric'] == m]['Value'].values[0] for m in metrics]
            ax.bar(x + i * width, values, width, label=method, alpha=0.8)

        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Method Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_ablation_study(
        self,
        ablation_results: Dict[str, float],
        baseline_score: float,
        metric_name: str = 'AC@1',
        save_path: Optional[Path] = None
    ):
        """
        Plot ablation study results

        Args:
            ablation_results: Dict mapping ablation name to score
            baseline_score: Full model score
            metric_name: Name of metric being shown
            save_path: Path to save figure
        """
        # Add baseline
        all_results = {'Full Model': baseline_score}
        all_results.update(ablation_results)

        # Sort by score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        configs, scores = zip(*sorted_results)

        # Plot
        fig, ax = plt.subplots(figsize=(10, max(8, len(configs) * 0.5)))

        y_pos = np.arange(len(configs))
        colors = ['#4CAF50' if c == 'Full Model' else '#2196F3' for c in configs]

        ax.barh(y_pos, scores, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(configs)
        ax.invert_yaxis()
        ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'Ablation Study - {metric_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, score in enumerate(scores):
            ax.text(score, i, f' {score:.3f}', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        service_names: List[str],
        save_path: Optional[Path] = None
    ):
        """
        Plot confusion matrix for root cause prediction

        Args:
            y_true: True root cause services
            y_pred: Predicted root cause services
            service_names: List of all service names
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred, labels=service_names)

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=service_names,
            yticklabels=service_names,
            ax=ax
        )

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Root Cause Localization', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# Example usage
if __name__ == '__main__':
    print("Testing visualization utilities...")

    # Generate synthetic data
    np.random.seed(42)

    # 1. Metrics visualization
    print("\n1. Testing MetricsVisualizer...")
    metrics_data = pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f'metric_{i}' for i in range(5)]
    )
    metrics_viz = MetricsVisualizer()
    metrics_viz.plot_time_series(metrics_data, highlight_anomalies=[80, 85, 90])

    # 2. Logs visualization
    print("\n2. Testing LogsVisualizer...")
    logs_viz = LogsVisualizer()
    templates = [f'Template {i}' for i in range(20)]
    counts = np.random.randint(10, 1000, 20).tolist()
    logs_viz.plot_log_template_distribution(templates, counts)

    # 3. Traces visualization
    print("\n3. Testing TracesVisualizer...")
    traces_viz = TracesVisualizer()
    services = ['frontend', 'api-gateway', 'auth', 'db', 'cache']
    adj_matrix = np.random.rand(5, 5) > 0.7
    traces_viz.plot_service_dependency_graph(adj_matrix, services, root_cause_service='db')

    # 4. Results visualization
    print("\n4. Testing ResultsVisualizer...")
    results_viz = ResultsVisualizer()
    results = {
        'Method A': {'AC@1': 0.75, 'AC@3': 0.85, 'AC@5': 0.90, 'MRR': 0.82},
        'Method B': {'AC@1': 0.65, 'AC@3': 0.78, 'AC@5': 0.85, 'MRR': 0.75},
        'Method C': {'AC@1': 0.80, 'AC@3': 0.88, 'AC@5': 0.92, 'MRR': 0.85}
    }
    results_viz.plot_method_comparison(results)

    print("\n✅ All visualization utilities tested successfully!")
