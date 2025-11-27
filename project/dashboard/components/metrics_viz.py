"""
================================================================================
METRICS & TIME SERIES VISUALIZATION COMPONENTS
================================================================================

Real-time metrics visualization for the RCA dashboard.
Features: Multi-service metrics, anomaly detection, interactive zooming.

Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# Color palette
COLORS = {
    'primary': '#6366f1',
    'secondary': '#10b981',
    'accent': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'purple': '#8b5cf6',
    'pink': '#ec4899',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'bg': 'rgba(0,0,0,0)',
    'grid': 'rgba(255,255,255,0.1)'
}

SERVICE_COLORS = px.colors.qualitative.Set3


def create_multi_metric_timeline(
    metrics_data: Dict[str, np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    metric_names: List[str] = ['CPU %', 'Memory %', 'Network I/O', 'Latency (ms)'],
    anomaly_window: Optional[Tuple[int, int]] = None,
    height: int = 600
) -> go.Figure:
    """
    Create a multi-panel time series visualization for service metrics.
    
    Args:
        metrics_data: Dict mapping service name to (timesteps, features) array
        timestamps: Optional timestamp array
        metric_names: Names of the metrics to display
        anomaly_window: Optional (start, end) indices for anomaly highlighting
        height: Figure height
        
    Returns:
        Plotly figure with subplots for each metric
    """
    services = list(metrics_data.keys())
    n_metrics = min(len(metric_names), 4)
    
    # Generate timestamps if not provided
    if timestamps is None:
        max_len = max(arr.shape[0] for arr in metrics_data.values())
        timestamps = np.arange(max_len)
    
    # Create subplots
    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=metric_names[:n_metrics]
    )
    
    # Add traces for each service and metric
    for service_idx, (service, data) in enumerate(metrics_data.items()):
        color = SERVICE_COLORS[service_idx % len(SERVICE_COLORS)]
        
        for metric_idx in range(min(n_metrics, data.shape[1] if len(data.shape) > 1 else 1)):
            if len(data.shape) > 1:
                values = data[:, metric_idx]
            else:
                values = data
            
            # Normalize to reasonable range
            values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps[:len(values)],
                    y=values,
                    name=service,
                    legendgroup=service,
                    showlegend=(metric_idx == 0),
                    line=dict(color=color, width=1.5),
                    hovertemplate=f'{service}<br>{metric_names[metric_idx]}: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>'
                ),
                row=metric_idx + 1,
                col=1
            )
    
    # Add anomaly window shading
    if anomaly_window is not None:
        start, end = anomaly_window
        for row in range(1, n_metrics + 1):
            fig.add_vrect(
                x0=timestamps[start],
                x1=timestamps[min(end, len(timestamps)-1)],
                fillcolor='rgba(239, 68, 68, 0.15)',
                line_width=0,
                row=row,
                col=1
            )
            # Add annotation for anomaly window
            if row == 1:
                fig.add_annotation(
                    x=timestamps[(start + end) // 2],
                    y=1.1,
                    yref='paper',
                    text="⚠️ Anomaly Window",
                    showarrow=False,
                    font=dict(color='#ef4444', size=12)
                )
    
    # Update layout
    fig.update_layout(
        height=height,
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(color=COLORS['text'], size=10),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=60, r=20, t=40, b=80),
        hovermode='x unified'
    )
    
    # Update axes styling
    for i in range(1, n_metrics + 1):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text_secondary']),
            row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text_secondary']),
            title=dict(font=dict(color=COLORS['text_secondary'], size=10)),
            row=i, col=1
        )
    
    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.color = COLORS['text']
        annotation.font.size = 12
    
    return fig


def create_anomaly_detection_chart(
    values: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    threshold_upper: Optional[float] = None,
    threshold_lower: Optional[float] = None,
    service_name: str = "Service",
    metric_name: str = "Metric"
) -> go.Figure:
    """
    Create a chart showing anomaly detection with thresholds.
    
    Args:
        values: Time series values
        timestamps: Optional timestamps
        threshold_upper: Upper threshold for anomaly detection
        threshold_lower: Lower threshold for anomaly detection
        service_name: Name of the service
        metric_name: Name of the metric
        
    Returns:
        Plotly figure with anomaly visualization
    """
    if timestamps is None:
        timestamps = np.arange(len(values))
    
    # Calculate thresholds if not provided
    if threshold_upper is None:
        threshold_upper = np.mean(values) + 2 * np.std(values)
    if threshold_lower is None:
        threshold_lower = np.mean(values) - 2 * np.std(values)
    
    # Identify anomalies
    anomalies = (values > threshold_upper) | (values < threshold_lower)
    anomaly_indices = np.where(anomalies)[0]
    
    fig = go.Figure()
    
    # Add normal range shading
    fig.add_trace(go.Scatter(
        x=np.concatenate([timestamps, timestamps[::-1]]),
        y=np.concatenate([
            np.full_like(timestamps, threshold_upper, dtype=float),
            np.full_like(timestamps, threshold_lower, dtype=float)
        ]),
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.1)',
        line=dict(width=0),
        hoverinfo='skip',
        name='Normal Range',
        showlegend=True
    ))
    
    # Add main time series
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines',
        name=metric_name,
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate=f'{metric_name}: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_hline(
        y=threshold_upper,
        line_dash='dash',
        line_color=COLORS['accent'],
        annotation_text='Upper Threshold',
        annotation_font=dict(color=COLORS['accent'])
    )
    fig.add_hline(
        y=threshold_lower,
        line_dash='dash',
        line_color=COLORS['accent'],
        annotation_text='Lower Threshold',
        annotation_font=dict(color=COLORS['accent'])
    )
    
    # Add anomaly markers
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=timestamps[anomaly_indices],
            y=values[anomaly_indices],
            mode='markers',
            name='Anomalies',
            marker=dict(
                color=COLORS['danger'],
                size=10,
                symbol='x',
                line=dict(width=2, color='white')
            ),
            hovertemplate='⚠️ Anomaly<br>Value: %{y:.2f}<br>Time: %{x}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{service_name} - {metric_name}</b>',
            font=dict(color=COLORS['text'], size=14)
        ),
        xaxis=dict(
            title='Time',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=True,
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title=metric_name,
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=True,
            gridcolor=COLORS['grid']
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        legend=dict(
            font=dict(color=COLORS['text']),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=350,
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_log_frequency_chart(
    log_counts: Dict[str, np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    top_n: int = 5
) -> go.Figure:
    """
    Create a stacked area chart showing log template frequencies.
    
    Args:
        log_counts: Dict mapping template names to count arrays
        timestamps: Optional timestamps
        top_n: Number of top templates to show
        
    Returns:
        Plotly figure with log frequency visualization
    """
    if not log_counts:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No log data available",
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            showarrow=False,
            font=dict(color=COLORS['text_secondary'], size=14)
        )
        fig.update_layout(
            plot_bgcolor=COLORS['bg'],
            paper_bgcolor=COLORS['bg'],
            height=300
        )
        return fig
    
    # Get top templates by total count
    total_counts = {k: v.sum() for k, v in log_counts.items()}
    top_templates = sorted(total_counts.keys(), key=lambda x: total_counts[x], reverse=True)[:top_n]
    
    # Generate timestamps
    if timestamps is None:
        max_len = max(len(v) for v in log_counts.values())
        timestamps = np.arange(max_len)
    
    fig = go.Figure()
    
    colors = px.colors.sequential.Viridis
    
    for i, template in enumerate(top_templates):
        values = log_counts[template]
        fig.add_trace(go.Scatter(
            x=timestamps[:len(values)],
            y=values,
            name=template[:30] + '...' if len(template) > 30 else template,
            mode='lines',
            stackgroup='one',
            fillcolor=colors[i % len(colors)].replace(')', ', 0.6)').replace('rgb', 'rgba'),
            line=dict(width=0.5, color=colors[i % len(colors)]),
            hovertemplate=f'{template[:20]}...<br>Count: %{{y}}<br>Time: %{{x}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>Log Template Frequency Over Time</b>',
            font=dict(color=COLORS['text'], size=14)
        ),
        xaxis=dict(
            title='Time',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=True,
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title='Count',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=True,
            gridcolor=COLORS['grid']
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        legend=dict(
            font=dict(color=COLORS['text'], size=9),
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        height=350,
        margin=dict(l=60, r=20, t=50, b=80)
    )
    
    return fig


def create_trace_latency_heatmap(
    latency_data: np.ndarray,
    services: List[str],
    methods: Optional[List[str]] = None,
    timestamps: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Create a heatmap showing trace latencies across services.
    
    Args:
        latency_data: (services, timestamps) or (services, methods, timestamps) array
        services: List of service names
        methods: Optional list of method names
        timestamps: Optional timestamp labels
        
    Returns:
        Plotly figure with latency heatmap
    """
    # Aggregate to (services, timestamps) if needed
    if len(latency_data.shape) == 3:
        latency_data = latency_data.mean(axis=1)  # Average across methods
    
    if timestamps is None:
        timestamps = [f't{i}' for i in range(latency_data.shape[1])]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=latency_data,
        x=timestamps,
        y=services,
        colorscale=[
            [0, 'rgba(16, 185, 129, 0.2)'],
            [0.3, 'rgba(99, 102, 241, 0.5)'],
            [0.6, 'rgba(245, 158, 11, 0.7)'],
            [1, 'rgba(239, 68, 68, 1)']
        ],
        colorbar=dict(
            title='Latency (ms)',
            titlefont=dict(color=COLORS['text']),
            tickfont=dict(color=COLORS['text'])
        ),
        hovertemplate='Service: %{y}<br>Time: %{x}<br>Latency: %{z:.1f}ms<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Service Latency Heatmap</b>',
            font=dict(color=COLORS['text'], size=14)
        ),
        xaxis=dict(
            title='Time',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=False
        ),
        yaxis=dict(
            title='Service',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=False
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        height=400,
        margin=dict(l=120, r=20, t=50, b=60)
    )
    
    return fig


def create_service_health_dashboard(
    services: List[str],
    health_scores: np.ndarray,
    metrics_summary: Optional[Dict[str, Dict[str, float]]] = None
) -> go.Figure:
    """
    Create a service health overview dashboard.
    
    Args:
        services: List of service names
        health_scores: Array of health scores (0-1) per service
        metrics_summary: Optional dict with service metrics summaries
        
    Returns:
        Plotly figure with health indicators
    """
    n_services = len(services)
    
    # Create grid layout
    n_cols = min(5, n_services)
    n_rows = (n_services + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{'type': 'indicator'} for _ in range(n_cols)] for _ in range(n_rows)],
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )
    
    for i, (service, score) in enumerate(zip(services, health_scores)):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Determine color based on score
        if score >= 0.8:
            color = COLORS['secondary']  # Green
        elif score >= 0.5:
            color = COLORS['accent']  # Orange
        else:
            color = COLORS['danger']  # Red
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={'suffix': '%', 'font': {'color': COLORS['text'], 'size': 20}},
                title={'text': service[:15], 'font': {'color': COLORS['text_secondary'], 'size': 11}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': COLORS['text_secondary'], 'size': 8}},
                    'bar': {'color': color},
                    'bgcolor': 'rgba(30, 41, 59, 0.8)',
                    'borderwidth': 1,
                    'bordercolor': COLORS['grid'],
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.1)'},
                        {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ]
                }
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=150 * n_rows + 50,
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig


def create_correlation_matrix(
    services: List[str],
    correlation_matrix: np.ndarray
) -> go.Figure:
    """
    Create a correlation matrix visualization.
    
    Args:
        services: List of service names
        correlation_matrix: (n_services, n_services) correlation matrix
        
    Returns:
        Plotly figure with correlation heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=services,
        y=services,
        colorscale=[
            [0, 'rgba(59, 130, 246, 1)'],      # Blue for negative
            [0.5, 'rgba(30, 41, 59, 0.8)'],     # Dark for zero
            [1, 'rgba(239, 68, 68, 1)']         # Red for positive
        ],
        zmid=0,
        colorbar=dict(
            title='Correlation',
            titlefont=dict(color=COLORS['text']),
            tickfont=dict(color=COLORS['text'])
        ),
        hovertemplate='%{y} ↔ %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    # Add diagonal line effect
    for i in range(len(services)):
        fig.add_annotation(
            x=services[i],
            y=services[i],
            text='•',
            showarrow=False,
            font=dict(color='white', size=20)
        )
    
    fig.update_layout(
        title=dict(
            text='<b>Service Metric Correlation</b>',
            font=dict(color=COLORS['text'], size=14)
        ),
        xaxis=dict(
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            tickangle=45,
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['text_secondary'], size=10),
            showgrid=False
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        height=450,
        margin=dict(l=100, r=20, t=50, b=100)
    )
    
    return fig


def generate_demo_metrics(
    services: List[str],
    n_timesteps: int = 100,
    anomaly_service_idx: int = 0,
    anomaly_start: int = 60
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Generate demo metrics data for visualization testing.
    
    Args:
        services: List of service names
        n_timesteps: Number of timesteps to generate
        anomaly_service_idx: Index of service with anomaly
        anomaly_start: Timestep where anomaly starts
        
    Returns:
        Tuple of (metrics_data dict, timestamps array)
    """
    np.random.seed(42)
    timestamps = np.arange(n_timesteps)
    metrics_data = {}
    
    for i, service in enumerate(services):
        # Base patterns
        base_cpu = 30 + 10 * np.sin(timestamps * 0.1) + np.random.randn(n_timesteps) * 3
        base_mem = 50 + 5 * np.cos(timestamps * 0.05) + np.random.randn(n_timesteps) * 2
        base_net = 100 + 20 * np.sin(timestamps * 0.15) + np.random.randn(n_timesteps) * 10
        base_lat = 50 + 10 * np.random.randn(n_timesteps)
        
        # Add anomaly to specific service
        if i == anomaly_service_idx:
            base_cpu[anomaly_start:] += 40 + np.random.randn(n_timesteps - anomaly_start) * 10
            base_lat[anomaly_start:] += 100 + np.random.randn(n_timesteps - anomaly_start) * 20
        
        # Add small correlated anomalies to nearby services
        elif abs(i - anomaly_service_idx) <= 2:
            delay = abs(i - anomaly_service_idx) * 3
            if anomaly_start + delay < n_timesteps:
                base_lat[anomaly_start + delay:] += 20 + np.random.randn(n_timesteps - anomaly_start - delay) * 5
        
        metrics_data[service] = np.column_stack([
            np.clip(base_cpu, 0, 100),
            np.clip(base_mem, 0, 100),
            np.clip(base_net, 0, None),
            np.clip(base_lat, 0, None)
        ]).astype(np.float32)
    
    return metrics_data, timestamps
