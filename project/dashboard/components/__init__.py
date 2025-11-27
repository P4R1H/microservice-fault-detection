"""
Dashboard Components Module

Interactive visualization components for the Multimodal RCA Dashboard.
"""

from .causal_graph import (
    CausalGraphVisualization,
    GraphNode,
    GraphEdge,
    create_3d_causal_graph,
    create_animated_propagation,
    render_causal_graph_widget
)

from .metrics_viz import (
    create_multi_metric_timeline,
    create_anomaly_detection_chart,
    create_log_frequency_chart,
    create_trace_latency_heatmap,
    create_service_health_dashboard,
    create_correlation_matrix,
    generate_demo_metrics
)

from .explanations import (
    create_attention_heatmap,
    create_modality_importance_chart,
    create_modality_bar_chart,
    create_prediction_breakdown,
    create_feature_importance_waterfall,
    create_confidence_gauge,
    create_explanation_summary,
    create_llm_explanation_card,
    generate_demo_explanations
)

__all__ = [
    # Causal graph
    'CausalGraphVisualization',
    'GraphNode', 
    'GraphEdge',
    'create_3d_causal_graph',
    'create_animated_propagation',
    'render_causal_graph_widget',
    
    # Metrics
    'create_multi_metric_timeline',
    'create_anomaly_detection_chart',
    'create_log_frequency_chart',
    'create_trace_latency_heatmap',
    'create_service_health_dashboard',
    'create_correlation_matrix',
    'generate_demo_metrics',
    
    # Explanations
    'create_attention_heatmap',
    'create_modality_importance_chart',
    'create_modality_bar_chart',
    'create_prediction_breakdown',
    'create_feature_importance_waterfall',
    'create_confidence_gauge',
    'create_explanation_summary',
    'create_llm_explanation_card',
    'generate_demo_explanations'
]
