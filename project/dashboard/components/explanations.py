"""
================================================================================
MODEL EXPLANATION VISUALIZATION COMPONENTS
================================================================================

Explainability visualizations for the Multimodal RCA model.
Features: Attention visualization, feature importance, prediction breakdown.

Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
Date: November 2025
"""

import numpy as np
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
    'purple': '#8b5cf6',
    'pink': '#ec4899',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'bg': 'rgba(0,0,0,0)',
    'grid': 'rgba(255,255,255,0.1)'
}


def create_attention_heatmap(
    attention_weights: np.ndarray,
    services: List[str],
    title: str = "Cross-Service Attention Weights"
) -> go.Figure:
    """
    Create a beautiful attention heatmap visualization.
    
    Args:
        attention_weights: (n_services, n_services) attention matrix
        services: List of service names
        title: Chart title
        
    Returns:
        Plotly figure with styled heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=services,
        y=services,
        colorscale=[
            [0, 'rgba(15, 23, 42, 0.9)'],
            [0.2, 'rgba(99, 102, 241, 0.3)'],
            [0.4, 'rgba(99, 102, 241, 0.5)'],
            [0.6, 'rgba(139, 92, 246, 0.7)'],
            [0.8, 'rgba(168, 85, 247, 0.85)'],
            [1, 'rgba(236, 72, 153, 1)']
        ],
        hovertemplate='<b>From:</b> %{y}<br><b>To:</b> %{x}<br><b>Attention:</b> %{z:.4f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(
                text='Attention<br>Weight',
                font=dict(color=COLORS['text'], size=11)
            ),
            tickfont=dict(color=COLORS['text'], size=10),
            thickness=15,
            len=0.8
        )
    ))
    
    # Add value annotations for high attention
    annotations = []
    for i, row in enumerate(attention_weights):
        for j, val in enumerate(row):
            if val > 0.15:  # Only annotate significant values
                annotations.append(dict(
                    x=services[j],
                    y=services[i],
                    text=f'{val:.2f}',
                    showarrow=False,
                    font=dict(
                        color='white' if val > 0.3 else COLORS['text_secondary'],
                        size=9
                    )
                ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(color=COLORS['text'], size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Target Service',
            titlefont=dict(color=COLORS['text_secondary'], size=11),
            tickfont=dict(color=COLORS['text'], size=10),
            tickangle=45,
            showgrid=False,
            side='bottom'
        ),
        yaxis=dict(
            title='Source Service',
            titlefont=dict(color=COLORS['text_secondary'], size=11),
            tickfont=dict(color=COLORS['text'], size=10),
            showgrid=False,
            autorange='reversed'
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        height=500,
        margin=dict(l=120, r=40, t=60, b=120),
        annotations=annotations
    )
    
    return fig


def create_modality_importance_chart(
    modality_weights: Dict[str, float],
    title: str = "Modality Contribution"
) -> go.Figure:
    """
    Create a beautiful radial/polar chart for modality importance.
    
    Args:
        modality_weights: Dict mapping modality name to importance weight
        title: Chart title
        
    Returns:
        Plotly figure with modality importance visualization
    """
    modalities = list(modality_weights.keys())
    weights = list(modality_weights.values())
    
    # Colors for each modality
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    
    fig = go.Figure()
    
    # Add polar bars
    fig.add_trace(go.Barpolar(
        r=weights,
        theta=modalities,
        width=[0.8] * len(modalities),
        marker=dict(
            color=colors[:len(modalities)],
            line=dict(color='white', width=2),
            opacity=0.85
        ),
        hovertemplate='<b>%{theta}</b><br>Importance: %{r:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(color=COLORS['text'], size=16),
            x=0.5,
            xanchor='center'
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(weights) * 1.2],
                tickfont=dict(color=COLORS['text_secondary'], size=10),
                gridcolor=COLORS['grid'],
                tickformat='.0%'
            ),
            angularaxis=dict(
                tickfont=dict(color=COLORS['text'], size=12),
                gridcolor=COLORS['grid'],
                linecolor=COLORS['grid']
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor=COLORS['bg'],
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=60, b=40)
    )
    
    return fig


def create_modality_bar_chart(
    modality_weights: Dict[str, float],
    title: str = "Modality Importance"
) -> go.Figure:
    """
    Create a horizontal bar chart for modality importance.
    
    Args:
        modality_weights: Dict mapping modality name to importance weight
        title: Chart title
        
    Returns:
        Plotly figure with bar chart
    """
    modalities = list(modality_weights.keys())
    weights = list(modality_weights.values())
    
    # Gradient colors
    colors = ['#6366f1', '#10b981', '#f59e0b']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=modalities,
        x=weights,
        orientation='h',
        marker=dict(
            color=colors[:len(modalities)],
            line=dict(color='white', width=1)
        ),
        text=[f'{w:.1%}' for w in weights],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=12),
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(color=COLORS['text'], size=14)
        ),
        xaxis=dict(
            title='Contribution',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=True,
            gridcolor=COLORS['grid'],
            range=[0, max(weights) * 1.3],
            tickformat='.0%'
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['text']),
            showgrid=False
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        height=250,
        margin=dict(l=100, r=60, t=50, b=40)
    )
    
    return fig


def create_prediction_breakdown(
    services: List[str],
    probabilities: np.ndarray,
    root_cause_idx: Optional[int] = None,
    top_k: int = 5
) -> go.Figure:
    """
    Create a prediction breakdown visualization showing ranked services.
    
    Args:
        services: List of service names
        probabilities: Prediction probabilities per service
        root_cause_idx: Index of actual root cause (for highlighting)
        top_k: Number of top services to show
        
    Returns:
        Plotly figure with prediction breakdown
    """
    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1][:top_k]
    sorted_services = [services[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    
    # Determine colors
    colors = []
    for i in sorted_indices:
        if root_cause_idx is not None and i == root_cause_idx:
            colors.append(COLORS['danger'])  # Red for actual root cause
        elif probabilities[i] > 0.5:
            colors.append(COLORS['purple'])  # Purple for high confidence
        elif probabilities[i] > 0.2:
            colors.append(COLORS['primary'])  # Blue for medium
        else:
            colors.append('#475569')  # Gray for low
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=sorted_probs,
        y=sorted_services,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=1),
            opacity=0.9
        ),
        text=[f'{p:.1%}' for p in sorted_probs],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11),
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
    ))
    
    # Add rank badges
    for i, (service, prob) in enumerate(zip(sorted_services, sorted_probs)):
        fig.add_annotation(
            x=-0.02,
            y=service,
            text=f'#{i+1}',
            showarrow=False,
            font=dict(
                color=COLORS['text_secondary'],
                size=10,
                family='monospace'
            ),
            xanchor='right'
        )
    
    # Add legend annotation for root cause
    if root_cause_idx is not None and root_cause_idx in sorted_indices:
        fig.add_annotation(
            x=1,
            y=1.1,
            xref='paper',
            yref='paper',
            text='🎯 = Actual Root Cause',
            showarrow=False,
            font=dict(color=COLORS['danger'], size=11),
            xanchor='right'
        )
    
    fig.update_layout(
        title=dict(
            text='<b>Root Cause Prediction Ranking</b>',
            font=dict(color=COLORS['text'], size=16)
        ),
        xaxis=dict(
            title='Probability',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=True,
            gridcolor=COLORS['grid'],
            range=[0, 1.1],
            tickformat='.0%'
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS['text'], size=11),
            showgrid=False,
            categoryorder='total ascending'
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        height=50 * top_k + 100,
        margin=dict(l=140, r=60, t=60, b=40)
    )
    
    return fig


def create_feature_importance_waterfall(
    feature_names: List[str],
    importance_values: np.ndarray,
    base_value: float = 0.5,
    title: str = "Feature Contribution to Prediction"
) -> go.Figure:
    """
    Create a waterfall chart showing feature contributions.
    
    Args:
        feature_names: Names of features
        importance_values: Contribution values (can be negative)
        base_value: Base prediction value
        title: Chart title
        
    Returns:
        Plotly figure with waterfall visualization
    """
    # Sort by absolute importance
    sorted_indices = np.argsort(np.abs(importance_values))[::-1][:10]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_values = importance_values[sorted_indices]
    
    # Calculate cumulative for waterfall
    cumulative = base_value + np.cumsum(sorted_values)
    
    colors = [COLORS['secondary'] if v > 0 else COLORS['danger'] for v in sorted_values]
    
    fig = go.Figure(go.Waterfall(
        name="Feature Contribution",
        orientation="v",
        measure=["relative"] * len(sorted_values) + ["total"],
        x=sorted_names + ["Final"],
        y=list(sorted_values) + [None],
        textposition="outside",
        text=[f'{v:+.3f}' for v in sorted_values] + [f'{cumulative[-1]:.3f}'],
        textfont=dict(color=COLORS['text'], size=10),
        connector=dict(line=dict(color=COLORS['grid'], width=1)),
        increasing=dict(marker=dict(color=COLORS['secondary'])),
        decreasing=dict(marker=dict(color=COLORS['danger'])),
        totals=dict(marker=dict(color=COLORS['purple']))
    ))
    
    # Add base value line
    fig.add_hline(
        y=base_value,
        line_dash='dash',
        line_color=COLORS['text_secondary'],
        annotation_text=f'Base: {base_value:.2f}',
        annotation_font=dict(color=COLORS['text_secondary'])
    )
    
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(color=COLORS['text'], size=14)
        ),
        xaxis=dict(
            tickfont=dict(color=COLORS['text_secondary'], size=9),
            tickangle=45
        ),
        yaxis=dict(
            title='Cumulative Score',
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary']),
            showgrid=True,
            gridcolor=COLORS['grid']
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        height=400,
        margin=dict(l=60, r=20, t=60, b=120),
        showlegend=False
    )
    
    return fig


def create_confidence_gauge(
    confidence: float,
    service_name: str,
    is_correct: Optional[bool] = None
) -> go.Figure:
    """
    Create a confidence gauge for a prediction.
    
    Args:
        confidence: Confidence score (0-1)
        service_name: Name of predicted service
        is_correct: Whether prediction is correct (for coloring)
        
    Returns:
        Plotly figure with gauge
    """
    # Determine color
    if is_correct is True:
        bar_color = COLORS['secondary']
    elif is_correct is False:
        bar_color = COLORS['danger']
    elif confidence > 0.7:
        bar_color = COLORS['purple']
    elif confidence > 0.4:
        bar_color = COLORS['accent']
    else:
        bar_color = COLORS['text_secondary']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        number={
            'suffix': '%',
            'font': {'color': COLORS['text'], 'size': 36}
        },
        title={
            'text': f'<b>{service_name}</b><br><span style="font-size:11px;color:{COLORS["text_secondary"]}">Predicted Root Cause</span>',
            'font': {'color': COLORS['text'], 'size': 14}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickfont': {'color': COLORS['text_secondary'], 'size': 10},
                'ticksuffix': '%'
            },
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': 'rgba(30, 41, 59, 0.8)',
            'borderwidth': 2,
            'bordercolor': COLORS['grid'],
            'steps': [
                {'range': [0, 40], 'color': 'rgba(71, 85, 105, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(99, 102, 241, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(139, 92, 246, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.8,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg']
    )
    
    return fig


def create_explanation_summary(
    predicted_service: str,
    confidence: float,
    modality_weights: Dict[str, float],
    top_features: List[Tuple[str, float]],
    causal_evidence: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a comprehensive explanation summary visualization.
    
    Args:
        predicted_service: Name of predicted root cause
        confidence: Prediction confidence
        modality_weights: Importance of each modality
        top_features: List of (feature_name, importance) tuples
        causal_evidence: Optional list of causal evidence strings
        
    Returns:
        Plotly figure with comprehensive explanation
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'indicator'}, {'type': 'bar'}],
            [{'type': 'bar', 'colspan': 2}, None]
        ],
        subplot_titles=[
            'Prediction Confidence',
            'Modality Contributions',
            'Top Contributing Features'
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # 1. Confidence gauge
    bar_color = COLORS['purple'] if confidence > 0.5 else COLORS['accent']
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={'suffix': '%', 'font': {'color': COLORS['text'], 'size': 28}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'color': COLORS['text_secondary'], 'size': 9}},
                'bar': {'color': bar_color},
                'bgcolor': 'rgba(30, 41, 59, 0.8)',
                'borderwidth': 1,
                'bordercolor': COLORS['grid']
            }
        ),
        row=1, col=1
    )
    
    # 2. Modality bar chart
    modalities = list(modality_weights.keys())
    weights = list(modality_weights.values())
    mod_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    
    fig.add_trace(
        go.Bar(
            y=modalities,
            x=weights,
            orientation='h',
            marker=dict(color=mod_colors[:len(modalities)]),
            text=[f'{w:.0%}' for w in weights],
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10)
        ),
        row=1, col=2
    )
    
    # 3. Feature importance
    feature_names = [f[0] for f in top_features[:8]]
    feature_values = [f[1] for f in top_features[:8]]
    feat_colors = [COLORS['secondary'] if v > 0 else COLORS['danger'] for v in feature_values]
    
    fig.add_trace(
        go.Bar(
            y=feature_names,
            x=feature_values,
            orientation='h',
            marker=dict(color=feat_colors),
            text=[f'{v:+.3f}' for v in feature_values],
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=9)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        margin=dict(l=100, r=40, t=60, b=40),
        title=dict(
            text=f'<b>Explanation: {predicted_service}</b>',
            font=dict(color=COLORS['text'], size=16)
        )
    )
    
    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.color = COLORS['text_secondary']
        annotation.font.size = 11
    
    # Update axes
    for i in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=COLORS['grid'],
            tickfont=dict(color=COLORS['text_secondary']),
            row=i
        )
        fig.update_yaxes(
            tickfont=dict(color=COLORS['text']),
            row=i
        )
    
    return fig


def create_llm_explanation_card(
    explanation_text: str,
    confidence: float,
    sources: List[str]
) -> str:
    """
    Create HTML for an LLM explanation card.
    
    Args:
        explanation_text: The LLM-generated explanation
        confidence: Confidence score
        sources: List of evidence sources
        
    Returns:
        HTML string for the explanation card
    """
    confidence_color = '#10b981' if confidence > 0.7 else '#f59e0b' if confidence > 0.4 else '#ef4444'
    
    sources_html = ''.join([
        f'<span style="background: rgba(99, 102, 241, 0.2); padding: 2px 8px; '
        f'border-radius: 4px; margin: 2px; font-size: 11px;">{s}</span>'
        for s in sources
    ])
    
    html = f"""
    <div style="background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
                border-radius: 16px; padding: 24px; 
                border: 1px solid rgba(99, 102, 241, 0.2);
                box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="color: white; margin: 0; font-size: 16px;">
                🤖 AI Explanation
            </h3>
            <span style="background: {confidence_color}; padding: 4px 12px; 
                        border-radius: 12px; font-size: 12px; color: white;">
                {confidence:.0%} confidence
            </span>
        </div>
        
        <p style="color: #e2e8f0; font-size: 14px; line-height: 1.6; margin-bottom: 16px;">
            {explanation_text}
        </p>
        
        <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;">
            <span style="color: #94a3b8; font-size: 11px;">Evidence Sources:</span>
            <div style="margin-top: 8px;">
                {sources_html}
            </div>
        </div>
    </div>
    """
    
    return html


def generate_demo_explanations(
    services: List[str],
    predicted_idx: int
) -> Dict:
    """
    Generate demo explanation data for visualization testing.
    
    Args:
        services: List of service names
        predicted_idx: Index of predicted service
        
    Returns:
        Dict with explanation components
    """
    np.random.seed(predicted_idx)
    
    # Modality weights
    modality_weights = {
        'Metrics': 0.45 + np.random.randn() * 0.1,
        'Logs': 0.32 + np.random.randn() * 0.08,
        'Traces': 0.23 + np.random.randn() * 0.05
    }
    total = sum(modality_weights.values())
    modality_weights = {k: v/total for k, v in modality_weights.items()}
    
    # Feature importance
    features = [
        ('cpu_usage_spike', 0.15),
        ('error_rate_increase', 0.12),
        ('latency_p99', 0.10),
        ('memory_usage', 0.08),
        ('connection_errors', 0.07),
        ('gc_pause_time', -0.05),
        ('request_rate', 0.04),
        ('log_error_count', 0.06)
    ]
    
    # Attention weights
    n = len(services)
    attention = np.random.rand(n, n) * 0.3
    attention[predicted_idx, :] += 0.2
    attention[:, predicted_idx] += 0.15
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    # Confidence
    confidence = 0.5 + np.random.rand() * 0.4
    
    return {
        'modality_weights': modality_weights,
        'features': features,
        'attention': attention,
        'confidence': confidence
    }
