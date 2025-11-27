"""
================================================================================
INTERACTIVE CAUSAL GRAPH VISUALIZATION
================================================================================

Advanced causal graph visualization using Pyvis and custom D3.js-style rendering.
Features: Force-directed layout, animations, highlighting, and detailed tooltips.

Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
Date: November 2025
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from dataclasses import dataclass
import streamlit as st


@dataclass
class GraphNode:
    """Represents a service node in the causal graph."""
    id: str
    name: str
    x: float
    y: float
    probability: float = 0.0
    is_root_cause: bool = False
    is_predicted: bool = False
    modality_scores: Optional[Dict[str, float]] = None
    
    @property
    def color(self) -> str:
        if self.is_root_cause:
            return "#ef4444"  # Red
        elif self.is_predicted:
            return "#8b5cf6"  # Purple
        elif self.probability > 0.5:
            return "#f59e0b"  # Orange
        elif self.probability > 0.2:
            return "#6366f1"  # Indigo
        else:
            return "#475569"  # Slate
    
    @property
    def size(self) -> int:
        base_size = 35
        if self.is_root_cause:
            return base_size + 20
        elif self.is_predicted:
            return base_size + 15
        else:
            return base_size + int(30 * self.probability)


@dataclass
class GraphEdge:
    """Represents a causal relationship edge."""
    source: str
    target: str
    weight: float
    is_highlighted: bool = False
    
    @property
    def color(self) -> str:
        alpha = 0.3 + 0.5 * self.weight
        if self.is_highlighted:
            return f"rgba(239, 68, 68, {alpha})"  # Red for highlighted
        return f"rgba(99, 102, 241, {alpha})"  # Indigo default
    
    @property
    def width(self) -> float:
        return 1 + 4 * self.weight


class CausalGraphVisualization:
    """
    Advanced causal graph visualization with multiple layout algorithms
    and interactive features.
    """
    
    def __init__(
        self,
        services: List[str],
        causal_weights: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        root_cause_idx: Optional[int] = None,
        threshold: float = 0.1
    ):
        self.services = services
        self.causal_weights = causal_weights
        self.predictions = predictions
        self.root_cause_idx = root_cause_idx
        self.threshold = threshold
        self.n_services = len(services)
        
        # Initialize nodes and edges
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        
        self._create_nodes()
        self._create_edges()
    
    def _create_nodes(self):
        """Create graph nodes from services."""
        # Initial circular layout
        angles = np.linspace(0, 2 * np.pi, self.n_services, endpoint=False)
        radius = 2.0
        
        for i, service in enumerate(self.services):
            prob = self.predictions[i] if self.predictions is not None else 0.0
            
            node = GraphNode(
                id=f"node_{i}",
                name=service,
                x=radius * np.cos(angles[i] - np.pi/2),
                y=radius * np.sin(angles[i] - np.pi/2),
                probability=prob,
                is_root_cause=(self.root_cause_idx == i),
                is_predicted=(self.predictions is not None and i == np.argmax(self.predictions))
            )
            self.nodes.append(node)
    
    def _create_edges(self):
        """Create edges from causal weight matrix."""
        for i in range(self.n_services):
            for j in range(self.n_services):
                if i != j and self.causal_weights[i, j] > self.threshold:
                    edge = GraphEdge(
                        source=f"node_{i}",
                        target=f"node_{j}",
                        weight=self.causal_weights[i, j]
                    )
                    self.edges.append(edge)
    
    def apply_force_directed_layout(self, iterations: int = 100):
        """
        Apply force-directed layout algorithm.
        
        Uses spring-electrical model with:
        - Attractive forces between connected nodes
        - Repulsive forces between all nodes
        """
        # Position array
        positions = np.array([[n.x, n.y] for n in self.nodes])
        
        # Parameters
        k = 1.0  # Ideal spring length
        c = 2.0  # Repulsion constant
        dt = 0.1  # Time step
        
        for _ in range(iterations):
            forces = np.zeros_like(positions)
            
            # Repulsive forces (all pairs)
            for i in range(self.n_services):
                for j in range(i + 1, self.n_services):
                    delta = positions[i] - positions[j]
                    dist = np.linalg.norm(delta) + 0.01
                    
                    # Coulomb repulsion
                    force = c / (dist ** 2) * delta / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Attractive forces (edges)
            for edge in self.edges:
                i = int(edge.source.split('_')[1])
                j = int(edge.target.split('_')[1])
                
                delta = positions[j] - positions[i]
                dist = np.linalg.norm(delta) + 0.01
                
                # Hooke's law
                force = (dist - k) * delta / dist * edge.weight
                forces[i] += force
                forces[j] -= force
            
            # Update positions
            positions += dt * forces
            
            # Center the graph
            positions -= positions.mean(axis=0)
        
        # Update node positions
        for i, node in enumerate(self.nodes):
            node.x = positions[i, 0]
            node.y = positions[i, 1]
    
    def apply_hierarchical_layout(self):
        """
        Apply hierarchical layout based on causal influence.
        Most influential nodes at top.
        """
        # Calculate influence scores
        influence = self.causal_weights.sum(axis=1)
        
        # Sort by influence
        sorted_idx = np.argsort(influence)[::-1]
        
        # Arrange in layers
        n_layers = min(4, self.n_services)
        nodes_per_layer = self.n_services // n_layers
        
        for rank, idx in enumerate(sorted_idx):
            layer = rank // nodes_per_layer
            pos_in_layer = rank % nodes_per_layer
            total_in_layer = min(nodes_per_layer, self.n_services - layer * nodes_per_layer)
            
            # Calculate position
            self.nodes[idx].y = 2 - layer * 1.2  # Top to bottom
            self.nodes[idx].x = (pos_in_layer - total_in_layer / 2 + 0.5) * 1.5
    
    def highlight_path_to(self, target_idx: int):
        """Highlight causal paths leading to target node."""
        # Find incoming edges
        for edge in self.edges:
            target_node_idx = int(edge.target.split('_')[1])
            if target_node_idx == target_idx:
                edge.is_highlighted = True
    
    def create_figure(
        self,
        width: int = 700,
        height: int = 600,
        show_labels: bool = True,
        animate: bool = False
    ) -> go.Figure:
        """
        Create Plotly figure with the causal graph.
        
        Args:
            width: Figure width
            height: Figure height
            show_labels: Whether to show node labels
            animate: Whether to add animation effects
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add edges first (so they're behind nodes)
        for edge in self.edges:
            source_idx = int(edge.source.split('_')[1])
            target_idx = int(edge.target.split('_')[1])
            
            source_node = self.nodes[source_idx]
            target_node = self.nodes[target_idx]
            
            # Calculate bezier curve control point
            mid_x = (source_node.x + target_node.x) / 2
            mid_y = (source_node.y + target_node.y) / 2
            
            # Perpendicular offset for curve
            dx = target_node.x - source_node.x
            dy = target_node.y - source_node.y
            
            curve_offset = 0.2 + 0.1 * edge.weight
            ctrl_x = mid_x - curve_offset * dy
            ctrl_y = mid_y + curve_offset * dx
            
            # Create bezier curve points
            t = np.linspace(0, 1, 30)
            bx = (1-t)**2 * source_node.x + 2*(1-t)*t * ctrl_x + t**2 * target_node.x
            by = (1-t)**2 * source_node.y + 2*(1-t)*t * ctrl_y + t**2 * target_node.y
            
            # Add edge trace
            fig.add_trace(go.Scatter(
                x=bx.tolist(),
                y=by.tolist(),
                mode='lines',
                line=dict(
                    color=edge.color,
                    width=edge.width,
                    shape='spline'
                ),
                hoverinfo='text',
                hovertext=f"{source_node.name} → {target_node.name}<br>Weight: {edge.weight:.3f}",
                showlegend=False
            ))
            
            # Add arrowhead
            arrow_size = 0.08 + 0.04 * edge.weight
            
            # Direction at end of curve
            end_dx = bx[-1] - bx[-2]
            end_dy = by[-1] - by[-2]
            angle = np.arctan2(end_dy, end_dx)
            
            arrow_x = [
                bx[-1],
                bx[-1] - arrow_size * np.cos(angle - 0.35),
                bx[-1] - arrow_size * np.cos(angle + 0.35),
                bx[-1]
            ]
            arrow_y = [
                by[-1],
                by[-1] - arrow_size * np.sin(angle - 0.35),
                by[-1] - arrow_size * np.sin(angle + 0.35),
                by[-1]
            ]
            
            fig.add_trace(go.Scatter(
                x=arrow_x,
                y=arrow_y,
                fill='toself',
                fillcolor=edge.color,
                line=dict(color=edge.color, width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add nodes
        node_x = [n.x for n in self.nodes]
        node_y = [n.y for n in self.nodes]
        node_colors = [n.color for n in self.nodes]
        node_sizes = [n.size for n in self.nodes]
        
        # Node hover text
        hover_texts = []
        for node in self.nodes:
            text = f"<b>{node.name}</b><br>"
            if node.probability > 0:
                text += f"Probability: {node.probability:.1%}<br>"
            if node.is_root_cause:
                text += "<span style='color:#ef4444'>● Root Cause</span><br>"
            if node.is_predicted:
                text += "<span style='color:#8b5cf6'>● Predicted</span><br>"
            
            # Add influence metrics
            idx = int(node.id.split('_')[1])
            outgoing = self.causal_weights[idx].sum()
            incoming = self.causal_weights[:, idx].sum()
            text += f"Outgoing influence: {outgoing:.2f}<br>"
            text += f"Incoming influence: {incoming:.2f}"
            hover_texts.append(text)
        
        # Add outer glow for important nodes
        glow_x, glow_y, glow_sizes, glow_colors = [], [], [], []
        for node in self.nodes:
            if node.is_root_cause or node.is_predicted:
                glow_x.append(node.x)
                glow_y.append(node.y)
                glow_sizes.append(node.size + 20)
                color = node.color.replace(')', ', 0.3)').replace('rgb', 'rgba')
                if '#' in node.color:
                    # Convert hex to rgba
                    hex_color = node.color.lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    color = f'rgba({r}, {g}, {b}, 0.3)'
                glow_colors.append(color)
        
        if glow_x:
            fig.add_trace(go.Scatter(
                x=glow_x,
                y=glow_y,
                mode='markers',
                marker=dict(
                    size=glow_sizes,
                    color=glow_colors,
                    line=dict(width=0)
                ),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Main node scatter
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(
                    width=3,
                    color='white'
                ),
                symbol='circle'
            ),
            text=[n.name[:10] + '..' if len(n.name) > 10 else n.name for n in self.nodes] if show_labels else None,
            textposition='bottom center',
            textfont=dict(
                color='white',
                size=10,
                family='Arial'
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=False
        ))
        
        # Add legend manually
        legend_items = [
            ("Root Cause", "#ef4444"),
            ("Predicted", "#8b5cf6"),
            ("High Prob", "#f59e0b"),
            ("Normal", "#475569")
        ]
        
        for i, (name, color) in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=name,
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(color='white', size=11),
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_x) - 1, max(node_x) + 1]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[min(node_y) - 1, max(node_y) + 1],
                scaleanchor="x",
                scaleratio=1
            ),
            width=width,
            height=height,
            margin=dict(l=20, r=20, t=40, b=60),
            hovermode='closest'
        )
        
        # Add title
        fig.add_annotation(
            text="<b>Service Causal Graph</b>",
            x=0.5, y=1.05,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="white")
        )
        
        return fig
    
    def get_graph_statistics(self) -> Dict:
        """Calculate graph statistics."""
        n_edges = len(self.edges)
        max_edges = self.n_services * (self.n_services - 1)
        density = n_edges / max_edges if max_edges > 0 else 0
        
        # Influence scores
        outgoing = self.causal_weights.sum(axis=1)
        incoming = self.causal_weights.sum(axis=0)
        
        return {
            'n_nodes': self.n_services,
            'n_edges': n_edges,
            'density': density,
            'most_influential': self.services[np.argmax(outgoing)],
            'most_affected': self.services[np.argmax(incoming)],
            'avg_edge_weight': np.mean([e.weight for e in self.edges]) if self.edges else 0,
            'max_edge_weight': max([e.weight for e in self.edges]) if self.edges else 0
        }


def create_3d_causal_graph(
    services: List[str],
    causal_weights: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    root_cause_idx: Optional[int] = None,
    threshold: float = 0.1
) -> go.Figure:
    """
    Create a 3D interactive causal graph visualization.
    
    Adds depth dimension for temporal/hierarchical representation.
    """
    n_services = len(services)
    
    # 3D spherical layout
    phi = np.linspace(0, 2 * np.pi, n_services, endpoint=False)
    theta = np.linspace(np.pi/4, 3*np.pi/4, n_services)
    radius = 2.0
    
    # Calculate influence for z-position
    influence = causal_weights.sum(axis=1)
    influence_norm = (influence - influence.min()) / (influence.max() - influence.min() + 0.01)
    
    x_pos = radius * np.cos(phi)
    y_pos = radius * np.sin(phi)
    z_pos = influence_norm * 2 - 1  # Range [-1, 1]
    
    fig = go.Figure()
    
    # Add edges
    for i in range(n_services):
        for j in range(n_services):
            if i != j and causal_weights[i, j] > threshold:
                weight = causal_weights[i, j]
                
                # Line from i to j
                fig.add_trace(go.Scatter3d(
                    x=[x_pos[i], x_pos[j]],
                    y=[y_pos[i], y_pos[j]],
                    z=[z_pos[i], z_pos[j]],
                    mode='lines',
                    line=dict(
                        color=f'rgba(99, 102, 241, {0.2 + 0.6 * weight})',
                        width=2 + 4 * weight
                    ),
                    hoverinfo='text',
                    hovertext=f'{services[i]} → {services[j]}<br>Weight: {weight:.3f}',
                    showlegend=False
                ))
    
    # Add nodes
    node_colors = []
    node_sizes = []
    hover_texts = []
    
    for i, service in enumerate(services):
        if root_cause_idx is not None and i == root_cause_idx:
            color = '#ef4444'
            size = 20
        elif predictions is not None and i == np.argmax(predictions):
            color = '#8b5cf6'
            size = 16
        elif predictions is not None and predictions[i] > 0.3:
            color = '#f59e0b'
            size = 12
        else:
            color = '#6366f1'
            size = 10
        
        node_colors.append(color)
        node_sizes.append(size)
        
        text = f"<b>{service}</b><br>"
        if predictions is not None:
            text += f"P: {predictions[i]:.1%}<br>"
        text += f"Influence: {influence[i]:.2f}"
        hover_texts.append(text)
    
    fig.add_trace(go.Scatter3d(
        x=x_pos,
        y=y_pos,
        z=z_pos,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        text=services,
        textposition='top center',
        textfont=dict(color='white', size=10),
        hoverinfo='text',
        hovertext=hover_texts,
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(
                visible=True,
                title='Influence Level',
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)',
                backgroundcolor='rgba(0,0,0,0)'
            ),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    
    return fig


def create_animated_propagation(
    services: List[str],
    causal_weights: np.ndarray,
    start_idx: int,
    n_frames: int = 30
) -> go.Figure:
    """
    Create an animated visualization showing fault propagation.
    
    Shows how a fault in one service propagates through the system
    based on causal relationships.
    """
    n_services = len(services)
    
    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n_services, endpoint=False)
    radius = 2.0
    x_pos = radius * np.cos(angles - np.pi/2)
    y_pos = radius * np.sin(angles - np.pi/2)
    
    # Simulate propagation
    activation = np.zeros(n_services)
    activation[start_idx] = 1.0
    
    frames = []
    activations_over_time = [activation.copy()]
    
    for _ in range(n_frames - 1):
        # Propagate through causal links
        new_activation = activation.copy()
        for i in range(n_services):
            if activation[i] > 0.1:
                # Spread to connected services
                for j in range(n_services):
                    if causal_weights[i, j] > 0.1:
                        new_activation[j] = max(
                            new_activation[j],
                            activation[i] * causal_weights[i, j] * 0.8
                        )
        activation = new_activation
        activations_over_time.append(activation.copy())
    
    # Create figure with frames
    fig = go.Figure()
    
    # Initial frame
    initial_activation = activations_over_time[0]
    node_colors = [
        f'rgba(239, 68, 68, {min(1, a + 0.2)})' if a > 0.1 
        else 'rgba(71, 85, 105, 0.8)' 
        for a in initial_activation
    ]
    
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=[30 + 20 * a for a in initial_activation],
            color=node_colors
        ),
        text=services,
        textposition='bottom center',
        textfont=dict(color='white', size=10)
    ))
    
    # Add edges (static)
    for i in range(n_services):
        for j in range(n_services):
            if i != j and causal_weights[i, j] > 0.1:
                fig.add_trace(go.Scatter(
                    x=[x_pos[i], x_pos[j]],
                    y=[y_pos[i], y_pos[j]],
                    mode='lines',
                    line=dict(
                        color='rgba(99, 102, 241, 0.2)',
                        width=1
                    ),
                    showlegend=False
                ))
    
    # Create animation frames
    for frame_idx, activation in enumerate(activations_over_time):
        node_colors = [
            f'rgba(239, 68, 68, {min(1, a + 0.2)})' if a > 0.1 
            else 'rgba(71, 85, 105, 0.8)' 
            for a in activation
        ]
        
        frames.append(go.Frame(
            data=[go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                marker=dict(
                    size=[30 + 20 * a for a in activation],
                    color=node_colors
                ),
                text=services,
                textposition='bottom center',
                textfont=dict(color='white', size=10)
            )],
            name=str(frame_idx)
        ))
    
    fig.frames = frames
    
    # Add play button
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 200, "redraw": True},
                            "fromcurrent": True
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate"
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 200, "redraw": True}, "mode": "immediate"}],
                 "label": str(i), "method": "animate"}
                for i, f in enumerate(frames)
            ],
            "x": 0.1,
            "len": 0.8,
            "y": -0.05,
            "currentvalue": {
                "prefix": "Time Step: ",
                "visible": True,
                "xanchor": "center",
                "font": {"color": "white"}
            },
            "font": {"color": "white"},
            "tickcolor": "white"
        }],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, range=[-3, 3]),
        yaxis=dict(visible=False, range=[-3, 3], scaleanchor="x"),
        height=600,
        margin=dict(l=20, r=20, t=40, b=80),
        title=dict(
            text=f"<b>Fault Propagation from {services[start_idx]}</b>",
            font=dict(color='white', size=16)
        )
    )
    
    return fig


# Streamlit component wrapper
def render_causal_graph_widget(
    services: List[str],
    causal_weights: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    root_cause_idx: Optional[int] = None,
    key: str = "causal_graph"
):
    """
    Render an interactive causal graph widget in Streamlit.
    
    Provides controls for layout, threshold, and visualization options.
    """
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        layout = st.selectbox(
            "Layout",
            ["Circular", "Force-Directed", "Hierarchical"],
            key=f"{key}_layout"
        )
    
    with col2:
        threshold = st.slider(
            "Edge Threshold",
            0.0, 0.5, 0.1, 0.05,
            key=f"{key}_threshold"
        )
    
    with col3:
        show_labels = st.checkbox("Show Labels", value=True, key=f"{key}_labels")
    
    with col4:
        view_3d = st.checkbox("3D View", value=False, key=f"{key}_3d")
    
    if view_3d:
        fig = create_3d_causal_graph(
            services, causal_weights, predictions, root_cause_idx, threshold
        )
    else:
        viz = CausalGraphVisualization(
            services, causal_weights, predictions, root_cause_idx, threshold
        )
        
        if layout == "Force-Directed":
            viz.apply_force_directed_layout()
        elif layout == "Hierarchical":
            viz.apply_hierarchical_layout()
        
        fig = viz.create_figure(show_labels=show_labels)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    if not view_3d:
        stats = viz.get_graph_statistics()
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Nodes", stats['n_nodes'])
        with stat_cols[1]:
            st.metric("Edges", stats['n_edges'])
        with stat_cols[2]:
            st.metric("Density", f"{stats['density']:.1%}")
        with stat_cols[3]:
            st.metric("Avg Weight", f"{stats['avg_edge_weight']:.2f}")
