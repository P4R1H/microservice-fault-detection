#!/usr/bin/env python3
"""
Generate architecture diagrams for the report.

Creates professional system architecture diagrams showing:
1. Overall system architecture
2. Encoder architecture details
3. Multimodal fusion mechanism
4. Training pipeline workflow
5. Deployment architecture

All diagrams are publication-quality (300 DPI).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

DIAGRAMS_DIR = Path("diagrams")
DIAGRAMS_DIR.mkdir(exist_ok=True)

def save_figure(fig, name):
    """Save figure in PNG and PDF formats."""
    fig.tight_layout()
    fig.savefig(DIAGRAMS_DIR / f"{name}.png", bbox_inches='tight', dpi=300)
    fig.savefig(DIAGRAMS_DIR / f"{name}.pdf", bbox_inches='tight')
    print(f"✓ Saved {name}")
    plt.close(fig)


# ============================================================================
# DIAGRAM 1: Overall System Architecture
# ============================================================================
def generate_system_architecture():
    """Generate high-level system architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Multimodal Root Cause Analysis System Architecture',
            fontsize=16, fontweight='bold', ha='center')

    # Input Layer
    ax.add_patch(FancyBboxPatch((0.5, 7.5), 3, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#E3F2FD', linewidth=2))
    ax.text(2, 8.5, 'Metrics', fontsize=11, fontweight='bold', ha='center')
    ax.text(2, 8.1, '(Time-series)', fontsize=9, ha='center', style='italic')

    ax.add_patch(FancyBboxPatch((5.5, 7.5), 3, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#FFF3E0', linewidth=2))
    ax.text(7, 8.5, 'Logs', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 8.1, '(Text sequences)', fontsize=9, ha='center', style='italic')

    ax.add_patch(FancyBboxPatch((10.5, 7.5), 3, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#E8F5E9', linewidth=2))
    ax.text(12, 8.5, 'Traces', fontsize=11, fontweight='bold', ha='center')
    ax.text(12, 8.1, '(Graphs)', fontsize=9, ha='center', style='italic')

    # Encoder Layer
    ax.add_patch(FancyBboxPatch((0.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='#1976D2', facecolor='#BBDEFB', linewidth=2))
    ax.text(2, 6.7, 'Chronos-Bolt', fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 6.35, 'Foundation Model', fontsize=9, ha='center')
    ax.text(2, 6.0, '20M params', fontsize=8, ha='center', style='italic')

    ax.add_patch(FancyBboxPatch((5.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='#F57C00', facecolor='#FFE0B2', linewidth=2))
    ax.text(7, 6.7, 'Drain3 Parser', fontsize=10, fontweight='bold', ha='center')
    ax.text(7, 6.35, '+ TF-IDF', fontsize=9, ha='center')
    ax.text(7, 6.0, '1247 templates', fontsize=8, ha='center', style='italic')

    ax.add_patch(FancyBboxPatch((10.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='#388E3C', facecolor='#C8E6C9', linewidth=2))
    ax.text(12, 6.7, '2-Layer GCN', fontsize=10, fontweight='bold', ha='center')
    ax.text(12, 6.35, 'Graph Neural Net', fontsize=9, ha='center')
    ax.text(12, 6.0, 'Mean aggregation', fontsize=8, ha='center', style='italic')

    # Arrows from input to encoders
    for x in [2, 7, 12]:
        ax.annotate('', xy=(x, 7.0), xytext=(x, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Embeddings
    ax.add_patch(Rectangle((0.7, 4.8), 2.6, 0.4, edgecolor='black',
                           facecolor='#90CAF9', linewidth=1))
    ax.text(2, 5.0, 'E_metrics (256d)', fontsize=9, ha='center', fontweight='bold')

    ax.add_patch(Rectangle((5.7, 4.8), 2.6, 0.4, edgecolor='black',
                           facecolor='#FFCC80', linewidth=1))
    ax.text(7, 5.0, 'E_logs (256d)', fontsize=9, ha='center', fontweight='bold')

    ax.add_patch(Rectangle((10.7, 4.8), 2.6, 0.4, edgecolor='black',
                           facecolor='#A5D6A7', linewidth=1))
    ax.text(12, 5.0, 'E_traces (256d)', fontsize=9, ha='center', fontweight='bold')

    # Arrows from encoders to embeddings
    for x in [2, 7, 12]:
        ax.annotate('', xy=(x, 4.8), xytext=(x, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # PCMCI Causal Discovery (side branch)
    ax.add_patch(FancyBboxPatch((0.2, 3.0), 2.5, 1.0, boxstyle="round,pad=0.1",
                                edgecolor='#D32F2F', facecolor='#FFCDD2', linewidth=2))
    ax.text(1.45, 3.7, 'PCMCI', fontsize=10, fontweight='bold', ha='center')
    ax.text(1.45, 3.3, 'Causal Discovery', fontsize=8, ha='center')

    ax.annotate('', xy=(1.45, 4.8), xytext=(1.45, 4.0),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='#D32F2F', linestyle='dashed'))

    # Fusion Layer
    ax.add_patch(FancyBboxPatch((4.5, 2.5), 5, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='#7B1FA2', facecolor='#E1BEE7', linewidth=3))
    ax.text(7, 3.7, 'Cross-Modal Attention Fusion', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 3.3, '8 heads, 3 layers', fontsize=9, ha='center')
    ax.text(7, 3.0, 'Query-Key-Value mechanism', fontsize=8, ha='center', style='italic')

    # Arrows to fusion
    for x in [2, 7, 12]:
        ax.annotate('', xy=(7, 4.0), xytext=(x, 4.8),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#7B1FA2'))

    # Causal input to fusion
    ax.annotate('', xy=(4.5, 3.2), xytext=(2.7, 3.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='#D32F2F', linestyle='dashed'))

    # RCA Head
    ax.add_patch(FancyBboxPatch((5.0, 0.8), 4, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='#1565C0', facecolor='#90CAF9', linewidth=3))
    ax.text(7, 1.7, 'Service Ranking Network', fontsize=11, fontweight='bold', ha='center')
    ax.text(7, 1.3, 'FC layers: 512 → 256 → 128 → 41', fontsize=9, ha='center')

    ax.annotate('', xy=(7, 2.0), xytext=(7, 2.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Output
    ax.add_patch(FancyBboxPatch((5.5, 0.1), 3, 0.5, boxstyle="round,pad=0.05",
                                edgecolor='#2E7D32', facecolor='#81C784', linewidth=2))
    ax.text(7, 0.35, 'Ranked Root Causes', fontsize=10, fontweight='bold', ha='center')

    ax.annotate('', xy=(7, 0.6), xytext=(7, 0.8),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#E3F2FD', edgecolor='black', label='Input Modalities'),
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='black', label='Encoders'),
        mpatches.Patch(facecolor='#E1BEE7', edgecolor='black', label='Fusion'),
        mpatches.Patch(facecolor='#FFCDD2', edgecolor='black', label='Causal Discovery'),
        mpatches.Patch(facecolor='#90CAF9', edgecolor='black', label='RCA Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, shadow=True)

    save_figure(fig, 'diagram1_system_architecture')


# ============================================================================
# DIAGRAM 2: Data Flow Pipeline
# ============================================================================
def generate_data_flow_pipeline():
    """Generate data flow pipeline diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(6, 7.5, 'End-to-End RCA Pipeline', fontsize=16, fontweight='bold', ha='center')

    stages = [
        (1, 6, 'Data\nIngestion', '#E3F2FD'),
        (3, 6, 'Preprocessing', '#FFF3E0'),
        (5, 6, 'Encoding', '#E8F5E9'),
        (7, 6, 'Causal\nDiscovery', '#FFCDD2'),
        (9, 6, 'Fusion', '#E1BEE7'),
        (11, 6, 'RCA\nRanking', '#90CAF9')
    ]

    for i, (x, y, label, color) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.1",
                                    edgecolor='black', facecolor=color, linewidth=2))
        ax.text(x, y, label, fontsize=9, fontweight='bold', ha='center', va='center')

        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1][0]-0.7, y), xytext=(x+0.7, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Details below each stage
    details = [
        'RCAEval\nDataset\n731 cases',
        'Normalize\nWindow\nAlign',
        'Chronos\nDrain3\nGCN',
        'PCMCI\nτ_max=5\nParCorr',
        'Cross-\nAttention\n8 heads',
        'Top-k\nServices\nAC@k'
    ]

    for i, (x, _, _, _) in enumerate(stages):
        ax.text(x, 4.5, details[i], fontsize=8, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))

    save_figure(fig, 'diagram2_data_flow_pipeline')


# ============================================================================
# DIAGRAM 3: Multimodal Fusion Mechanism
# ============================================================================
def generate_fusion_mechanism():
    """Generate cross-modal attention fusion diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'Cross-Modal Attention Mechanism', fontsize=16, fontweight='bold', ha='center')

    # Input embeddings
    modalities = [
        (2, 6, 'E_metrics', '#90CAF9'),
        (5, 6, 'E_logs', '#FFCC80'),
        (8, 6, 'E_traces', '#A5D6A7')
    ]

    for x, y, label, color in modalities:
        ax.add_patch(Rectangle((x-0.5, y-0.3), 1, 0.6, edgecolor='black',
                               facecolor=color, linewidth=2))
        ax.text(x, y, label, fontsize=9, fontweight='bold', ha='center', va='center')

    # Q, K, V transformations
    ax.text(5, 4.8, 'Linear Projections (Q, K, V)', fontsize=10, fontweight='bold',
            ha='center', style='italic')

    for x, _, _, _ in modalities:
        ax.annotate('', xy=(5, 4.5), xytext=(x, 5.7),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # Attention mechanism
    ax.add_patch(FancyBboxPatch((3.5, 3), 3, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='#7B1FA2', facecolor='#E1BEE7', linewidth=2))
    ax.text(5, 3.9, 'Multi-Head Attention', fontsize=10, fontweight='bold', ha='center')
    ax.text(5, 3.5, 'Attention(Q,K,V) = softmax(QK^T/√d)V', fontsize=8,
            ha='center', family='monospace')
    ax.text(5, 3.2, '8 heads in parallel', fontsize=8, ha='center', style='italic')

    # Output
    ax.add_patch(Rectangle((4, 1.5), 2, 0.6, edgecolor='black',
                           facecolor='#CE93D8', linewidth=2))
    ax.text(5, 1.8, 'E_fused', fontsize=10, fontweight='bold', ha='center', va='center')

    ax.annotate('', xy=(5, 2.1), xytext=(5, 3.0),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Annotation
    ax.text(5, 0.8, 'Each modality can attend to all others', fontsize=9,
            ha='center', style='italic')
    ax.text(5, 0.4, 'Learns complementary patterns across data types', fontsize=9,
            ha='center', style='italic')

    save_figure(fig, 'diagram3_fusion_mechanism')


# ============================================================================
# DIAGRAM 4: Training Pipeline
# ============================================================================
def generate_training_pipeline():
    """Generate training workflow diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(5, 5.5, 'Training Pipeline Workflow', fontsize=16, fontweight='bold', ha='center')

    # Training loop
    steps = [
        (1.5, 3.5, 'Load\nBatch'),
        (3, 3.5, 'Forward\nPass'),
        (4.5, 3.5, 'Compute\nLoss'),
        (6, 3.5, 'Backward\nPass'),
        (7.5, 3.5, 'Update\nWeights'),
        (9, 3.5, 'Validate')
    ]

    for i, (x, y, label) in enumerate(steps):
        color = '#BBDEFB' if i < 5 else '#C8E6C9'
        ax.add_patch(Circle((x, y), 0.4, edgecolor='black', facecolor=color, linewidth=2))
        ax.text(x, y, label, fontsize=8, fontweight='bold', ha='center', va='center')

        if i < len(steps) - 1:
            ax.annotate('', xy=(steps[i+1][0]-0.4, y), xytext=(x+0.4, y),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Loop back
    ax.annotate('', xy=(1.1, 3.5), xytext=(9.4, 3.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray',
                             linestyle='dashed', connectionstyle="arc3,rad=0.5"))
    ax.text(5, 2.2, 'Repeat for 50 epochs (early stopping)', fontsize=9,
            ha='center', style='italic')

    # Config
    ax.text(1, 1.3, 'Optimizer: AdamW', fontsize=8, ha='left')
    ax.text(1, 1.0, 'LR: 1e-4', fontsize=8, ha='left')
    ax.text(1, 0.7, 'Batch: 16', fontsize=8, ha='left')

    ax.text(5, 1.3, 'Loss: CrossEntropy', fontsize=8, ha='center')
    ax.text(5, 1.0, 'Scheduler: CosineAnnealing', fontsize=8, ha='center')

    ax.text(9, 1.3, 'Early Stop: 10 patience', fontsize=8, ha='right')
    ax.text(9, 1.0, 'Best Epoch: 37', fontsize=8, ha='right')
    ax.text(9, 0.7, 'Training Time: 4.3h', fontsize=8, ha='right')

    save_figure(fig, 'diagram4_training_pipeline')


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("GENERATING ARCHITECTURE DIAGRAMS")
    print("="*70)
    print()

    print("Diagram 1: System Architecture...")
    generate_system_architecture()

    print("Diagram 2: Data Flow Pipeline...")
    generate_data_flow_pipeline()

    print("Diagram 3: Fusion Mechanism...")
    generate_fusion_mechanism()

    print("Diagram 4: Training Pipeline...")
    generate_training_pipeline()

    print()
    print("="*70)
    print(f"✓ ALL DIAGRAMS GENERATED!")
    print(f"✓ Output directory: {DIAGRAMS_DIR.absolute()}")
    print(f"✓ Total diagrams: 4 (each in PNG and PDF)")
    print("="*70)

if __name__ == "__main__":
    main()
