#!/usr/bin/env python3
"""
Generate ALL figures from JSON data files.

This script reads from raw_results/*.json and generates publication-quality
figures in the figures/ directory.

When you have new experimental data, just replace the JSON files and re-run this script!

Usage:
    python generate_all_figures.py

Output:
    - All figures saved to results/figures/
    - 300 DPI PNG + PDF formats
    - IEEE/Science style formatting
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Directories
RAW_RESULTS = Path("raw_results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

def load_json(filename):
    """Load JSON data file."""
    with open(RAW_RESULTS / filename, 'r') as f:
        return json.load(f)

def save_figure(fig, name):
    """Save figure in PNG and PDF formats."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches='tight', dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches='tight')
    print(f"✓ Saved {name}")
    plt.close(fig)


# ============================================================================
# FIGURE 1: Baseline Comparison (Bar Chart)
# ============================================================================
def generate_baseline_comparison():
    data = load_json("baseline_comparison.json")
    results = data['results']

    methods = list(results.keys())
    ac1 = [results[m]['ac_at_1'] for m in methods]
    ac3 = [results[m]['ac_at_3'] for m in methods]
    ac5 = [results[m]['ac_at_5'] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax.bar(x - width, ac1, width, label='AC@1', color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, ac3, width, label='AC@3', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, ac5, width, label='AC@5', color='#2ca02c', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Root Cause Analysis Performance: Baseline Comparison', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])

    # Highlight our method
    for i, method in enumerate(methods):
        if 'Ours' in method:
            ax.axvspan(i-0.5, i+0.5, alpha=0.1, color='green')

    save_figure(fig, 'fig1_baseline_comparison')


# ============================================================================
# FIGURE 2: Ablation Study - Incremental Gains (Grouped Bar)
# ============================================================================
def generate_ablation_incremental():
    data = load_json("ablation_study.json")
    ablations = data['modality_ablations']

    configs = [
        'Metrics Only (Chronos)',
        'Metrics + Logs',
        'Metrics + Traces',
        'All Modalities (No Causal)',
        'All + PCMCI (No Cross-Attention)',
        'Full System (All + PCMCI + Cross-Attention)'
    ]

    ac1 = [ablations[c]['ac_at_1'] for c in configs]
    ac3 = [ablations[c]['ac_at_3'] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, ac1, width, label='AC@1', color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ac3, width, label='AC@3', color='#ff7f0e', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Ablation Study: Incremental Component Gains', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])

    save_figure(fig, 'fig2_ablation_incremental')


# ============================================================================
# FIGURE 3: Performance by Fault Type (Heatmap)
# ============================================================================
def generate_fault_type_heatmap():
    data = load_json("performance_by_fault_type.json")
    faults = data['fault_types']

    fault_names = list(faults.keys())
    metrics = ['ac_at_1', 'ac_at_3', 'ac_at_5', 'mrr']
    metric_labels = ['AC@1', 'AC@3', 'AC@5', 'MRR']

    # Create matrix
    matrix = np.array([[faults[f][m] for m in metrics] for f in fault_names])

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0.6, vmax=1.0)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(fault_names)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticklabels(fault_names)

    # Add text annotations
    for i in range(len(fault_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Performance by Fault Injection Type', fontweight='bold', pad=15)
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Fault Type', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=15, fontweight='bold')

    save_figure(fig, 'fig3_performance_by_fault_type')


# ============================================================================
# FIGURE 4: Modality Comparison (Radar Chart)
# ============================================================================
def generate_modality_radar():
    data = load_json("ablation_study.json")
    ablations = data['modality_ablations']

    categories = ['AC@1', 'AC@3', 'AC@5', 'MRR']

    metrics_only = [ablations['Metrics Only (Chronos)']['ac_at_1'],
                    ablations['Metrics Only (Chronos)']['ac_at_3'],
                    ablations['Metrics Only (Chronos)']['ac_at_5'],
                    ablations['Metrics Only (Chronos)']['mrr']]

    logs_only = [ablations['Logs Only (Drain3)']['ac_at_1'],
                 ablations['Logs Only (Drain3)']['ac_at_3'],
                 ablations['Logs Only (Drain3)']['ac_at_5'],
                 ablations['Logs Only (Drain3)']['mrr']]

    traces_only = [ablations['Traces Only (GCN)']['ac_at_1'],
                   ablations['Traces Only (GCN)']['ac_at_3'],
                   ablations['Traces Only (GCN)']['ac_at_5'],
                   ablations['Traces Only (GCN)']['mrr']]

    full_system = [ablations['Full System (All + PCMCI + Cross-Attention)']['ac_at_1'],
                   ablations['Full System (All + PCMCI + Cross-Attention)']['ac_at_3'],
                   ablations['Full System (All + PCMCI + Cross-Attention)']['ac_at_5'],
                   ablations['Full System (All + PCMCI + Cross-Attention)']['mrr']]

    # Close the plot
    metrics_only += metrics_only[:1]
    logs_only += logs_only[:1]
    traces_only += traces_only[:1]
    full_system += full_system[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    ax.plot(angles, metrics_only, 'o-', linewidth=2, label='Metrics Only', color='#1f77b4')
    ax.fill(angles, metrics_only, alpha=0.15, color='#1f77b4')

    ax.plot(angles, logs_only, 's-', linewidth=2, label='Logs Only', color='#ff7f0e')
    ax.fill(angles, logs_only, alpha=0.15, color='#ff7f0e')

    ax.plot(angles, traces_only, '^-', linewidth=2, label='Traces Only', color='#2ca02c')
    ax.fill(angles, traces_only, alpha=0.15, color='#2ca02c')

    ax.plot(angles, full_system, 'D-', linewidth=3, label='Full System', color='#d62728')
    ax.fill(angles, full_system, alpha=0.2, color='#d62728')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_title('Single-Modality vs Multi-Modal Performance', fontweight='bold', pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)

    save_figure(fig, 'fig4_modality_radar')


# ============================================================================
# FIGURE 5: Dataset Statistics (Combined)
# ============================================================================
def generate_dataset_statistics():
    data = load_json("dataset_statistics.json")
    fault_dist = data['fault_distribution']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Fault distribution pie chart
    faults = list(fault_dist.keys())
    counts = list(fault_dist.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(faults)))

    wedges, texts, autotexts = ax1.pie(counts, labels=faults, autopct='%1.1f%%',
                                         colors=colors, startangle=90,
                                         textprops={'fontsize': 9})
    ax1.set_title('Fault Type Distribution\n(192 Test Cases)', fontweight='bold', pad=15)

    # Fault distribution bar chart
    ax2.bar(faults, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Fault Type', fontweight='bold')
    ax2.set_ylabel('Number of Cases', fontweight='bold')
    ax2.set_title('Fault Type Frequency', fontweight='bold', pad=15)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    save_figure(fig, 'fig5_dataset_statistics')


# ============================================================================
# FIGURE 6: Cross-Modal Attention Heatmap
# ============================================================================
def generate_attention_heatmap():
    data = load_json("attention_weights_sample.json")
    attn = data['service_attention_heatmap']

    services = attn['services']

    # Create attention matrix
    matrix = np.array([
        attn['metrics_attention'],
        attn['logs_attention'],
        attn['traces_attention'],
        attn['fused_attention']
    ])

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    ax.set_xticks(np.arange(len(services)))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(services, rotation=45, ha='right')
    ax.set_yticklabels(['Metrics', 'Logs', 'Traces', 'Fused'])

    # Add text annotations
    for i in range(4):
        for j in range(len(services)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if matrix[i, j] > 0.5 else "black",
                          fontsize=8)

    ax.set_title('Cross-Modal Attention Weights (Sample Case: CPU Fault)\nGround Truth: ts-order-service',
                 fontweight='bold', pad=15)
    ax.set_xlabel('Suspected Services (Top-10)', fontweight='bold')
    ax.set_ylabel('Modality', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Score', rotation=270, labelpad=15, fontweight='bold')

    save_figure(fig, 'fig6_attention_heatmap')


# ============================================================================
# FIGURE 7: Inference Time vs Accuracy Trade-off
# ============================================================================
def generate_time_accuracy_tradeoff():
    data = load_json("baseline_comparison.json")
    results = data['results']

    methods = list(results.keys())
    ac1 = [results[m]['ac_at_1'] for m in methods]
    times = [results[m]['inference_time_sec'] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['gray' if 'Ours' not in m else 'red' for m in methods]
    sizes = [100 if 'Ours' not in m else 300 for m in methods]

    for i, method in enumerate(methods):
        ax.scatter(times[i], ac1[i], s=sizes[i], c=colors[i], alpha=0.6, edgecolors='black', linewidth=1.5)

        if 'Ours' in method or 'RUN' in method or 'BARO' in method:
            ax.annotate(method, (times[i], ac1[i]), xytext=(10, 10),
                       textcoords='offset points', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        else:
            ax.annotate(method, (times[i], ac1[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, alpha=0.7)

    ax.set_xlabel('Inference Time (seconds/case)', fontweight='bold')
    ax.set_ylabel('AC@1 Accuracy', fontweight='bold')
    ax.set_title('Accuracy vs Inference Time Trade-off', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, max(times) * 1.2])
    ax.set_ylim([0, 1.0])

    # Ideal region annotation
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(0.5, 0.95, 'Ideal Region\n(High Accuracy, Low Latency)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    save_figure(fig, 'fig7_time_accuracy_tradeoff')


# ============================================================================
# FIGURE 8: Component Contribution (Stacked Bar)
# ============================================================================
def generate_component_contribution():
    data = load_json("ablation_study.json")
    gains = data['incremental_gains']['gains']

    components = ['Logs\nContribution', 'Traces\nContribution', 'PCMCI\nContribution', 'Cross-Attention\nContribution']
    values = [gains['logs_contribution'], gains['traces_contribution'],
              gains['pcmci_contribution'], gains['cross_attention_contribution']]

    baseline = data['incremental_gains']['baseline_metrics_only']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Stacked bar
    bottom = baseline
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (comp, val, color) in enumerate(zip(components, values, colors)):
        ax.bar(0, val, bottom=bottom, color=color, label=comp.replace('\n', ' '),
               edgecolor='black', linewidth=1)
        ax.text(0, bottom + val/2, f'+{val:.3f}\n({val*100:.1f}%)',
                ha='center', va='center', fontweight='bold', fontsize=10)
        bottom += val

    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(0.5, baseline, f'Baseline (Metrics Only): {baseline:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('AC@1 Accuracy', fontweight='bold')
    ax.set_title('Incremental Component Contributions to Performance', fontweight='bold', pad=15)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0.5, 0.85])
    ax.set_xticks([])
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    save_figure(fig, 'fig8_component_contribution')


# ============================================================================
# FIGURE 9: Fusion Strategy Comparison
# ============================================================================
def generate_fusion_comparison():
    data = load_json("ablation_study.json")
    fusion = data['fusion_ablations']

    strategies = list(fusion.keys())
    ac1 = [fusion[s]['ac_at_1'] for s in strategies]
    ac3 = [fusion[s]['ac_at_3'] for s in strategies]
    times = [fusion[s]['inference_time_sec'] for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax1.bar(x - width/2, ac1, width, label='AC@1', color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, ac3, width, label='AC@3', color='#ff7f0e', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Fusion Strategy', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Fusion Strategy Performance', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Early', 'Late', 'Intermediate\n(Ours)'], fontsize=10)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0.6, 1.0])

    # Inference time comparison
    bars3 = ax2.bar(x, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Fusion Strategy', fontweight='bold')
    ax2.set_ylabel('Inference Time (sec)', fontweight='bold')
    ax2.set_title('Computational Cost', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Early', 'Late', 'Intermediate\n(Ours)'], fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, time) in enumerate(zip(bars3, times)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    save_figure(fig, 'fig9_fusion_comparison')


# ============================================================================
# FIGURE 10: Performance by System Scale
# ============================================================================
def generate_system_scale():
    data = load_json("performance_by_system.json")
    systems = data['systems']

    system_names = list(systems.keys())
    num_services = [systems[s]['num_services'] for s in system_names]
    ac1 = [systems[s]['ac_at_1'] for s in system_names]
    ac3 = [systems[s]['ac_at_3'] for s in system_names]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(num_services, ac1, s=300, marker='o', color='#1f77b4',
              label='AC@1', edgecolors='black', linewidth=2, alpha=0.7)
    ax.scatter(num_services, ac3, s=300, marker='s', color='#ff7f0e',
              label='AC@3', edgecolors='black', linewidth=2, alpha=0.7)

    for i, name in enumerate(system_names):
        ax.annotate(name, (num_services[i], ac1[i]), xytext=(10, -15),
                   textcoords='offset points', fontsize=10, fontweight='bold')

    # Trend line
    z = np.polyfit(num_services, ac1, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(num_services), max(num_services), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.4, linewidth=2, label='AC@1 Trend')

    ax.set_xlabel('Number of Services', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Performance vs System Scale (Service Count)', fontweight='bold', pad=15)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.7, 1.0])

    save_figure(fig, 'fig10_system_scale')


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("GENERATING ALL FIGURES")
    print("="*70)
    print()

    print("Figure 1: Baseline Comparison...")
    generate_baseline_comparison()

    print("Figure 2: Ablation Study - Incremental Gains...")
    generate_ablation_incremental()

    print("Figure 3: Performance by Fault Type...")
    generate_fault_type_heatmap()

    print("Figure 4: Single-Modality vs Multi-Modal...")
    generate_modality_radar()

    print("Figure 5: Dataset Statistics...")
    generate_dataset_statistics()

    print("Figure 6: Cross-Modal Attention...")
    generate_attention_heatmap()

    print("Figure 7: Time-Accuracy Tradeoff...")
    generate_time_accuracy_tradeoff()

    print("Figure 8: Component Contributions...")
    generate_component_contribution()

    print("Figure 9: Fusion Strategy Comparison...")
    generate_fusion_comparison()

    print("Figure 10: Performance vs System Scale...")
    generate_system_scale()

    print()
    print("="*70)
    print(f"✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"✓ Output directory: {FIGURES_DIR.absolute()}")
    print(f"✓ Total figures: 10 (each in PNG and PDF)")
    print("="*70)
    print()
    print("To regenerate with REAL data:")
    print("1. Replace JSON files in raw_results/ with your experimental results")
    print("2. Run: python generate_all_figures.py")
    print("3. All figures will be updated automatically!")
    print()

if __name__ == "__main__":
    main()
