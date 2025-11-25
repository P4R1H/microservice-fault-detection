#!/usr/bin/env python3
"""
Generate ALL result tables from JSON data files.

This script creates markdown tables, CSV files, and LaTeX tables
from the experimental results in raw_results/*.json

Usage:
    python generate_all_tables.py

Output:
    - Markdown tables for report
    - CSV files for easy viewing
    - LaTeX tables for paper submission
"""

import json
import pandas as pd
from pathlib import Path

# Directories
RAW_RESULTS = Path("raw_results")
TABLES_DIR = Path("tables")
TABLES_DIR.mkdir(exist_ok=True)

def load_json(filename):
    """Load JSON data file."""
    with open(RAW_RESULTS / filename, 'r') as f:
        return json.load(f)

def save_table(df, name, caption=""):
    """Save table in multiple formats."""
    # CSV
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)

    # Markdown
    with open(TABLES_DIR / f"{name}.md", 'w') as f:
        if caption:
            f.write(f"**{caption}**\n\n")
        f.write(df.to_markdown(index=False))

    # LaTeX
    latex = df.to_latex(index=False, float_format="%.3f", caption=caption, label=f"tab:{name}")
    with open(TABLES_DIR / f"{name}.tex", 'w') as f:
        f.write(latex)

    print(f"✓ Saved {name} (CSV, MD, TEX)")


# ============================================================================
# TABLE 1: Baseline Comparison
# ============================================================================
def generate_baseline_table():
    data = load_json("baseline_comparison.json")
    results = data['results']

    rows = []
    for method, metrics in results.items():
        rows.append({
            'Method': method,
            'AC@1': f"{metrics['ac_at_1']:.3f}",
            'AC@3': f"{metrics['ac_at_3']:.3f}",
            'AC@5': f"{metrics['ac_at_5']:.3f}",
            'MRR': f"{metrics['mrr']:.3f}",
            'Time (s)': f"{metrics['inference_time_sec']:.3f}"
        })

    df = pd.DataFrame(rows)
    save_table(df, "table1_baseline_comparison",
               "Table 1: Comparison with baseline methods on RCAEval TrainTicket RE2")


# ============================================================================
# TABLE 2: Ablation Study Results
# ============================================================================
def generate_ablation_table():
    data = load_json("ablation_study.json")
    modality_abl = data['modality_ablations']

    rows = []
    baseline_ac1 = modality_abl['Metrics Only (Chronos)']['ac_at_1']

    for config, metrics in modality_abl.items():
        improvement = ((metrics['ac_at_1'] - baseline_ac1) / baseline_ac1) * 100

        rows.append({
            'Configuration': config,
            'AC@1': f"{metrics['ac_at_1']:.3f}",
            'AC@3': f"{metrics['ac_at_3']:.3f}",
            'AC@5': f"{metrics['ac_at_5']:.3f}",
            'Δ vs Baseline': f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
        })

    df = pd.DataFrame(rows)
    save_table(df, "table2_ablation_study",
               "Table 2: Ablation study - incremental component contributions")


# ============================================================================
# TABLE 3: Performance by Fault Type
# ============================================================================
def generate_fault_type_table():
    data = load_json("performance_by_fault_type.json")
    faults = data['fault_types']

    rows = []
    for fault_name, metrics in faults.items():
        rows.append({
            'Fault Type': fault_name,
            'Cases': metrics['num_cases'],
            'AC@1': f"{metrics['ac_at_1']:.3f}",
            'AC@3': f"{metrics['ac_at_3']:.3f}",
            'AC@5': f"{metrics['ac_at_5']:.3f}",
            'MRR': f"{metrics['mrr']:.3f}"
        })

    df = pd.DataFrame(rows)
    save_table(df, "table3_performance_by_fault_type",
               "Table 3: Performance breakdown by fault injection type")


# ============================================================================
# TABLE 4: Performance by System
# ============================================================================
def generate_system_table():
    data = load_json("performance_by_system.json")
    systems = data['systems']

    rows = []
    for system_name, metrics in systems.items():
        rows.append({
            'System': system_name,
            'Services': metrics['num_services'],
            'Cases': metrics['num_test_cases'],
            'AC@1': f"{metrics['ac_at_1']:.3f}",
            'AC@3': f"{metrics['ac_at_3']:.3f}",
            'AC@5': f"{metrics['ac_at_5']:.3f}",
            'Time (s)': f"{metrics['avg_inference_time']:.3f}"
        })

    df = pd.DataFrame(rows)
    save_table(df, "table4_performance_by_system",
               "Table 4: Performance across different microservice systems")


# ============================================================================
# TABLE 5: Encoder Comparison
# ============================================================================
def generate_encoder_table():
    data = load_json("ablation_study.json")
    encoders = data['encoder_ablations']

    # Add full system for comparison
    full_system = data['modality_ablations']['Full System (All + PCMCI + Cross-Attention)']
    encoders_with_baseline = {
        'Full System (Chronos + GCN + Drain3)': full_system,
        **encoders
    }

    rows = []
    for encoder_name, metrics in encoders_with_baseline.items():
        rows.append({
            'Encoder Configuration': encoder_name,
            'AC@1': f"{metrics['ac_at_1']:.3f}",
            'AC@3': f"{metrics['ac_at_3']:.3f}",
            'AC@5': f"{metrics['ac_at_5']:.3f}",
            'Time (s)': f"{metrics['inference_time_sec']:.3f}"
        })

    df = pd.DataFrame(rows)
    save_table(df, "table5_encoder_comparison",
               "Table 5: Comparison of different encoder architectures")


# ============================================================================
# TABLE 6: Fusion Strategy Comparison
# ============================================================================
def generate_fusion_table():
    data = load_json("ablation_study.json")
    fusion = data['fusion_ablations']

    rows = []
    for strategy, metrics in fusion.items():
        rows.append({
            'Fusion Strategy': strategy,
            'AC@1': f"{metrics['ac_at_1']:.3f}",
            'AC@3': f"{metrics['ac_at_3']:.3f}",
            'AC@5': f"{metrics['ac_at_5']:.3f}",
            'Time (s)': f"{metrics['inference_time_sec']:.3f}"
        })

    df = pd.DataFrame(rows)
    save_table(df, "table6_fusion_strategies",
               "Table 6: Comparison of multimodal fusion strategies")


# ============================================================================
# TABLE 7: Model Specifications
# ============================================================================
def generate_model_specs_table():
    data = load_json("model_specifications.json")

    specs = [
        ['Component', 'Specification', 'Details'],
        ['Metrics Encoder', 'Chronos-Bolt-Tiny', f"{data['metrics_encoder']['parameters']} params, {data['metrics_encoder']['model_size_mb']}MB"],
        ['Logs Encoder', 'Drain3 + TF-IDF', f"{data['logs_encoder']['vocabulary_size']} templates"],
        ['Traces Encoder', '2-layer GCN', f"{data['traces_encoder']['hidden_dim']}d hidden, mean aggregation"],
        ['Causal Discovery', 'PCMCI', f"tau_max={data['causal_discovery']['tau_max']}, ParCorr test"],
        ['Fusion', 'Cross-Attention', f"{data['fusion_module']['num_heads']} heads, {data['fusion_module']['num_layers']} layers"],
        ['Training', 'AdamW', f"LR={data['training']['learning_rate']}, {data['training']['num_epochs']} epochs"],
        ['Total Parameters', '-', f"{data['training']['total_parameters']} ({data['training']['trainable_parameters']} trainable)"],
    ]

    df = pd.DataFrame(specs[1:], columns=specs[0])
    save_table(df, "table7_model_specifications",
               "Table 7: Complete model architecture and hyperparameters")


# ============================================================================
# TABLE 8: Dataset Statistics
# ============================================================================
def generate_dataset_stats_table():
    data = load_json("dataset_statistics.json")
    tt_re2 = data['trainticket_re2']

    stats = [
        ['Statistic', 'Value'],
        ['Total Cases', f"{tt_re2['total_cases']}"],
        ['Train / Val / Test Split', f"{tt_re2['train_split']} / {tt_re2['val_split']} / {tt_re2['test_split']}"],
        ['Number of Services', f"{tt_re2['num_services']}"],
        ['Metrics per Service', f"{tt_re2['metrics']['num_metrics_per_service']}"],
        ['Total Metrics', f"{tt_re2['metrics']['total_metrics']}"],
        ['Avg Logs per Case', f"{tt_re2['logs']['avg_logs_per_case']:,}"],
        ['Avg Spans per Case', f"{tt_re2['traces']['avg_spans_per_case']:,}"],
        ['Case Duration (avg)', f"{tt_re2['avg_case_duration_min']} minutes"],
    ]

    df = pd.DataFrame(stats[1:], columns=stats[0])
    save_table(df, "table8_dataset_statistics",
               "Table 8: RCAEval TrainTicket RE2 dataset statistics")


# ============================================================================
# TABLE 9: Statistical Significance Tests
# ============================================================================
def generate_statistical_tests_table():
    data = load_json("baseline_comparison.json")
    sig_tests = data['statistical_significance']

    rows = []
    for comparison, results in sig_tests.items():
        comparison_name = comparison.replace('_', ' ').title()
        rows.append({
            'Comparison': comparison_name,
            'p-value': f"{results['p_value_ac_at_1']:.4f}",
            "Cohen's d": f"{results['effect_size_cohens_d']:.2f}",
            'Significant (α=0.05)': 'Yes ✓' if results['significant'] else 'No'
        })

    df = pd.DataFrame(rows)
    save_table(df, "table9_statistical_significance",
               "Table 9: Statistical significance tests (paired t-test, AC@1)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print("GENERATING ALL RESULT TABLES")
    print("="*70)
    print()

    print("Table 1: Baseline Comparison...")
    generate_baseline_table()

    print("Table 2: Ablation Study...")
    generate_ablation_table()

    print("Table 3: Performance by Fault Type...")
    generate_fault_type_table()

    print("Table 4: Performance by System...")
    generate_system_table()

    print("Table 5: Encoder Comparison...")
    generate_encoder_table()

    print("Table 6: Fusion Strategies...")
    generate_fusion_table()

    print("Table 7: Model Specifications...")
    generate_model_specs_table()

    print("Table 8: Dataset Statistics...")
    generate_dataset_stats_table()

    print("Table 9: Statistical Significance...")
    generate_statistical_tests_table()

    print()
    print("="*70)
    print(f"✓ ALL TABLES GENERATED SUCCESSFULLY!")
    print(f"✓ Output directory: {TABLES_DIR.absolute()}")
    print(f"✓ Total tables: 9 (each in CSV, MD, TEX)")
    print(f"✓ Total files: {len(list(TABLES_DIR.glob('*')))}")
    print("="*70)
    print()
    print("To regenerate with REAL data:")
    print("1. Replace JSON files in raw_results/ with your experimental results")
    print("2. Run: python generate_all_tables.py")
    print("3. All tables will be updated automatically!")
    print()

if __name__ == "__main__":
    main()
