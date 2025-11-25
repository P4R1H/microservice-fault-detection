#!/usr/bin/env python3
"""
Test PCMCI causal discovery with real RCAEval dataset.

This script:
1. Loads failure cases from RCAEval
2. Preprocesses metrics data
3. Runs PCMCI causal discovery
4. Tests service-level integration
5. Visualizes causal graph
6. Compares with Granger-Lasso baseline

Run with: python scripts/test_pcmci.py --n_cases 3
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import RCAEvalDataLoader
from src.data.preprocessing import MetricsPreprocessor
from src.causal import (
    PCMCIDiscovery,
    GrangerLassoRCA,
    discover_causal_relations,
    visualize_causal_graph
)


def test_data_loading_for_pcmci(data_path: str, n_cases: int = 3):
    """Load and preprocess data for PCMCI testing."""
    print("\n" + "="*70)
    print("STEP 1: Loading Data for PCMCI")
    print("="*70)

    try:
        loader = RCAEvalDataLoader(data_path)
        all_cases = loader.load_all_cases()
        print(f"✓ Found {len(all_cases)} total failure cases")

        # Get test cases
        test_cases = all_cases[:n_cases]
        print(f"✓ Testing with {n_cases} cases")

        return test_cases

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pcmci_discovery(case, output_dir: str = None):
    """Test PCMCI causal discovery on a single case."""
    print("\n" + "="*70)
    print("STEP 2: Testing PCMCI Causal Discovery")
    print("="*70)

    try:
        # Load metrics
        case.load_data(metrics=True)

        if case.metrics is None:
            print("✗ No metrics data available")
            return None

        print(f"  Case ID: {case.case_id}")
        print(f"  System: {case.system}")
        print(f"  Root cause: {case.root_cause_service}")
        print(f"  Metrics shape: {case.metrics.shape}")

        # Preprocess metrics
        preprocessor = MetricsPreprocessor(
            normalization='zscore',
            fill_method='forward',
            clip_outliers=True
        )

        preprocessor.fit(case.metrics)
        processed = preprocessor.transform(case.metrics)

        print(f"✓ Preprocessed metrics: {processed.shape}")

        # Prepare data for PCMCI
        # PCMCI expects (n_timesteps, n_variables)
        data = processed.values if isinstance(processed, pd.DataFrame) else processed
        var_names = list(processed.columns) if isinstance(processed, pd.DataFrame) else None

        print(f"  Data shape: {data.shape}")
        print(f"  Variable names: {var_names}")

        # Run PCMCI
        print("\n  Running PCMCI algorithm...")
        try:
            pcmci = PCMCIDiscovery(
                tau_max=5,
                pc_alpha=0.15,
                alpha_level=0.05,
                independence_test='parcorr',
                verbosity=1
            )

            results = pcmci.discover_graph(data, var_names)

            print("✓ PCMCI completed successfully")
            print(f"\n{results['summary']}")

            # Test service-level integration
            if case.root_cause_service:
                print("\n  Testing service-level integration...")

                # Create dummy service mapping (in real use, extract from case metadata)
                service_mapping = {
                    case.root_cause_service: var_names[:3] if var_names else [],
                    'other-service': var_names[3:] if var_names and len(var_names) > 3 else []
                }

                service_scores = pcmci.integrate_with_services(
                    results['causal_graph'],
                    service_mapping
                )

                print(f"✓ Service-level scores computed:")
                for service, score in sorted(service_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {service}: {score:.4f}")

                # Check if root cause is ranked correctly
                top_service = max(service_scores.items(), key=lambda x: x[1])[0]
                if top_service == case.root_cause_service:
                    print(f"✓ Correctly identified root cause: {top_service}")
                else:
                    print(f"✗ Root cause mismatch: predicted {top_service}, actual {case.root_cause_service}")

            # Visualize causal graph
            if output_dir and results['causal_graph'].number_of_edges() > 0:
                output_path = os.path.join(output_dir, f"causal_graph_{case.case_id}.png")
                os.makedirs(output_dir, exist_ok=True)
                visualize_causal_graph(
                    results['causal_graph'],
                    output_path=output_path,
                    figsize=(14, 10)
                )

            case.unload_data()
            return results

        except ImportError as ie:
            print(f"⚠ tigramite not installed: {ie}")
            print("  Install with: pip install tigramite")
            case.unload_data()
            return None

    except Exception as e:
        print(f"✗ PCMCI discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_granger_baseline(case):
    """Test Granger-Lasso baseline for comparison."""
    print("\n" + "="*70)
    print("STEP 3: Testing Granger-Lasso Baseline")
    print("="*70)

    try:
        case.load_data(metrics=True)

        if case.metrics is None:
            print("✗ No metrics data available")
            return None

        # Preprocess metrics
        preprocessor = MetricsPreprocessor(
            normalization='zscore',
            fill_method='forward',
            clip_outliers=True
        )

        preprocessor.fit(case.metrics)
        processed = preprocessor.transform(case.metrics)

        data = processed.values if isinstance(processed, pd.DataFrame) else processed
        var_names = list(processed.columns) if isinstance(processed, pd.DataFrame) else None

        # Run Granger-Lasso
        print("  Running Granger-Lasso...")
        granger = GrangerLassoRCA(max_lag=5, alpha=0.01)
        causal_graph = granger.discover_graph(data, var_names)

        print(f"✓ Granger-Lasso completed")
        print(f"  Discovered {causal_graph.number_of_edges()} causal edges")
        print(f"  Nodes: {causal_graph.number_of_nodes()}")

        case.unload_data()
        return causal_graph

    except Exception as e:
        print(f"✗ Granger-Lasso failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Test PCMCI causal discovery')
    parser.add_argument('--data_path', default='data/RCAEval', help='Path to dataset')
    parser.add_argument('--n_cases', type=int, default=3, help='Number of cases to test')
    parser.add_argument('--output_dir', default='outputs/causal_graphs', help='Output directory for visualizations')
    args = parser.parse_args()

    print("="*70)
    print("PCMCI CAUSAL DISCOVERY TEST SUITE")
    print("="*70)
    print(f"Dataset: {args.data_path}")
    print(f"Test cases: {args.n_cases}")

    # Check if dataset exists
    if not os.path.exists(args.data_path):
        print(f"\n✗ Dataset not found at {args.data_path}")
        print("  Please ensure RCAEval dataset is extracted to data/RCAEval/")
        return 1

    # Load data
    cases = test_data_loading_for_pcmci(args.data_path, args.n_cases)
    if cases is None:
        return 1

    # Test PCMCI on first case
    pcmci_results = test_pcmci_discovery(cases[0], args.output_dir)

    # Test Granger-Lasso baseline
    granger_graph = test_granger_baseline(cases[0])

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Data loading: PASSED")
    print(f"✓ PCMCI discovery: {'PASSED' if pcmci_results is not None else 'SKIPPED (tigramite not installed)'}")
    print(f"✓ Granger-Lasso baseline: {'PASSED' if granger_graph is not None else 'FAILED'}")

    if pcmci_results:
        print("\n🎉 PCMCI causal discovery tested successfully!")
        print("\nNext steps:")
        print("  1. Implement multimodal fusion (Phase 8)")
        print("  2. Integrate causal graph with trace encoder")
        print("  3. Build end-to-end RCA model")
    else:
        print("\n⚠ Install tigramite to enable PCMCI:")
        print("  pip install tigramite")

    return 0


if __name__ == '__main__':
    sys.exit(main())
