"""
Test Script: Statistical Baselines

Tests all statistical baseline methods on real RCAEval data.

Usage:
    cd project
    python tests/test_baselines.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import RCAEvalDataLoader
from src.baselines import (
    ThreeSigmaDetector,
    ARIMAForecaster,
    GrangerLassoRCA,
    RandomWalkBaseline,
    evaluate_ranking
)


def test_baselines():
    """Test statistical baselines on RCAEval data"""

    print("=" * 80)
    print("BASELINE TESTING ON RCAEVAL DATASET")
    print("=" * 80)

    # Load dataset
    print("\n📂 Loading dataset...")
    try:
        loader = RCAEvalDataLoader('data/RCAEval')
        train, val, test = loader.load_splits(random_seed=42)
        print(f"✅ Loaded: {len(train)} train, {len(val)} val, {len(test)} test cases")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

    # Use validation set for testing (smaller, faster)
    test_cases = val[:10]  # Test on first 10 validation cases
    print(f"\n🧪 Testing on {len(test_cases)} cases from validation set")

    # Initialize baseline methods
    print("\n🔧 Initializing baseline methods...")
    three_sigma = ThreeSigmaDetector(n_sigma=3.0, window_size=50)
    arima = ARIMAForecaster(order=(5, 1, 0))
    granger = GrangerLassoRCA(max_lag=3, alpha=0.01)
    random = RandomWalkBaseline(random_seed=42)

    # Results storage
    results = {
        '3-Sigma': {'AC@1': [], 'AC@3': [], 'AC@5': [], 'MRR': []},
        'ARIMA': {'AC@1': [], 'AC@3': [], 'AC@5': [], 'MRR': []},
        'Granger-Lasso': {'AC@1': [], 'AC@3': [], 'AC@5': [], 'MRR': []},
        'Random Walk': {'AC@1': [], 'AC@3': [], 'AC@5': [], 'MRR': []}
    }

    print("\n" + "=" * 80)
    print("RUNNING BASELINES")
    print("=" * 80)

    for i, case in enumerate(test_cases):
        print(f"\n📦 Case {i+1}/{len(test_cases)}: {case.case_id}")
        print(f"   System: {case.system}")
        print(f"   Fault: {case.fault_type}")
        print(f"   Ground Truth: {case.root_cause_service}")

        # Check if metrics file exists
        if not case.has_metrics():
            print("   ⚠️  No metrics file - skipping")
            continue

        # LAZY LOAD: Load metrics data
        try:
            case.load_data(metrics=True, logs=False, traces=False)
        except Exception as e:
            print(f"   ⚠️  Error loading metrics: {e}")
            continue

        # Check if sufficient data
        if case.metrics is None or len(case.metrics) < 50:
            print(f"   ⚠️  Insufficient metrics data ({len(case.metrics) if case.metrics is not None else 0} timesteps < 50) - skipping")
            case.unload_data()  # Free memory
            continue

        # Extract metrics
        metrics = case.metrics

        # Create service mapping (simplified - map metric names to services)
        # In real implementation, this should come from metadata
        service_names = list(set([col.split('_')[0] if '_' in col else col for col in metrics.columns]))
        service_mapping = {col: col.split('_')[0] if '_' in col else col for col in metrics.columns}

        # Test 1: Three-Sigma
        try:
            ranking = three_sigma.rank_services(metrics, service_mapping)
            if ranking:
                metrics_result = evaluate_ranking(ranking, case.root_cause_service)
                results['3-Sigma']['AC@1'].append(metrics_result['AC@1'])
                results['3-Sigma']['AC@3'].append(metrics_result['AC@3'])
                results['3-Sigma']['AC@5'].append(metrics_result['AC@5'])
                results['3-Sigma']['MRR'].append(metrics_result['MRR'])
                print(f"   3-Sigma: AC@1={metrics_result['AC@1']:.2f}, MRR={metrics_result['MRR']:.3f}")
        except Exception as e:
            print(f"   3-Sigma: ❌ Error - {e}")

        # Test 2: ARIMA
        try:
            ranking = arima.rank_services(metrics, service_mapping)
            if ranking:
                metrics_result = evaluate_ranking(ranking, case.root_cause_service)
                results['ARIMA']['AC@1'].append(metrics_result['AC@1'])
                results['ARIMA']['AC@3'].append(metrics_result['AC@3'])
                results['ARIMA']['AC@5'].append(metrics_result['AC@5'])
                results['ARIMA']['MRR'].append(metrics_result['MRR'])
                print(f"   ARIMA: AC@1={metrics_result['AC@1']:.2f}, MRR={metrics_result['MRR']:.3f}")
        except Exception as e:
            print(f"   ARIMA: ❌ Error - {e}")

        # Test 3: Granger-Lasso (slower, limit variables)
        try:
            ranking = granger.rank_services(metrics, service_mapping, max_vars=15)
            if ranking:
                metrics_result = evaluate_ranking(ranking, case.root_cause_service)
                results['Granger-Lasso']['AC@1'].append(metrics_result['AC@1'])
                results['Granger-Lasso']['AC@3'].append(metrics_result['AC@3'])
                results['Granger-Lasso']['AC@5'].append(metrics_result['AC@5'])
                results['Granger-Lasso']['MRR'].append(metrics_result['MRR'])
                print(f"   Granger: AC@1={metrics_result['AC@1']:.2f}, MRR={metrics_result['MRR']:.3f}")
        except Exception as e:
            print(f"   Granger: ❌ Error - {e}")

        # Test 4: Random Walk
        try:
            ranking = random.rank_services(service_names)
            metrics_result = evaluate_ranking(ranking, case.root_cause_service)
            results['Random Walk']['AC@1'].append(metrics_result['AC@1'])
            results['Random Walk']['AC@3'].append(metrics_result['AC@3'])
            results['Random Walk']['AC@5'].append(metrics_result['AC@5'])
            results['Random Walk']['MRR'].append(metrics_result['MRR'])
            print(f"   Random: AC@1={metrics_result['AC@1']:.2f}, MRR={metrics_result['MRR']:.3f}")
        except Exception as e:
            print(f"   Random: ❌ Error - {e}")

        # Unload data to free memory
        case.unload_data()

    # Aggregate results
    print("\n" + "=" * 80)
    print("BASELINE PERFORMANCE SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<20} {'AC@1':<8} {'AC@3':<8} {'AC@5':<8} {'MRR':<8} {'N':<8}")
    print("-" * 80)

    for method, metrics in results.items():
        if metrics['AC@1']:
            ac1 = np.mean(metrics['AC@1'])
            ac3 = np.mean(metrics['AC@3'])
            ac5 = np.mean(metrics['AC@5'])
            mrr = np.mean(metrics['MRR'])
            n = len(metrics['AC@1'])

            print(f"{method:<20} {ac1:<8.3f} {ac3:<8.3f} {ac5:<8.3f} {mrr:<8.3f} {n:<8}")
        else:
            print(f"{method:<20} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'0':<8}")

    print("\n" + "=" * 80)
    print("✅ BASELINE TESTING COMPLETE!")
    print("=" * 80)

    # Save results
    output_dir = Path('project/outputs/baseline_tests')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        method: {
            'AC@1': np.mean(metrics['AC@1']) if metrics['AC@1'] else 0.0,
            'AC@3': np.mean(metrics['AC@3']) if metrics['AC@3'] else 0.0,
            'AC@5': np.mean(metrics['AC@5']) if metrics['AC@5'] else 0.0,
            'MRR': np.mean(metrics['MRR']) if metrics['MRR'] else 0.0,
            'N': len(metrics['AC@1'])
        }
        for method, metrics in results.items()
    }).T

    results_file = output_dir / 'baseline_results.csv'
    results_df.to_csv(results_file)
    print(f"\n💾 Results saved to: {results_file}")

    print("\n📊 Next Steps:")
    print("   1. Run EDA: python scripts/eda_analysis.py --all")
    print("   2. Review baseline performance")
    print("   3. Begin implementing advanced methods (Chronos, GCN, PCMCI)")

    return True


if __name__ == '__main__':
    success = test_baselines()
    sys.exit(0 if success else 1)
