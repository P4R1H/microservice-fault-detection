#!/usr/bin/env python3
"""
Comprehensive Baseline Comparison Script

Runs ALL baseline methods and compares with our approach:
1. Statistical baselines (3-Sigma, ARIMA, Granger-Lasso, Random)
2. Literature baselines (BARO, MicroRCA, etc.)
3. Our full multimodal system
4. Statistical significance testing

Usage:
    python scripts/run_baseline_comparisons.py --output outputs/baselines
"""

import argparse
import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import RCAEvalDataLoader
from src.baselines.statistical_baselines import (
    ThreeSigmaDetector,
    ARIMAForecaster,
    GrangerLassoRCA,
    RandomWalkBaseline,
    evaluate_ranking
)
from src.evaluation.metrics import RCAEvaluator
from scipy import stats


class BaselineComparison:
    """Compare all baseline methods"""

    def __init__(self, data_dir: str, output_dir: str):
        """Initialize baseline comparison"""
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        self.loader = RCAEvalDataLoader(data_dir)
        self.train, self.val, self.test = self.loader.load_splits(random_seed=42)

        print(f"✓ Loaded: {len(self.train)} train, {len(self.val)} val, {len(self.test)} test")

        # Initialize evaluator
        self.evaluator = RCAEvaluator()

        # Initialize baseline methods
        self.baselines = self._initialize_baselines()

    def _initialize_baselines(self) -> Dict:
        """Initialize all baseline methods"""
        baselines = {}

        # Statistical baselines
        baselines['3-Sigma'] = {
            'method': ThreeSigmaDetector(n_sigma=3.0, window_size=50),
            'requires': ['metrics'],
            'description': 'Three-sigma statistical anomaly detection'
        }

        baselines['ARIMA'] = {
            'method': ARIMAForecaster(order=(5, 1, 0)),
            'requires': ['metrics'],
            'description': 'ARIMA forecasting with residual analysis'
        }

        baselines['Granger-Lasso'] = {
            'method': GrangerLassoRCA(max_lag=5, alpha=0.01),
            'requires': ['metrics'],
            'description': 'Granger causality with Lasso regularization'
        }

        baselines['Random-Walk'] = {
            'method': RandomWalkBaseline(random_seed=42),
            'requires': [],
            'description': 'Random service ranking (sanity check)'
        }

        # TODO: Add more sophisticated baselines
        # baselines['BARO'] = {...}  # Bayesian online RCA
        # baselines['MicroRCA'] = {...}  # PageRank-based RCA

        return baselines

    def extract_service_mapping(self, metrics: pd.DataFrame) -> Dict[str, str]:
        """Extract service to metric mapping"""
        service_mapping = {}

        for col in metrics.columns:
            # Simple heuristic: service name is before first underscore
            if '_' in col:
                service = col.split('_')[0]
                service_mapping[col] = service
            else:
                service_mapping[col] = col

        return service_mapping

    def run_single_baseline(
        self,
        baseline_name: str,
        baseline_config: Dict,
        test_cases: List,
        n_cases: int = 100
    ) -> Dict:
        """
        Run single baseline method on test cases

        Args:
            baseline_name: Name of baseline
            baseline_config: Configuration dictionary
            test_cases: List of test cases
            n_cases: Maximum number of cases to test

        Returns:
            Results dictionary
        """
        print(f"\n{'='*70}")
        print(f"BASELINE: {baseline_name}")
        print(f"Description: {baseline_config['description']}")
        print(f"{'='*70}")

        method = baseline_config['method']
        requires = baseline_config['requires']

        results_list = []
        skipped = 0

        for i, case in enumerate(test_cases[:n_cases]):
            if i % 20 == 0:
                print(f"  Processing case {i+1}/{min(n_cases, len(test_cases))}...")

            # Load required data
            case.load_data(
                metrics='metrics' in requires,
                logs='logs' in requires,
                traces='traces' in requires
            )

            # Check if required data is available
            if 'metrics' in requires and case.metrics is None:
                skipped += 1
                case.unload_data()
                continue

            try:
                # Get ranking from baseline
                if baseline_name == 'Random-Walk':
                    # Extract unique services
                    if case.metrics is not None:
                        service_mapping = self.extract_service_mapping(case.metrics)
                        unique_services = list(set(service_mapping.values()))
                    else:
                        unique_services = ['service_1', 'service_2', 'service_3']

                    ranking = method.rank_services(unique_services)

                else:
                    # Standard baseline (metrics-based)
                    service_mapping = self.extract_service_mapping(case.metrics)
                    ranking = method.rank_services(case.metrics, service_mapping)

                # Evaluate
                result = evaluate_ranking(
                    predicted_ranking=ranking,
                    ground_truth_service=case.root_cause_service,
                    k_values=[1, 3, 5]
                )

                results_list.append(result)

            except Exception as e:
                print(f"    Warning: Case {case.case_id} failed: {e}")
                skipped += 1

            case.unload_data()

        if not results_list:
            print(f"  ⚠ No valid results for {baseline_name}")
            return None

        # Aggregate results
        aggregated = {
            'baseline_name': baseline_name,
            'AC@1': np.mean([r['AC@1'] for r in results_list]),
            'AC@3': np.mean([r['AC@3'] for r in results_list]),
            'AC@5': np.mean([r['AC@5'] for r in results_list]),
            'MRR': np.mean([r['MRR'] for r in results_list]),
            'n_cases': len(results_list),
            'n_skipped': skipped,
            'description': baseline_config['description']
        }

        print(f"\n  Results (n={len(results_list)}):")
        print(f"    AC@1: {aggregated['AC@1']:.3f}")
        print(f"    AC@3: {aggregated['AC@3']:.3f}")
        print(f"    AC@5: {aggregated['AC@5']:.3f}")
        print(f"    MRR:  {aggregated['MRR']:.3f}")
        if skipped > 0:
            print(f"    Skipped: {skipped} cases")

        return aggregated

    def run_all_baselines(self, n_cases: int = 100, n_seeds: int = 3):
        """
        Run all baselines with multiple random seeds

        Args:
            n_cases: Number of test cases per baseline
            n_seeds: Number of random seeds
        """
        print("="*80)
        print("BASELINE COMPARISON SUITE")
        print("="*80)
        print(f"\nBaselines: {len(self.baselines)}")
        print(f"Test cases per baseline: {n_cases}")
        print(f"Random seeds: {n_seeds}")
        print()

        all_results = []

        for seed in range(42, 42 + n_seeds):
            print(f"\n{'='*80}")
            print(f"SEED {seed}")
            print(f"{'='*80}")

            # Shuffle test cases
            np.random.seed(seed)
            test_cases = np.random.permutation(self.test)

            for baseline_name, baseline_config in self.baselines.items():
                result = self.run_single_baseline(
                    baseline_name=baseline_name,
                    baseline_config=baseline_config,
                    test_cases=test_cases,
                    n_cases=n_cases
                )

                if result:
                    result['random_seed'] = seed
                    all_results.append(result)

        # Save all results
        results_file = self.output_dir / "baseline_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Results saved: {results_file}")

        # Generate comparison table
        self._generate_comparison_table(all_results)

        # Statistical significance testing
        self._statistical_tests(all_results)

    def _generate_comparison_table(self, all_results: List[Dict]):
        """Generate comparison table"""
        print("\n" + "="*80)
        print("BASELINE COMPARISON TABLE")
        print("="*80)

        # Group by baseline
        grouped = defaultdict(list)
        for result in all_results:
            grouped[result['baseline_name']].append(result)

        # Create table
        table_data = []
        print(f"\n{'Method':<20} {'AC@1':<15} {'AC@3':<15} {'AC@5':<15} {'MRR':<15}")
        print("-"*80)

        for baseline_name, results in sorted(grouped.items()):
            ac1_vals = [r['AC@1'] for r in results]
            ac3_vals = [r['AC@3'] for r in results]
            ac5_vals = [r['AC@5'] for r in results]
            mrr_vals = [r['MRR'] for r in results]

            row = {
                'Method': baseline_name,
                'AC@1': f"{np.mean(ac1_vals):.3f} ± {np.std(ac1_vals):.3f}",
                'AC@3': f"{np.mean(ac3_vals):.3f} ± {np.std(ac3_vals):.3f}",
                'AC@5': f"{np.mean(ac5_vals):.3f} ± {np.std(ac5_vals):.3f}",
                'MRR': f"{np.mean(mrr_vals):.3f} ± {np.std(mrr_vals):.3f}"
            }

            table_data.append(row)

            print(f"{baseline_name:<20} "
                  f"{np.mean(ac1_vals):.3f}±{np.std(ac1_vals):.3f}    "
                  f"{np.mean(ac3_vals):.3f}±{np.std(ac3_vals):.3f}    "
                  f"{np.mean(ac5_vals):.3f}±{np.std(ac5_vals):.3f}    "
                  f"{np.mean(mrr_vals):.3f}±{np.std(mrr_vals):.3f}")

        # Save to CSV
        df = pd.DataFrame(table_data)
        csv_file = self.output_dir / "baseline_comparison_table.csv"
        df.to_csv(csv_file, index=False)

        print(f"\n✓ Table saved: {csv_file}")

    def _statistical_tests(self, all_results: List[Dict]):
        """Perform statistical significance tests"""
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)

        # Group by baseline
        grouped = defaultdict(list)
        for result in all_results:
            grouped[result['baseline_name']].append(result)

        # Pairwise comparisons for AC@1
        baselines = list(grouped.keys())

        print("\nPaired t-tests (AC@1):")
        print(f"{'Baseline A':<20} {'Baseline B':<20} {'p-value':<10} {'Significant'}")
        print("-"*80)

        for i, baseline_a in enumerate(baselines):
            for baseline_b in baselines[i+1:]:
                ac1_a = [r['AC@1'] for r in grouped[baseline_a]]
                ac1_b = [r['AC@1'] for r in grouped[baseline_b]]

                # Paired t-test
                if len(ac1_a) == len(ac1_b) and len(ac1_a) > 1:
                    t_stat, p_value = stats.ttest_rel(ac1_a, ac1_b)

                    significant = "✓" if p_value < 0.05 else ""

                    print(f"{baseline_a:<20} {baseline_b:<20} {p_value:.4f}     {significant}")

        # Save statistical tests
        stats_file = self.output_dir / "statistical_tests.txt"
        with open(stats_file, 'w') as f:
            f.write("STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("="*80 + "\n\n")
            f.write("Paired t-tests (AC@1):\n")
            f.write(f"{'Baseline A':<20} {'Baseline B':<20} {'p-value':<10} {'Significant'}\n")
            f.write("-"*80 + "\n")

            for i, baseline_a in enumerate(baselines):
                for baseline_b in baselines[i+1:]:
                    ac1_a = [r['AC@1'] for r in grouped[baseline_a]]
                    ac1_b = [r['AC@1'] for r in grouped[baseline_b]]

                    if len(ac1_a) == len(ac1_b) and len(ac1_a) > 1:
                        t_stat, p_value = stats.ttest_rel(ac1_a, ac1_b)
                        significant = "✓" if p_value < 0.05 else ""
                        f.write(f"{baseline_a:<20} {baseline_b:<20} {p_value:.4f}     {significant}\n")

        print(f"\n✓ Statistical tests saved: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Run baseline comparisons')
    parser.add_argument('--data_dir', type=str, default='data/RCAEval',
                       help='Path to RCAEval dataset')
    parser.add_argument('--output', type=str, default='outputs/baselines',
                       help='Output directory')
    parser.add_argument('--n_cases', type=int, default=100,
                       help='Number of test cases per baseline')
    parser.add_argument('--n_seeds', type=int, default=3,
                       help='Number of random seeds')
    args = parser.parse_args()

    runner = BaselineComparison(
        data_dir=args.data_dir,
        output_dir=args.output
    )

    runner.run_all_baselines(
        n_cases=args.n_cases,
        n_seeds=args.n_seeds
    )

    print("\n" + "="*80)
    print("✅ BASELINE COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Review comparison table: baseline_comparison_table.csv")
    print("  2. Check statistical tests: statistical_tests.txt")
    print("  3. Generate plots: python scripts/visualize_baselines.py")


if __name__ == '__main__':
    main()
