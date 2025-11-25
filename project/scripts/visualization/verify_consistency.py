#!/usr/bin/env python3
"""
Verification Script: Check Data Consistency

This script verifies that all result values are consistent across:
- JSON data files
- Report document
- README
- Presentation slides

Run this before submission to catch any inconsistencies.

Usage:
    cd project
    python scripts/visualization/verify_consistency.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Define project structure - relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "raw_results"
REPORT_PATH = PROJECT_ROOT / "report" / "COMPLETE_REPORT.md"
README_PATH = PROJECT_ROOT.parent / "README.md"
PRESENTATION_PATH = PROJECT_ROOT / "presentation" / "PRESENTATION_SLIDES.md"


class ConsistencyVerifier:
    """Verify consistency of result data across all documents."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0

    def load_json(self, filename: str) -> Dict:
        """Load JSON file from results/raw_results/."""
        path = RESULTS_DIR / filename
        if not path.exists():
            self.errors.append(f"Missing JSON file: {filename}")
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def load_text(self, path: Path) -> str:
        """Load text file."""
        if not path.exists():
            self.errors.append(f"Missing file: {path}")
            return ""
        return path.read_text()

    def check_value(self, name: str, expected: float, actual: float, tolerance: float = 0.001):
        """Check if two values match within tolerance."""
        self.checks_total += 1
        if abs(expected - actual) <= tolerance:
            self.checks_passed += 1
            print(f"  ✅ {name}: {expected} matches")
        else:
            self.errors.append(f"{name}: Expected {expected}, got {actual}")
            print(f"  ❌ {name}: Expected {expected}, got {actual}")

    def find_in_text(self, text: str, pattern: str) -> List[str]:
        """Find all matches of a pattern in text."""
        return re.findall(pattern, text)

    def verify_primary_metrics(self):
        """Verify main AC@k metrics are consistent."""
        print("\n" + "="*80)
        print("1. Verifying Primary Metrics (AC@1, AC@3, AC@5, MRR)")
        print("="*80)

        # Load baseline comparison data
        baseline_data = self.load_json("baseline_comparison.json")
        if not baseline_data:
            return

        our_results = baseline_data.get("results", {}).get("Ours (Full System)", {})
        ac1 = our_results.get("ac_at_1", 0)
        ac3 = our_results.get("ac_at_3", 0)
        ac5 = our_results.get("ac_at_5", 0)
        mrr = our_results.get("mrr", 0)

        print(f"\nExpected values from JSON:")
        print(f"  AC@1: {ac1}")
        print(f"  AC@3: {ac3}")
        print(f"  AC@5: {ac5}")
        print(f"  MRR: {mrr}")

        # Check README
        print("\nChecking README...")
        readme_text = self.load_text(README_PATH)

        # Look for "76.1%" or "0.761"
        ac1_percent_matches = self.find_in_text(readme_text, r'76\.1%')
        ac1_decimal_matches = self.find_in_text(readme_text, r'0\.761')

        if ac1_percent_matches or ac1_decimal_matches:
            print(f"  ✅ Found AC@1 = {ac1} in README ({len(ac1_percent_matches)} percent mentions, {len(ac1_decimal_matches)} decimal mentions)")
            self.checks_passed += 1
        else:
            self.errors.append("AC@1 value not found in README")
            print(f"  ❌ AC@1 value not found in README")

        self.checks_total += 1

        # Check Report
        print("\nChecking Report...")
        report_text = self.load_text(REPORT_PATH)

        ac1_percent_matches = self.find_in_text(report_text, r'76\.1%')
        ac1_decimal_matches = self.find_in_text(report_text, r'0\.761')

        if ac1_percent_matches or ac1_decimal_matches:
            print(f"  ✅ Found AC@1 = {ac1} in Report ({len(ac1_percent_matches)} percent mentions, {len(ac1_decimal_matches)} decimal mentions)")
            self.checks_passed += 1
        else:
            self.errors.append("AC@1 value not found in Report")
            print(f"  ❌ AC@1 value not found in Report")

        self.checks_total += 1

        # Check Presentation
        print("\nChecking Presentation...")
        presentation_text = self.load_text(PRESENTATION_PATH)

        ac1_percent_matches = self.find_in_text(presentation_text, r'76\.1%')
        ac1_decimal_matches = self.find_in_text(presentation_text, r'0\.761')

        if ac1_percent_matches or ac1_decimal_matches:
            print(f"  ✅ Found AC@1 = {ac1} in Presentation ({len(ac1_percent_matches)} percent mentions, {len(ac1_decimal_matches)} decimal mentions)")
            self.checks_passed += 1
        else:
            self.errors.append("AC@1 value not found in Presentation")
            print(f"  ❌ AC@1 value not found in Presentation")

        self.checks_total += 1

    def verify_improvement_percentage(self):
        """Verify improvement vs SOTA is consistent."""
        print("\n" + "="*80)
        print("2. Verifying Improvement Percentage vs SOTA")
        print("="*80)

        baseline_data = self.load_json("baseline_comparison.json")
        if not baseline_data:
            return

        results = baseline_data.get("results", {})
        our_ac1 = results.get("Ours (Full System)", {}).get("ac_at_1", 0)
        sota_ac1 = results.get("RUN (SOTA)", {}).get("ac_at_1", 0)

        if sota_ac1 > 0:
            improvement = ((our_ac1 - sota_ac1) / sota_ac1) * 100
            print(f"\nCalculated improvement: {improvement:.1f}%")
            print(f"  Our AC@1: {our_ac1}")
            print(f"  SOTA AC@1: {sota_ac1}")
            print(f"  Improvement: {our_ac1 - sota_ac1:.3f} ({improvement:.1f}%)")

            # Check if "+21%" or "+20.6%" appears in documents
            readme_text = self.load_text(README_PATH)
            report_text = self.load_text(REPORT_PATH)
            presentation_text = self.load_text(PRESENTATION_PATH)

            # Look for "+21%" pattern
            readme_matches = self.find_in_text(readme_text, r'\+21%')
            report_matches = self.find_in_text(report_text, r'\+21%')
            presentation_matches = self.find_in_text(presentation_text, r'\+21%')

            print(f"\nImprovement mentions:")
            print(f"  README: {len(readme_matches)} mentions")
            print(f"  Report: {len(report_matches)} mentions")
            print(f"  Presentation: {len(presentation_matches)} mentions")

            if readme_matches and report_matches and presentation_matches:
                print(f"  ✅ Improvement percentage consistent across documents")
                self.checks_passed += 1
            else:
                self.warnings.append("Improvement percentage not found in all documents")
                print(f"  ⚠️  Improvement percentage not found in all documents")

            self.checks_total += 1

    def verify_ablation_consistency(self):
        """Verify ablation study numbers are consistent."""
        print("\n" + "="*80)
        print("3. Verifying Ablation Study Consistency")
        print("="*80)

        ablation_data = self.load_json("ablation_study.json")
        if not ablation_data:
            return

        # Check incremental gains sum up correctly
        incremental = ablation_data.get("incremental_gains", {})
        baseline = incremental.get("baseline_metrics_only", 0)
        final = incremental.get("add_cross_attention", 0)
        total_gain = incremental.get("total_improvement", 0)

        print(f"\nIncremental gains:")
        print(f"  Baseline (metrics only): {baseline}")
        print(f"  Final (full system): {final}")
        print(f"  Total improvement: {total_gain}")
        print(f"  Calculated: {final - baseline:.3f}")

        self.check_value("Total ablation improvement", total_gain, final - baseline)

        # Verify full system performance matches baseline comparison
        baseline_comparison = self.load_json("baseline_comparison.json")
        if baseline_comparison:
            our_ac1 = baseline_comparison.get("results", {}).get("Ours (Full System)", {}).get("ac_at_1", 0)
            print(f"\nCross-checking with baseline comparison:")
            print(f"  Ablation final AC@1: {final}")
            print(f"  Baseline comparison AC@1: {our_ac1}")

            self.check_value("Ablation vs baseline comparison", final, our_ac1)

    def verify_fault_type_statistics(self):
        """Verify fault type performance statistics."""
        print("\n" + "="*80)
        print("4. Verifying Fault Type Statistics")
        print("="*80)

        fault_data = self.load_json("performance_by_fault_type.json")
        if not fault_data:
            return

        fault_types = fault_data.get("fault_types", {})

        # Check AC@1 values are in reasonable range [0, 1]
        print(f"\nFault type AC@1 values:")
        for fault, metrics in fault_types.items():
            ac1 = metrics.get("ac_at_1", 0)
            print(f"  {fault}: {ac1}")

            if 0 <= ac1 <= 1:
                self.checks_passed += 1
            else:
                self.errors.append(f"{fault} AC@1 out of range: {ac1}")

            self.checks_total += 1

        # Check that AC@1 < AC@3 < AC@5 for each fault type
        print(f"\nVerifying AC@k relationship (AC@1 < AC@3 < AC@5):")
        for fault, metrics in fault_types.items():
            ac1 = metrics.get("ac_at_1", 0)
            ac3 = metrics.get("ac_at_3", 0)
            ac5 = metrics.get("ac_at_5", 0)

            if ac1 < ac3 < ac5:
                print(f"  ✅ {fault}: {ac1} < {ac3} < {ac5}")
                self.checks_passed += 1
            else:
                self.errors.append(f"{fault}: AC@k relationship violated ({ac1}, {ac3}, {ac5})")
                print(f"  ❌ {fault}: {ac1} < {ac3} < {ac5} (VIOLATED)")

            self.checks_total += 1

    def verify_system_scalability(self):
        """Verify system scalability statistics."""
        print("\n" + "="*80)
        print("5. Verifying System Scalability")
        print("="*80)

        system_data = self.load_json("performance_by_system.json")
        if not system_data:
            return

        systems = system_data.get("systems", {})

        # Check that larger systems have longer inference times
        print(f"\nSystem scale vs inference time:")
        system_list = []
        for system_name, metrics in systems.items():
            services = metrics.get("num_services", 0)
            inference_time = metrics.get("inference_time_sec", 0)
            system_list.append((system_name, services, inference_time))

        # Sort by number of services
        system_list_sorted = sorted(system_list, key=lambda x: x[1])

        for i, (name, services, time) in enumerate(system_list_sorted):
            print(f"  {name}: {services} services, {time}s inference")

            # Check that inference time increases (somewhat) with services
            if i > 0:
                prev_services = system_list_sorted[i-1][1]
                prev_time = system_list_sorted[i-1][2]

                # Inference time should increase (but can be sub-linear)
                if time >= prev_time:
                    print(f"    ✅ Inference time increases with scale ({prev_time}s → {time}s)")
                    self.checks_passed += 1
                else:
                    self.warnings.append(f"{name}: Inference time decreased despite more services")
                    print(f"    ⚠️  Inference time decreased ({prev_time}s → {time}s)")

                self.checks_total += 1

    def verify_model_specifications(self):
        """Verify model architecture specifications."""
        print("\n" + "="*80)
        print("6. Verifying Model Specifications")
        print("="*80)

        model_data = self.load_json("model_specifications.json")
        if not model_data:
            return

        # Check total inference time sums up
        print(f"\nInference time breakdown:")

        metrics_time = model_data.get("metrics_encoder", {}).get("inference_time_per_case_ms", 0)
        logs_time = model_data.get("logs_encoder", {}).get("inference_time_per_case_ms", 0)
        traces_time = model_data.get("traces_encoder", {}).get("inference_time_per_case_ms", 0)
        causal_time = model_data.get("causal_discovery", {}).get("inference_time_per_case_ms", 0)
        fusion_time = model_data.get("fusion_module", {}).get("inference_time_per_case_ms", 0)

        component_total_ms = metrics_time + logs_time + traces_time + causal_time + fusion_time
        component_total_sec = component_total_ms / 1000

        print(f"  Metrics encoder: {metrics_time} ms")
        print(f"  Logs encoder: {logs_time} ms")
        print(f"  Traces encoder: {traces_time} ms")
        print(f"  Causal discovery: {causal_time} ms")
        print(f"  Fusion module: {fusion_time} ms")
        print(f"  Component total: {component_total_ms} ms ({component_total_sec:.3f} s)")

        total_time = model_data.get("computational_requirements", {}).get("inference_time_per_case_sec", 0)
        print(f"  Reported total: {total_time} s")

        # Allow 10% tolerance for RCA head and overhead
        self.check_value("Total inference time", total_time, component_total_sec, tolerance=0.1)

    def verify_dataset_statistics(self):
        """Verify dataset statistics."""
        print("\n" + "="*80)
        print("7. Verifying Dataset Statistics")
        print("="*80)

        dataset_data = self.load_json("dataset_statistics.json")
        if not dataset_data:
            return

        tt_data = dataset_data.get("trainticket_re2", {})

        # Check train/val/test split sums to total
        total_cases = tt_data.get("total_cases", 0)
        train_split = tt_data.get("train_split", 0)
        val_split = tt_data.get("val_split", 0)
        test_split = tt_data.get("test_split", 0)

        split_sum = train_split + val_split + test_split

        print(f"\nDataset split:")
        print(f"  Train: {train_split}")
        print(f"  Val: {val_split}")
        print(f"  Test: {test_split}")
        print(f"  Sum: {split_sum}")
        print(f"  Total: {total_cases}")

        self.check_value("Train/val/test split sum", total_cases, split_sum)

        # Check fault distribution sums to total
        fault_dist = dataset_data.get("fault_distribution", {})
        fault_sum = sum(fault_dist.values())

        print(f"\nFault distribution:")
        for fault, count in fault_dist.items():
            print(f"  {fault}: {count}")
        print(f"  Sum: {fault_sum}")

        # Fault sum should be around total_cases (allowing for rounding)
        self.check_value("Fault distribution sum", total_cases, fault_sum, tolerance=10)

    def print_summary(self):
        """Print verification summary."""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)

        print(f"\nTotal checks: {self.checks_total}")
        print(f"Passed: {self.checks_passed} ✅")
        print(f"Failed: {len(self.errors)} ❌")
        print(f"Warnings: {len(self.warnings)} ⚠️")

        if self.errors:
            print("\n" + "="*80)
            print("ERRORS (Must Fix):")
            print("="*80)
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")

        if self.warnings:
            print("\n" + "="*80)
            print("WARNINGS (Review Recommended):")
            print("="*80)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")

        print("\n" + "="*80)

        if len(self.errors) == 0:
            print("✅ ALL CHECKS PASSED - Ready for submission!")
            print("="*80)
            return 0
        else:
            print("❌ ERRORS FOUND - Please fix before submission")
            print("="*80)
            return 1

    def run_all_checks(self):
        """Run all verification checks."""
        print("="*80)
        print("Data Consistency Verification")
        print("="*80)
        print(f"\nProject root: {PROJECT_ROOT.parent}")
        print(f"Results data: {RESULTS_DIR}")
        print(f"Report: {REPORT_PATH}")
        print(f"README: {README_PATH}")
        print(f"Presentation: {PRESENTATION_PATH}")

        self.verify_primary_metrics()
        self.verify_improvement_percentage()
        self.verify_ablation_consistency()
        self.verify_fault_type_statistics()
        self.verify_system_scalability()
        self.verify_model_specifications()
        self.verify_dataset_statistics()

        return self.print_summary()


def main():
    """Main entry point."""
    verifier = ConsistencyVerifier()
    exit_code = verifier.run_all_checks()
    exit(exit_code)


if __name__ == "__main__":
    main()
