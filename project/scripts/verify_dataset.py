"""
Dataset Verification Script

Verifies that the RCAEval dataset is properly loaded and shows detailed statistics.

Usage:
    cd project
    python scripts/verify_dataset.py
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import RCAEvalDataLoader


def verify_dataset():
    """Verify RCAEval dataset and show comprehensive statistics"""

    print("=" * 80)
    print("RCAEVAL DATASET VERIFICATION")
    print("=" * 80)

    # Check if dataset exists
    data_path = Path('data/RCAEval')
    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path.absolute()}")
        print("\nExpected directory structure:")
        print("  project/")
        print("    data/")
        print("      RCAEval/")
        print("        TrainTicket/")
        print("        SockShop/")
        print("        OnlineBoutique/")
        return False

    print(f"\n✅ Dataset directory found: {data_path.absolute()}")

    # Load dataset
    print("\n📂 Loading dataset...")
    try:
        loader = RCAEvalDataLoader('data/RCAEval')
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

    # Get all cases
    print("\n📊 Discovering all failure cases...")
    train, val, test = loader.load_splits(random_seed=42)
    all_cases = train + val + test

    print(f"\n✅ Total cases discovered: {len(all_cases)}")
    print(f"   Train: {len(train)} ({len(train)/len(all_cases)*100:.1f}%)")
    print(f"   Val:   {len(val)} ({len(val)/len(all_cases)*100:.1f}%)")
    print(f"   Test:  {len(test)} ({len(test)/len(all_cases)*100:.1f}%)")

    # Statistics by system
    print("\n" + "=" * 80)
    print("CASES BY SYSTEM")
    print("=" * 80)

    system_counts = Counter(case.system for case in all_cases)
    for system, count in sorted(system_counts.items()):
        print(f"  {system:20s}: {count:3d} cases ({count/len(all_cases)*100:5.1f}%)")

    # Statistics by fault type
    print("\n" + "=" * 80)
    print("CASES BY FAULT TYPE")
    print("=" * 80)

    fault_counts = Counter(case.fault_type for case in all_cases)
    for fault, count in sorted(fault_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fault:15s}: {count:3d} cases ({count/len(all_cases)*100:5.1f}%)")

    # Statistics by root cause service (top 15)
    print("\n" + "=" * 80)
    print("TOP 15 ROOT CAUSE SERVICES")
    print("=" * 80)

    service_counts = Counter(case.root_cause_service for case in all_cases)
    for service, count in service_counts.most_common(15):
        print(f"  {service:25s}: {count:3d} cases ({count/len(all_cases)*100:5.1f}%)")

    # Data availability
    print("\n" + "=" * 80)
    print("DATA MODALITY AVAILABILITY")
    print("=" * 80)

    has_metrics = sum(1 for case in all_cases if case.has_metrics())
    has_logs = sum(1 for case in all_cases if case.has_logs())
    has_traces = sum(1 for case in all_cases if case.has_traces())

    print(f"  Metrics: {has_metrics:3d}/{len(all_cases)} ({has_metrics/len(all_cases)*100:5.1f}%)")
    print(f"  Logs:    {has_logs:3d}/{len(all_cases)} ({has_logs/len(all_cases)*100:5.1f}%)")
    print(f"  Traces:  {has_traces:3d}/{len(all_cases)} ({has_traces/len(all_cases)*100:5.1f}%)")

    # Sample a few cases and show detailed info
    print("\n" + "=" * 80)
    print("SAMPLE CASES (5 RANDOM)")
    print("=" * 80)

    import random
    random.seed(42)
    sample_cases = random.sample(all_cases, min(5, len(all_cases)))

    for i, case in enumerate(sample_cases, 1):
        print(f"\n📦 Sample {i}: {case.case_id}")
        print(f"   System: {case.system}")
        print(f"   Fault Type: {case.fault_type}")
        print(f"   Root Cause: {case.root_cause_service}")
        print(f"   Has Metrics: {case.has_metrics()}")
        print(f"   Has Logs: {case.has_logs()}")
        print(f"   Has Traces: {case.has_traces()}")

        if case.has_metrics():
            print(f"   Metrics Path: {case.metrics_path}")

    # Load one case to verify data loading works
    print("\n" + "=" * 80)
    print("TESTING DATA LOADING")
    print("=" * 80)

    test_case = next((case for case in all_cases if case.has_metrics()), None)
    if test_case:
        print(f"\n📦 Testing case: {test_case.case_id}")
        print("   Loading metrics...")

        try:
            test_case.load_data(metrics=True, logs=False, traces=False)

            if test_case.metrics is not None:
                print(f"   ✅ Metrics loaded successfully!")
                print(f"      Shape: {test_case.metrics.shape}")
                print(f"      Columns: {len(test_case.metrics.columns)} metrics")
                print(f"      Timesteps: {len(test_case.metrics)} rows")
                print(f"      Sample columns: {list(test_case.metrics.columns[:5])}")
            else:
                print(f"   ❌ Metrics loaded but is None")

            test_case.unload_data()
            print(f"   ✅ Data unloaded successfully")

        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    else:
        print("\n⚠️  No cases with metrics found for testing")

    # Scenario statistics
    print("\n" + "=" * 80)
    print("SCENARIO STATISTICS")
    print("=" * 80)

    scenarios = defaultdict(list)
    for case in all_cases:
        scenario_key = (case.system, case.root_cause_service, case.fault_type)
        scenarios[scenario_key].append(case)

    print(f"\n  Total unique scenarios: {len(scenarios)}")
    print(f"  Avg cases per scenario: {len(all_cases)/len(scenarios):.1f}")
    print(f"  Min cases per scenario: {min(len(cases) for cases in scenarios.values())}")
    print(f"  Max cases per scenario: {max(len(cases) for cases in scenarios.values())}")

    # Show a few scenarios with their repetitions
    print(f"\n  Sample scenarios (top 5 by repetition):")
    sorted_scenarios = sorted(scenarios.items(), key=lambda x: len(x[1]), reverse=True)
    for (system, service, fault), cases in sorted_scenarios[:5]:
        print(f"    {system}/{service}/{fault}: {len(cases)} cases")

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    print(f"\n✅ Dataset loaded successfully!")
    print(f"✅ Total cases: {len(all_cases)}")
    print(f"✅ Unique scenarios: {len(scenarios)}")
    print(f"✅ Systems: {len(system_counts)}")
    print(f"✅ Fault types: {len(fault_counts)}")
    print(f"✅ Data modalities: Metrics ({has_metrics/len(all_cases)*100:.1f}%), Logs ({has_logs/len(all_cases)*100:.1f}%), Traces ({has_traces/len(all_cases)*100:.1f}%)")
    print(f"✅ Lazy loading: Working")
    print(f"\n🎉 Dataset is ready for analysis!")

    return True


if __name__ == '__main__':
    success = verify_dataset()
    sys.exit(0 if success else 1)
