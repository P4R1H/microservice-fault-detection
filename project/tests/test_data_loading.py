"""
Test Script: RCAEval Data Loading (LAZY VERSION)

Verifies dataset extraction and lazy data loading work correctly.

Usage:
    cd project
    python tests/test_data_loading.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import RCAEvalDataLoader


def test_data_loading():
    """Test RCAEval data loader with lazy loading"""

    print("=" * 80)
    print("TEST 1: Data Loader Initialization")
    print("=" * 80)

    try:
        loader = RCAEvalDataLoader('data/RCAEval')
        print("✅ Data loader initialized successfully\n")
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("Please ensure dataset is extracted to data/RCAEval/")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    print("=" * 80)
    print("TEST 2: Discover All Cases (Lazy - No CSV Loading)")
    print("=" * 80)

    try:
        # This should be FAST - only scans directories
        cases = loader.load_all_cases(verbose=True)

        if len(cases) > 0:
            print(f"\n✅ Discovered {len(cases)} failure cases (data NOT loaded yet)")
            print(f"   Systems found: {set(c.system for c in cases)}")
        else:
            print("⚠️  No cases discovered - check dataset structure")
            return False

    except Exception as e:
        print(f"❌ Error discovering cases: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("TEST 3: Load TrainTicket System Only")
    print("=" * 80)

    try:
        tt_cases = loader.load_all_cases(systems=['TrainTicket'], verbose=True)
        print(f"\n✅ Discovered {len(tt_cases)} TrainTicket cases")

        if len(tt_cases) == 0:
            print("⚠️  No TrainTicket cases found")
            return False

    except Exception as e:
        print(f"❌ Error loading TrainTicket: {e}")
        return False

    print("\n" + "=" * 80)
    print("TEST 4: Inspect Sample Case Metadata (No Data Loading)")
    print("=" * 80)

    sample = cases[0]
    print(f"\n📦 Sample Case: {sample.case_id}")
    print(f"   System: {sample.system}")
    print(f"   RE Version: {sample.re_version}")
    print(f"   Fault Type: {sample.fault_type}")
    print(f"   Root Cause Service: {sample.root_cause_service}")
    print(f"   Root Cause Indicator: {sample.root_cause_indicator}")

    # Check file availability (NOT loaded yet)
    print(f"\n📂 File Availability:")
    print(f"   Has metrics: {sample.has_metrics()}")
    print(f"   Has logs: {sample.has_logs()}")
    print(f"   Has traces: {sample.has_traces()}")

    # Show file paths
    if sample.metrics_path:
        print(f"\n📄 File Paths:")
        print(f"   Metrics: {sample.metrics_path}")
        if sample.logs_path:
            print(f"   Logs: {sample.logs_path}")
        if sample.traces_path:
            print(f"   Traces: {sample.traces_path}")

    print("\n" + "=" * 80)
    print("TEST 5: Load Data On-Demand (Metrics Only)")
    print("=" * 80)

    try:
        # Load only metrics for this case
        print(f"Loading metrics for {sample.case_id}...")
        sample.load_data(metrics=True, logs=False, traces=False, verbose=True)

        if sample.metrics is not None:
            print(f"\n✅ Metrics loaded: {sample.metrics.shape} (timesteps × features)")
            print(f"   Columns (first 5): {list(sample.metrics.columns)[:5]}")
            print(f"   Memory usage: ~{sample.metrics.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        else:
            print("❌ Metrics failed to load")

        # Logs and traces should still be None
        assert sample.logs is None, "Logs should not be loaded"
        assert sample.traces is None, "Traces should not be loaded"
        print("✅ Selective loading works (logs and traces not loaded)")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("TEST 6: Unload Data to Free Memory")
    print("=" * 80)

    sample.unload_data()
    assert sample.metrics is None, "Metrics should be unloaded"
    print("✅ Data unloaded successfully")

    print("\n" + "=" * 80)
    print("TEST 7: Load Train/Val/Test Splits (Grouped - No Leakage)")
    print("=" * 80)

    try:
        train, val, test = loader.load_splits(train_ratio=0.6, val_ratio=0.2, random_seed=42)

        print(f"\n✅ Dataset splits created (data NOT loaded yet):")
        print(f"   Train: {len(train)} cases ({len(train)/len(cases)*100:.1f}%)")
        print(f"   Val:   {len(val)} cases ({len(val)/len(cases)*100:.1f}%)")
        print(f"   Test:  {len(test)} cases ({len(test)/len(cases)*100:.1f}%)")

        # Verify no scenario overlap (grouped splitting)
        train_scenarios = set((c.system, c.root_cause_service, c.fault_type) for c in train)
        val_scenarios = set((c.system, c.root_cause_service, c.fault_type) for c in val)
        test_scenarios = set((c.system, c.root_cause_service, c.fault_type) for c in test)

        train_val_overlap = train_scenarios & val_scenarios
        train_test_overlap = train_scenarios & test_scenarios
        val_test_overlap = val_scenarios & test_scenarios

        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("❌ SCENARIO LEAKAGE detected:")
            if train_val_overlap:
                print(f"   Train-Val overlap: {len(train_val_overlap)} scenarios")
                print(f"   Example: {list(train_val_overlap)[:3]}")
            if train_test_overlap:
                print(f"   Train-Test overlap: {len(train_test_overlap)} scenarios")
                print(f"   Example: {list(train_test_overlap)[:3]}")
            if val_test_overlap:
                print(f"   Val-Test overlap: {len(val_test_overlap)} scenarios")
                print(f"   Example: {list(val_test_overlap)[:3]}")
            return False
        else:
            print("✅ No scenario leakage: all scenarios are disjoint across splits")
            print(f"   Train scenarios: {len(train_scenarios)}")
            print(f"   Val scenarios: {len(val_scenarios)}")
            print(f"   Test scenarios: {len(test_scenarios)}")

    except Exception as e:
        print(f"❌ Error creating splits: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("TEST 8: Fault Type Distribution")
    print("=" * 80)

    fault_dist = loader.get_fault_type_distribution(cases)
    print(f"\n🔥 Fault Types:")
    for fault, count in sorted(fault_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"   {fault}: {count} cases ({count/len(cases)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("TEST 9: System Distribution")
    print("=" * 80)

    system_dist = loader.get_system_distribution(cases)
    print(f"\n📦 Systems:")
    for system, count in sorted(system_dist.items()):
        print(f"   {system}: {count} cases ({count/len(cases)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("TEST 10: Top Services")
    print("=" * 80)

    service_dist = loader.get_service_distribution(cases)
    print(f"\n🎯 Top 10 Root Cause Services:")
    for service, count in sorted(service_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {service}: {count} cases")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nDataset Statistics Summary:")
    print(f"  Total cases: {len(cases)}")
    print(f"  Systems: {len(system_dist)}")
    print(f"  Fault types: {len(fault_dist)}")
    print(f"  Unique services: {len(service_dist)}")

    # Check data availability (paths exist, data not loaded)
    metrics_count = sum(1 for c in cases if c.has_metrics())
    logs_count = sum(1 for c in cases if c.has_logs())
    traces_count = sum(1 for c in cases if c.has_traces())
    print(f"\n  Data file availability:")
    print(f"    Metrics: {metrics_count}/{len(cases)} ({metrics_count/len(cases)*100:.1f}%)")
    print(f"    Logs: {logs_count}/{len(cases)} ({logs_count/len(cases)*100:.1f}%)")
    print(f"    Traces: {traces_count}/{len(cases)} ({traces_count/len(cases)*100:.1f}%)")

    print("\n🎉 Dataset is ready for experiments!")
    print("\n💡 Key Benefits of Lazy Loading:")
    print("   - FAST discovery (~seconds instead of 40+ minutes)")
    print("   - LOW memory usage (only paths stored, not data)")
    print("   - Load on-demand (case.load_data())")
    print("   - Selective loading (metrics only, logs only, etc.)")
    print("   - Memory control (case.unload_data())")

    return True


if __name__ == '__main__':
    success = test_data_loading()
    sys.exit(0 if success else 1)
