"""
RCAEval Data Loader - Lazy Loading Version

Handles the actual RCAEval dataset structure with LAZY LOADING:
- Scans directories and parses labels WITHOUT loading CSVs
- Stores file paths instead of DataFrames
- Loads data on-demand via .load_data() method
- Supports selective loading (metrics only, logs only, etc.)

Example paths:
  data/RCAEval/TrainTicket/RE2/RE2-TT/ts-auth-service_cpu/1/metrics.csv
  data/RCAEval/OnlineBoutique/RE1/RE1-OB/adservice_cpu/1/data.csv
  data/RCAEval/SockShop/RE3/RE3-SS/carts_f1/1/metrics.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class FailureCase:
    """
    Single failure case from RCAEval dataset (LAZY LOADING)

    Data is NOT loaded on initialization - only file paths are stored.
    Call .load_data() to actually read CSVs into memory.
    """
    case_id: str
    system: str  # 'TrainTicket', 'SockShop', 'OnlineBoutique'
    fault_type: str  # 'CPU', 'MEM', 'DISK', 'SOCKET', 'DELAY', 'LOSS'
    root_cause_service: str
    root_cause_indicator: str

    # File paths (always available)
    metrics_path: Optional[Path] = None
    logs_path: Optional[Path] = None
    traces_path: Optional[Path] = None

    # Data modalities (loaded on-demand, initially None)
    metrics: Optional[pd.DataFrame] = field(default=None, repr=False)
    logs: Optional[pd.DataFrame] = field(default=None, repr=False)
    traces: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Metadata
    re_version: Optional[str] = None  # RE1, RE2, or RE3
    case_number: Optional[int] = None
    timestamp: Optional[pd.Timestamp] = None
    duration_minutes: Optional[int] = None

    def load_data(
        self,
        metrics: bool = True,
        logs: bool = True,
        traces: bool = True,
        verbose: bool = False,
        max_rows_logs: Optional[int] = None,
        max_rows_traces: Optional[int] = None
    ) -> 'FailureCase':
        """
        Load data from CSV files into memory (LAZY LOADING)

        Args:
            metrics: Load metrics.csv
            logs: Load logs.csv
            traces: Load traces.csv
            verbose: Print loading progress
            max_rows_logs: Maximum rows to load from logs (None = all rows)
            max_rows_traces: Maximum rows to load from traces (None = all rows)

        Returns:
            Self (for chaining)
        """
        if verbose:
            print(f"Loading {self.case_id}...")

        # Load metrics
        if metrics and self.metrics_path is not None and self.metrics is None:
            try:
                self.metrics = pd.read_csv(self.metrics_path)
                if verbose:
                    print(f"  ✅ Metrics: {self.metrics.shape}")
            except Exception as e:
                if verbose:
                    print(f"  ❌ Metrics error: {e}")

        # Load logs (with optional row limit for performance)
        if logs and self.logs_path is not None and self.logs is None:
            try:
                self.logs = pd.read_csv(self.logs_path, nrows=max_rows_logs)
                if verbose:
                    print(f"  ✅ Logs: {len(self.logs)} entries")
            except Exception as e:
                if verbose:
                    print(f"  ❌ Logs error: {e}")

        # Load traces (with optional row limit for performance)
        if traces and self.traces_path is not None and self.traces is None:
            try:
                self.traces = pd.read_csv(self.traces_path, nrows=max_rows_traces)
                if verbose:
                    print(f"  ✅ Traces: {len(self.traces)} spans")
            except Exception as e:
                if verbose:
                    print(f"  ❌ Traces error: {e}")

        return self

    def unload_data(self):
        """Free memory by unloading DataFrames"""
        self.metrics = None
        self.logs = None
        self.traces = None

    def has_metrics(self) -> bool:
        """Check if metrics file exists"""
        return self.metrics_path is not None and self.metrics_path.exists()

    def has_logs(self) -> bool:
        """Check if logs file exists"""
        return self.logs_path is not None and self.logs_path.exists()

    def has_traces(self) -> bool:
        """Check if traces file exists"""
        return self.traces_path is not None and self.traces_path.exists()


class RCAEvalDataLoader:
    """
    Robust LAZY data loader for RCAEval benchmark dataset

    Key features:
    - LAZY LOADING: Only scans directories, doesn't load CSVs until requested
    - Recursive discovery of failure cases
    - Dynamic label parsing from folder names
    - Multiple metric filename formats
    - Fault code mapping

    Usage:
        loader = RCAEvalDataLoader(data_dir='data/RCAEval')
        cases = loader.load_all_cases()  # Fast - only scans directories

        # Load data for specific case
        case = cases[0]
        case.load_data(metrics=True, traces=False)  # Only load what you need

        # Process and unload
        process(case.metrics)
        case.unload_data()  # Free memory
    """

    # Fault code mappings
    FAULT_CODE_MAP = {
        'cpu': 'CPU',
        'mem': 'MEM',
        'memory': 'MEM',
        'disk': 'DISK',
        'delay': 'DELAY',
        'loss': 'LOSS',
        'packet_loss': 'LOSS',
        'socket': 'SOCKET',
        # SockShop/RE3 specific codes
        'f1': 'CPU',
        'f2': 'MEM',
        'f3': 'DISK',
        'f4': 'DELAY',
        'f5': 'LOSS',
        'f6': 'SOCKET',
    }

    # Metric filename priority (try in order)
    METRIC_FILENAMES = ['metrics.csv', 'data.csv', 'simple_metrics.csv']

    def __init__(self, data_dir: str = 'data/RCAEval'):
        """
        Initialize data loader

        Args:
            data_dir: Path to RCAEval dataset directory
        """
        self.data_dir = Path(data_dir)
        self.systems = ['TrainTicket', 'SockShop', 'OnlineBoutique']

        # Verify dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"RCAEval dataset not found at {self.data_dir}\n"
                f"Please run: python scripts/download_dataset.py --all"
            )

    def load_all_cases(
        self,
        systems: List[str] = None,
        re_versions: List[str] = None,
        verbose: bool = True
    ) -> List[FailureCase]:
        """
        Discover all failure cases (LAZY - doesn't load CSVs)

        Args:
            systems: List of system names to load (default: all)
            re_versions: List of RE versions to load (default: all)
            verbose: Print progress

        Returns:
            List of FailureCase objects (data NOT loaded yet)
        """
        if systems is None:
            systems = self.systems

        if re_versions is None:
            re_versions = ['RE1', 'RE2', 'RE3']

        cases = []

        for system in systems:
            system_dir = self.data_dir / system
            if not system_dir.exists():
                if verbose:
                    print(f"⚠️  System directory not found: {system_dir}")
                continue

            # Scan for RE versions
            for re_version in re_versions:
                re_version_dir = system_dir / re_version
                if not re_version_dir.exists():
                    continue

                # Recursively find all failure case directories (LAZY)
                case_dirs = self._discover_case_directories(re_version_dir)

                if verbose:
                    print(f"📂 {system}/{re_version}: Found {len(case_dirs)} cases")

                for case_dir in case_dirs:
                    case = self._create_case_metadata(system, re_version, case_dir)
                    if case is not None:
                        cases.append(case)

        if len(cases) == 0:
            print("❌ No cases found! Please check dataset structure.")
            print(f"   Expected structure: {self.data_dir}/{{System}}/{{RE_Version}}/")
        else:
            if verbose:
                print(f"\n✅ Discovered {len(cases)} failure cases (data not loaded yet)")

        return cases

    def _discover_case_directories(self, re_version_dir: Path) -> List[Path]:
        """
        Recursively discover directories that contain failure case data

        A directory is a failure case if it contains any of:
        - metrics.csv, data.csv, or simple_metrics.csv

        Args:
            re_version_dir: Path to RE version directory (e.g., TrainTicket/RE2/)

        Returns:
            List of paths to failure case directories
        """
        case_dirs = []

        # Recursively walk through directory tree
        for item in re_version_dir.rglob('*'):
            if item.is_dir():
                # Check if this directory contains a metrics file
                has_metrics = any((item / filename).exists() for filename in self.METRIC_FILENAMES)

                if has_metrics:
                    case_dirs.append(item)

        return case_dirs

    def _parse_folder_labels(self, case_dir: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse service and fault type from folder name

        Handles multiple formats:
          - {service}_{fault}/1/           → service='ts-auth-service', fault='CPU'
          - {service}_{fault_code}_{num}/  → service='ts-route', fault='DISK' (f3)

        Examples:
          - ts-auth-service_cpu → service='ts-auth-service', fault='CPU'
          - adservice_cpu → service='adservice', fault='CPU'
          - carts_f1 → service='carts', fault='CPU' (via mapping)
          - ts-route-service_f3_1 → service='ts-route-service', fault='DISK'

        Args:
            case_dir: Path to case directory

        Returns:
            (service, fault_type) or (None, None) if parsing fails
        """
        # Get the folder name (skip numeric suffix if present)
        folder_name = case_dir.name
        if folder_name.isdigit():
            folder_name = case_dir.parent.name

        # Split by underscore
        parts = folder_name.split('_')

        if len(parts) < 2:
            return None, None

        # Handle format: {service}_{fault_code}_{num}
        # Example: ts-route-service_f3_1
        if len(parts) >= 3 and parts[-1].isdigit():
            # Last part is number, second-to-last is fault code
            service = '_'.join(parts[:-2])
            fault_code = parts[-2]
        else:
            # Standard format: {service}_{fault}
            service = '_'.join(parts[:-1])
            fault_code = parts[-1]

        # Map fault code to standard fault type
        fault_type = self.FAULT_CODE_MAP.get(fault_code.lower())

        if fault_type is None:
            # Try exact match
            if fault_code.upper() in ['CPU', 'MEM', 'DISK', 'DELAY', 'LOSS', 'SOCKET']:
                fault_type = fault_code.upper()
            else:
                # Unknown fault code - skip this case
                return service, None

        return service, fault_type

    def _find_metric_file(self, case_dir: Path) -> Optional[Path]:
        """
        Find metrics file in directory (handles different naming conventions)

        Tries in priority order: metrics.csv, data.csv, simple_metrics.csv

        Args:
            case_dir: Path to case directory

        Returns:
            Path to metrics file or None
        """
        for filename in self.METRIC_FILENAMES:
            file_path = case_dir / filename
            if file_path.exists():
                return file_path
        return None

    def _create_case_metadata(
        self,
        system: str,
        re_version: str,
        case_dir: Path
    ) -> Optional[FailureCase]:
        """
        Create FailureCase object with metadata ONLY (no data loading)

        Args:
            system: System name (TrainTicket, SockShop, OnlineBoutique)
            re_version: RE version (RE1, RE2, RE3)
            case_dir: Path to case directory

        Returns:
            FailureCase object or None if parsing fails
        """
        try:
            # Parse labels from folder name
            service, fault_type = self._parse_folder_labels(case_dir)

            if service is None or fault_type is None:
                return None

            # Generate case ID
            case_id = f"{system}_{re_version}_{service}_{fault_type}_{case_dir.name}"

            # Find file paths (NO LOADING)
            metrics_path = self._find_metric_file(case_dir)
            logs_path = case_dir / 'logs.csv' if (case_dir / 'logs.csv').exists() else None
            traces_path = case_dir / 'traces.csv' if (case_dir / 'traces.csv').exists() else None

            if metrics_path is None:
                return None

            # Root cause indicator (heuristic: fault type mapped to metric)
            indicator_map = {
                'CPU': 'cpu_usage_percent',
                'MEM': 'memory_usage_mb',
                'DISK': 'disk_usage_percent',
                'DELAY': 'latency_ms',
                'LOSS': 'packet_loss_rate',
                'SOCKET': 'socket_count'
            }
            root_cause_indicator = indicator_map.get(fault_type, f"{fault_type.lower()}_metric")

            # Extract case number from directory name
            case_number = None
            if case_dir.name.isdigit():
                case_number = int(case_dir.name)

            # Create failure case object (NO DATA LOADED)
            case = FailureCase(
                case_id=case_id,
                system=system,
                fault_type=fault_type,
                root_cause_service=service,
                root_cause_indicator=root_cause_indicator,
                metrics_path=metrics_path,
                logs_path=logs_path,
                traces_path=traces_path,
                re_version=re_version,
                case_number=case_number,
                timestamp=None,
                duration_minutes=60
            )

            return case

        except Exception as e:
            # Silently skip malformed cases
            return None

    def load_splits(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42,
        stratify_by: str = 'fault_type'
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """
        Load dataset with GROUPED splits to prevent data leakage

        CRITICAL: Uses scenario-based splitting to prevent leakage.
        A scenario = (System, RootCauseService, FaultType).
        All repetitions (runs) of the same scenario stay together in one split.

        Args:
            train_ratio: Proportion for training (default: 0.6)
            val_ratio: Proportion for validation (default: 0.2)
            random_seed: Random seed for reproducibility
            stratify_by: Stratify by 'fault_type', 'system', or None

        Returns:
            (train_cases, val_cases, test_cases)
        """
        cases = self.load_all_cases(verbose=True)

        if len(cases) == 0:
            print("❌ No cases loaded - cannot create splits")
            return [], [], []

        # Use grouped splitting to prevent data leakage
        return self._grouped_split(cases, train_ratio, val_ratio, random_seed, stratify_by)

    def _group_by_scenario(self, cases: List[FailureCase]) -> Dict[Tuple[str, str, str], List[FailureCase]]:
        """
        Group cases by scenario to prevent data leakage

        A scenario is defined by: (System, RootCauseService, FaultType)
        All runs/repetitions of the same scenario are grouped together.

        Example:
          - ts-auth-service_cpu run 1, 2, 3 → all in same group
          - Prevents run 1 in train, run 2 in test (data leakage)

        Args:
            cases: List of all failure cases

        Returns:
            Dict mapping scenario tuple to list of cases
        """
        scenarios = {}

        for case in cases:
            # Define scenario key
            scenario_key = (case.system, case.root_cause_service, case.fault_type)

            if scenario_key not in scenarios:
                scenarios[scenario_key] = []

            scenarios[scenario_key].append(case)

        return scenarios

    def _grouped_split(
        self,
        cases: List[FailureCase],
        train_ratio: float,
        val_ratio: float,
        random_seed: int,
        stratify_by: Optional[str] = None
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """
        Split dataset by scenarios (not individual cases) to prevent data leakage

        Process:
        1. Group cases by scenario (System, Service, Fault)
        2. Split scenarios into train/val/test
        3. Expand scenarios back to individual cases

        Args:
            cases: List of all failure cases
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            random_seed: Random seed
            stratify_by: Optional stratification ('fault_type', 'system', or None)

        Returns:
            (train_cases, val_cases, test_cases)
        """
        np.random.seed(random_seed)

        # Group by scenario
        scenarios = self._group_by_scenario(cases)
        scenario_keys = list(scenarios.keys())

        print(f"\n📊 Scenario-based splitting:")
        print(f"   Total cases: {len(cases)}")
        print(f"   Unique scenarios: {len(scenarios)}")
        print(f"   Avg cases per scenario: {len(cases) / len(scenarios):.1f}")

        if stratify_by == 'fault_type':
            # Group scenarios by fault type
            fault_scenarios = {}
            for key in scenario_keys:
                fault = key[2]  # (system, service, fault) -> fault
                if fault not in fault_scenarios:
                    fault_scenarios[fault] = []
                fault_scenarios[fault].append(key)

            train_scenarios, val_scenarios, test_scenarios = [], [], []

            # Split each fault type's scenarios
            for fault, fault_keys in fault_scenarios.items():
                n_scenarios = len(fault_keys)
                # Shuffle while preserving tuple type
                indices = np.random.permutation(n_scenarios)
                shuffled = [fault_keys[i] for i in indices]

                n_train = int(n_scenarios * train_ratio)
                n_val = int(n_scenarios * val_ratio)

                train_scenarios.extend(shuffled[:n_train])
                val_scenarios.extend(shuffled[n_train:n_train+n_val])
                test_scenarios.extend(shuffled[n_train+n_val:])

        elif stratify_by == 'system':
            # Group scenarios by system
            system_scenarios = {}
            for key in scenario_keys:
                system = key[0]  # (system, service, fault) -> system
                if system not in system_scenarios:
                    system_scenarios[system] = []
                system_scenarios[system].append(key)

            train_scenarios, val_scenarios, test_scenarios = [], [], []

            # Split each system's scenarios
            for system, sys_keys in system_scenarios.items():
                n_scenarios = len(sys_keys)
                # Shuffle while preserving tuple type
                indices = np.random.permutation(n_scenarios)
                shuffled = [sys_keys[i] for i in indices]

                n_train = int(n_scenarios * train_ratio)
                n_val = int(n_scenarios * val_ratio)

                train_scenarios.extend(shuffled[:n_train])
                val_scenarios.extend(shuffled[n_train:n_train+n_val])
                test_scenarios.extend(shuffled[n_train+n_val:])

        else:
            # Random split of scenarios
            # Shuffle while preserving tuple type
            indices = np.random.permutation(len(scenario_keys))
            shuffled_keys = [scenario_keys[i] for i in indices]

            n_train = int(len(scenario_keys) * train_ratio)
            n_val = int(len(scenario_keys) * val_ratio)

            train_scenarios = shuffled_keys[:n_train]
            val_scenarios = shuffled_keys[n_train:n_train+n_val]
            test_scenarios = shuffled_keys[n_train+n_val:]

        # Expand scenarios back to individual cases
        train_cases = []
        for scenario_key in train_scenarios:
            train_cases.extend(scenarios[scenario_key])

        val_cases = []
        for scenario_key in val_scenarios:
            val_cases.extend(scenarios[scenario_key])

        test_cases = []
        for scenario_key in test_scenarios:
            test_cases.extend(scenarios[scenario_key])

        # Shuffle cases within each split
        np.random.shuffle(train_cases)
        np.random.shuffle(val_cases)
        np.random.shuffle(test_cases)

        # Verify no leakage
        self._verify_no_leakage(train_cases, val_cases, test_cases)

        self._print_split_stats(train_cases, val_cases, test_cases)
        return train_cases, val_cases, test_cases

    def _verify_no_leakage(
        self,
        train_cases: List[FailureCase],
        val_cases: List[FailureCase],
        test_cases: List[FailureCase]
    ):
        """
        Verify no data leakage between splits

        Checks that no scenario appears in multiple splits.
        """
        train_scenarios = set((c.system, c.root_cause_service, c.fault_type) for c in train_cases)
        val_scenarios = set((c.system, c.root_cause_service, c.fault_type) for c in val_cases)
        test_scenarios = set((c.system, c.root_cause_service, c.fault_type) for c in test_cases)

        # Check for overlaps
        train_val_overlap = train_scenarios & val_scenarios
        train_test_overlap = train_scenarios & test_scenarios
        val_test_overlap = val_scenarios & test_scenarios

        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("⚠️  WARNING: Scenario leakage detected!")
            if train_val_overlap:
                print(f"   Train-Val overlap: {len(train_val_overlap)} scenarios")
            if train_test_overlap:
                print(f"   Train-Test overlap: {len(train_test_overlap)} scenarios")
            if val_test_overlap:
                print(f"   Val-Test overlap: {len(val_test_overlap)} scenarios")
        else:
            print("✅ No scenario leakage: all scenarios are disjoint across splits")

    def _print_split_stats(
        self,
        train_cases: List[FailureCase],
        val_cases: List[FailureCase],
        test_cases: List[FailureCase]
    ):
        """Print statistics about dataset splits"""
        total = len(train_cases) + len(val_cases) + len(test_cases)

        print(f"\n📊 Dataset splits:")
        print(f"   Train: {len(train_cases)} cases ({len(train_cases)/total*100:.1f}%)")
        print(f"   Val:   {len(val_cases)} cases ({len(val_cases)/total*100:.1f}%)")
        print(f"   Test:  {len(test_cases)} cases ({len(test_cases)/total*100:.1f}%)")

    def get_fault_type_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of fault types in dataset"""
        return dict(Counter(case.fault_type for case in cases))

    def get_system_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of systems in dataset"""
        return dict(Counter(case.system for case in cases))

    def get_service_distribution(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Get distribution of root cause services"""
        return dict(Counter(case.root_cause_service for case in cases))


def load_rcaeval_dataset(
    data_dir: str = 'data/RCAEval',
    split: str = 'all'
) -> List[FailureCase]:
    """
    Convenience function to load RCAEval dataset

    Args:
        data_dir: Path to dataset directory
        split: 'all', 'train', 'val', or 'test'

    Returns:
        List of FailureCase objects (data NOT loaded)
    """
    loader = RCAEvalDataLoader(data_dir)

    if split == 'all':
        return loader.load_all_cases()
    else:
        train, val, test = loader.load_splits()
        if split == 'train':
            return train
        elif split == 'val':
            return val
        elif split == 'test':
            return test
        else:
            raise ValueError(f"Unknown split: {split}")


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("RCAEval Data Loader - LAZY LOADING Test")
    print("=" * 80)

    # Load dataset (FAST - only scans directories)
    loader = RCAEvalDataLoader('data/RCAEval')

    print("\n1. Discovering all cases (lazy - no CSV loading)...")
    cases = loader.load_all_cases()

    if cases:
        # Get splits
        print("\n2. Creating stratified splits...")
        train, val, test = loader.load_splits(stratify_by='fault_type')

        # Show statistics (NO DATA LOADED YET)
        print("\n3. Fault Type Distribution (All Cases):")
        for fault_type, count in sorted(loader.get_fault_type_distribution(cases).items()):
            print(f"   {fault_type}: {count} cases")

        print("\n4. System Distribution (All Cases):")
        for system, count in sorted(loader.get_system_distribution(cases).items()):
            print(f"   {system}: {count} cases")

        print("\n5. Top 10 Root Cause Services:")
        service_dist = loader.get_service_distribution(cases)
        for service, count in sorted(service_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {service}: {count} cases")

        # Show sample case WITHOUT loading data
        print("\n6. Sample Case Metadata (NO DATA LOADED):")
        sample = cases[0]
        print(f"   Case ID: {sample.case_id}")
        print(f"   System: {sample.system}")
        print(f"   RE Version: {sample.re_version}")
        print(f"   Fault Type: {sample.fault_type}")
        print(f"   Root Cause Service: {sample.root_cause_service}")
        print(f"   Has metrics: {sample.has_metrics()}")
        print(f"   Has logs: {sample.has_logs()}")
        print(f"   Has traces: {sample.has_traces()}")

        # NOW load data for this specific case
        print("\n7. Loading data for sample case...")
        sample.load_data(metrics=True, logs=True, traces=True, verbose=True)

        if sample.metrics is not None:
            print(f"\n   Metrics: {sample.metrics.shape} (timesteps × features)")
            print(f"      Columns (first 5): {list(sample.metrics.columns)[:5]}")

        if sample.logs is not None:
            print(f"   Logs: {len(sample.logs)} entries")

        if sample.traces is not None:
            print(f"   Traces: {len(sample.traces)} spans")

        # Unload to free memory
        print("\n8. Unloading data to free memory...")
        sample.unload_data()
        print("   ✅ Data unloaded")

        print("\n" + "=" * 80)
        print("✅ Lazy loading test complete!")
        print("=" * 80)
        print("\nKey points:")
        print("  - Discovery is FAST (no CSV loading)")
        print("  - Data loaded on-demand via .load_data()")
        print("  - Can unload data to free memory")
        print("  - Can load specific modalities (metrics only, etc.)")
    else:
        print("\n❌ No cases discovered - check dataset structure")
