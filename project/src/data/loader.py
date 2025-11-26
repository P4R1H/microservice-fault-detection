"""
Minimal compatibility wrapper for RCAEval data loading.

This module provides `RCAEvalDataLoader` for backward compatibility with older
scripts/tests that expect a unified loader API. It uses the existing
`rca_data` and `multimodal_data` modules under `src.data` to provide core
functionality.

Note: This implementation focuses on the functionality required by the
current test scripts (lazy discovery, splits, per-case loading). It is not a
full-featured reimplementation of the original loader.
"""

from pathlib import Path
from typing import List, Optional, Tuple

"""
Notes: imports for `rca_data` and `multimodal_data` are intentionally
performed lazily inside loader methods to avoid importing heavy runtime
dependencies (e.g., torch) at module import time. This improves testability
when running lightweight scripts that only need the loader API.
"""


class RCAEvalCaseWrapper:
    """Lightweight wrapper around FailureCase / MultimodalFailureCase.

    Adds a few legacy attributes used by older scripts/tests such as:
    - `re_version`
    - `root_cause_service`
    - `root_cause_indicator` (not available — set to None)

    And convenience methods:
    - `has_metrics()`, `has_logs()`, `has_traces()`
    - `load_data(metrics=True, logs=False, traces=False)`
    - `unload_data()`
    """

    def __init__(self, case):
        self._case = case
        self.case_id = case.case_id
        self.system = case.system
        self.fault_type = case.fault_type
        self.root_cause_service = getattr(case, 'root_cause', None)
        self.root_cause_indicator = None
        self.data_path = Path(case.data_path)
        self.metrics_path = str(self.data_path / 'data.csv') if (self.data_path / 'data.csv').exists() else str(self.data_path / 'metrics.csv') if (self.data_path / 'metrics.csv').exists() else None
        self.logs_path = str(self.data_path / 'logts.csv') if (self.data_path / 'logts.csv').exists() else None
        self.traces_path = str(self.data_path / 'tracets_lat.csv') if (self.data_path / 'tracets_lat.csv').exists() else None
        self.metrics = None
        self.logs = None
        self.traces = None

        # Extract re_version from path, fallback to None
        parts = list(self.data_path.parts)
        # Look for RE1/RE2/RE3 segment
        self.re_version = next((p for p in parts if p.startswith('RE')), None)

    def has_metrics(self) -> bool:
        return self.metrics_path is not None

    def has_logs(self) -> bool:
        return self.logs_path is not None

    def has_traces(self) -> bool:
        return self.traces_path is not None

    def load_data(self, metrics: bool = True, logs: bool = False, traces: bool = False, verbose: bool = False):
        """Load data on-demand. This uses the underlying case's methods.

        For `FailureCase` from `rca_data`, only metrics are available. For
        `MultimodalFailureCase`, logs/traces methods exist.
        """
        if metrics and self.has_metrics():
            try:
                # Some implementations provide load_metrics()
                if hasattr(self._case, 'load_metrics'):
                    self.metrics = self._case.load_metrics()
                # Older wrappers might use load_data(metrics=True) style
                elif hasattr(self._case, 'load_data'):
                    self._case.load_data(metrics=True, logs=False, traces=False)
                    self.metrics = getattr(self._case, 'metrics', None)
                else:
                    self.metrics = None
                if verbose:
                    print(f"Loaded metrics for {self.case_id} from {self.metrics_path}")
            except Exception:
                self.metrics = None

        if logs and self.has_logs():
            try:
                if hasattr(self._case, 'load_logts'):
                    self.logs = self._case.load_logts()
                elif hasattr(self._case, 'load_data'):
                    self._case.load_data(metrics=False, logs=True, traces=False)
                    self.logs = getattr(self._case, 'logs', None)
                if verbose:
                    print(f"Loaded logs for {self.case_id} from {self.logs_path}")
            except Exception:
                self.logs = None

        if traces and self.has_traces():
            try:
                if hasattr(self._case, 'load_tracets_lat'):
                    # load both latency and error traces
                    trlat = self._case.load_tracets_lat()
                    trerr = self._case.load_tracets_err()
                    self.traces = (trlat, trerr)
                elif hasattr(self._case, 'load_data'):
                    self._case.load_data(metrics=False, logs=False, traces=True)
                    self.traces = getattr(self._case, 'traces', None)
                if verbose:
                    print(f"Loaded traces for {self.case_id} from {self.traces_path}")
            except Exception:
                self.traces = None

    def unload_data(self):
        self.metrics = None
        self.logs = None
        self.traces = None


class RCAEvalDataLoader:
    """Compatibility loader that wraps discovery and splitting utilities.

    Provides:
    - load_all_cases(systems=None, verbose=False)
    - load_splits(train_ratio=0.7, val_ratio=0.15, seed=42)
    """

    def __init__(self, data_root: str = 'data/RCAEval'):
        self.data_root = data_root

    def load_all_cases(self, systems: Optional[List[str]] = None, verbose: bool = False) -> List[RCAEvalCaseWrapper]:
        """Discover and return all cases as compatibility wrapper objects."""
        # Lazy import to avoid heavy dependencies during import-time
        from .rca_data import discover_cases

        cases = discover_cases(self.data_root)
        if systems is not None:
            cases = [c for c in cases if c.system in systems]
        wrapped = [RCAEvalCaseWrapper(c) for c in cases]
        if verbose:
            print(f"Discovered {len(wrapped)} cases (filter systems={systems})")
        return wrapped

    def load_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, random_seed: int = 42) -> Tuple[List[RCAEvalCaseWrapper], List[RCAEvalCaseWrapper], List[RCAEvalCaseWrapper]]:
        """Split cases into train/val/test and return wrapped objects."""
        # Lazy import to avoid heavy dependencies during import-time
        from .rca_data import discover_cases, split_cases

        cases = discover_cases(self.data_root)
        train, val, test = split_cases(cases, seed=random_seed, train_ratio=train_ratio, val_ratio=val_ratio)  # type: ignore
        return ([RCAEvalCaseWrapper(c) for c in train], [RCAEvalCaseWrapper(c) for c in val], [RCAEvalCaseWrapper(c) for c in test])

    # Convenience methods (old API compatibility)
    def discover_cases(self):
        return self.load_all_cases()


# Simple test when run as script
if __name__ == '__main__':
    loader = RCAEvalDataLoader('data/RCAEval')
    cases = loader.load_all_cases(verbose=True)
    print(f"First case example: {cases[0].case_id if cases else 'None'}")
