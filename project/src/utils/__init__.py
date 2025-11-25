"""
Utility modules for the AIOps RCA project

Note: RCAEvalDataLoader and FailureCase are now in src.data.loader
This file provides backward compatibility imports.
"""

# Backward compatibility - import from src.data.loader
from src.data.loader import RCAEvalDataLoader, FailureCase

from .visualization import (
    MetricsVisualizer,
    LogsVisualizer,
    TracesVisualizer,
    ResultsVisualizer
)

# Convenience function for backward compatibility
def load_rcaeval_dataset(data_dir: str = 'data/RCAEval', **kwargs):
    """Load RCAEval dataset (backward compatibility wrapper)."""
    loader = RCAEvalDataLoader(data_dir)
    return loader.load_all_cases(**kwargs)

__all__ = [
    'RCAEvalDataLoader',
    'FailureCase',
    'load_rcaeval_dataset',
    'MetricsVisualizer',
    'LogsVisualizer',
    'TracesVisualizer',
    'ResultsVisualizer'
]
