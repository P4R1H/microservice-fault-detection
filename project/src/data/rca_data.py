"""
Data structures and loading utilities for RCA.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class FailureCase:
    """Single failure case with all data."""
    case_id: str
    system: str  # TrainTicket, SockShop, OnlineBoutique
    root_cause: str  # Service that caused the fault
    fault_type: str  # cpu, mem, delay, loss, disk
    data_path: str
    
    # Loaded data (lazy) - use field with default_factory for mutable defaults
    _metrics: Optional[pd.DataFrame] = field(default=None, repr=False)
    _service_metrics: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    
    def load_metrics(self) -> pd.DataFrame:
        """Load raw metrics from CSV."""
        if self._metrics is None:
            self._metrics = pd.read_csv(os.path.join(self.data_path, 'data.csv'))
        return self._metrics
    
    def get_service_metrics(self, services: List[str]) -> Dict[str, np.ndarray]:
        """Aggregate metrics by service."""
        if self._service_metrics is not None:
            return self._service_metrics
            
        df = self.load_metrics()
        self._service_metrics = {}
        
        for service in services:
            # Find columns for this service
            service_cols = [c for c in df.columns if c.startswith(service + '_')]
            if service_cols:
                data = df[service_cols].values.astype(np.float32)
                # Handle NaN/Inf
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                self._service_metrics[service] = data
            else:
                # Service not in this case - create zeros
                self._service_metrics[service] = np.zeros((len(df), 10), dtype=np.float32)
        
        return self._service_metrics
    
    def clear_cache(self):
        """Clear cached data to free memory."""
        self._metrics = None
        self._service_metrics = None


def discover_cases(data_root: Union[str, Path] = 'data/RCAEval') -> List[FailureCase]:
    """Discover all failure cases in the dataset."""
    cases = []
    data_root = Path(data_root) if isinstance(data_root, str) else data_root
    
    for system in ['TrainTicket', 'SockShop', 'OnlineBoutique']:
        sys_path = data_root / system
        if not sys_path.exists():
            continue
            
        for re_dir in sys_path.iterdir():
            if not re_dir.is_dir():
                continue
            for sub_dir in re_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                for fault_dir in sub_dir.iterdir():
                    if not fault_dir.is_dir():
                        continue
                    
                    # Parse service_faulttype from folder name
                    parts = fault_dir.name.rsplit('_', 1)
                    if len(parts) != 2:
                        continue
                    service, fault_type = parts
                    
                    for case_dir in fault_dir.iterdir():
                        if case_dir.is_dir() and (case_dir / 'data.csv').exists():
                            case_id = f"{system}_{re_dir.name}_{fault_dir.name}_{case_dir.name}"
                            cases.append(FailureCase(
                                case_id=case_id,
                                system=system,
                                root_cause=service,
                                fault_type=fault_type,
                                data_path=str(case_dir)
                            ))
    
    return cases


def get_all_services(cases: List[FailureCase]) -> List[str]:
    """Extract all unique service names."""
    services = set()
    for case in cases:
        services.add(case.root_cause)
    return sorted(list(services))


def split_cases(cases: List[FailureCase], 
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                seed: int = 42) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
    """Split cases by scenario to prevent data leakage."""
    np.random.seed(seed)
    
    # Group by (system, service, fault_type) to prevent leakage
    scenarios = defaultdict(list)
    for case in cases:
        key = (case.system, case.root_cause, case.fault_type)
        scenarios[key].append(case)
    
    # Shuffle scenarios
    scenario_keys = list(scenarios.keys())
    np.random.shuffle(scenario_keys)
    
    # Split scenarios
    n = len(scenario_keys)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_cases = []
    val_cases = []
    test_cases = []
    
    for i, key in enumerate(scenario_keys):
        if i < train_end:
            train_cases.extend(scenarios[key])
        elif i < val_end:
            val_cases.extend(scenarios[key])
        else:
            test_cases.extend(scenarios[key])
    
    # Shuffle within splits
    np.random.shuffle(train_cases)
    np.random.shuffle(val_cases)
    np.random.shuffle(test_cases)
    
    return train_cases, val_cases, test_cases


class RCADataset(Dataset):
    """PyTorch dataset for RCA training."""
    
    def __init__(self, 
                 cases: List[FailureCase],
                 services: List[str],
                 seq_len: int = 128,
                 n_features: int = 64):
        self.cases = cases
        self.services = services
        self.service_to_idx = {s: i for i, s in enumerate(services)}
        self.seq_len = seq_len
        self.n_features = n_features
        
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx]
        service_metrics = case.get_service_metrics(self.services)
        
        # Build tensor: (n_services, seq_len, n_features)
        n_services = len(self.services)
        X = np.zeros((n_services, self.seq_len, self.n_features), dtype=np.float32)
        
        for i, service in enumerate(self.services):
            data = service_metrics.get(service, np.zeros((self.seq_len, self.n_features)))
            
            # Take last seq_len timesteps
            if len(data) > self.seq_len:
                data = data[-self.seq_len:]
            
            # Truncate features if needed
            n_feat = min(data.shape[1], self.n_features)
            
            # Pad if needed
            if len(data) < self.seq_len:
                pad = np.zeros((self.seq_len - len(data), n_feat), dtype=np.float32)
                data = np.vstack([pad, data[:, :n_feat]])
            else:
                data = data[:, :n_feat]
            
            # Pad features if needed
            if n_feat < self.n_features:
                data = np.pad(data, ((0, 0), (0, self.n_features - n_feat)))
            
            # Z-score normalize per service
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-8
            data = (data - mean) / std
            
            X[i] = data
        
        # Target: index of root cause service
        target = self.service_to_idx.get(case.root_cause, 0)
        
        return {
            'metrics': torch.from_numpy(X).float(),  # (n_services, seq_len, n_features)
            'target': torch.tensor(target, dtype=torch.long),
            'fault_type': case.fault_type,
            'system': case.system,
            'case_id': case.case_id
        }  # type: ignore


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of samples."""
    return {
        'metrics': torch.stack([b['metrics'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'fault_type': [b['fault_type'] for b in batch],
        'system': [b['system'] for b in batch],
        'case_id': [b['case_id'] for b in batch]
    }


def create_data_loaders(
    data_root: str = 'data/RCAEval',
    batch_size: int = 16,
    seq_len: int = 128,
    n_features: int = 64,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train/val/test data loaders.
    
    Returns:
        train_loader, val_loader, test_loader, services
    """
    # Discover and split cases
    cases = discover_cases(data_root)
    services = get_all_services(cases)
    train_cases, val_cases, test_cases = split_cases(cases, seed=seed)
    
    # Create datasets
    train_dataset = RCADataset(train_cases, services, seq_len, n_features)
    val_dataset = RCADataset(val_cases, services, seq_len, n_features)
    test_dataset = RCADataset(test_cases, services, seq_len, n_features)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    return train_loader, val_loader, test_loader, services
