"""
Multimodal data loading for RCA with metrics, logs, and traces.

Uses pre-aggregated files from RCAEval RE2:
- metrics.csv or data.csv: Container metrics
- logts.csv: Log template counts per service per timestep
- tracets_lat.csv: Trace latency per service/method per timestep
- tracets_err.csv: Trace errors per service/method per timestep
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
class MultimodalFailureCase:
    """Single failure case with multimodal data."""
    case_id: str
    system: str  # TrainTicket, SockShop, OnlineBoutique
    root_cause: str  # Service that caused the fault
    fault_type: str  # cpu, mem, delay, loss, disk
    data_path: str
    has_multimodal: bool = False  # True if RE2 data available
    
    # Cached data
    _metrics: Optional[pd.DataFrame] = field(default=None, repr=False)
    _logts: Optional[pd.DataFrame] = field(default=None, repr=False)
    _tracets_lat: Optional[pd.DataFrame] = field(default=None, repr=False)
    _tracets_err: Optional[pd.DataFrame] = field(default=None, repr=False)
    
    def load_metrics(self) -> pd.DataFrame:
        """Load raw metrics from CSV."""
        if self._metrics is None:
            # Try metrics.csv first, then data.csv
            metrics_path = os.path.join(self.data_path, 'metrics.csv')
            if os.path.exists(metrics_path):
                self._metrics = pd.read_csv(metrics_path)
            else:
                self._metrics = pd.read_csv(os.path.join(self.data_path, 'data.csv'))
        return self._metrics
    
    def load_logts(self) -> Optional[pd.DataFrame]:
        """Load log template time series."""
        if not self.has_multimodal:
            return None
        if self._logts is None:
            path = os.path.join(self.data_path, 'logts.csv')
            if os.path.exists(path):
                self._logts = pd.read_csv(path)
        return self._logts
    
    def load_tracets_lat(self) -> Optional[pd.DataFrame]:
        """Load trace latency time series."""
        if not self.has_multimodal:
            return None
        if self._tracets_lat is None:
            path = os.path.join(self.data_path, 'tracets_lat.csv')
            if os.path.exists(path):
                self._tracets_lat = pd.read_csv(path)
        return self._tracets_lat
    
    def load_tracets_err(self) -> Optional[pd.DataFrame]:
        """Load trace error time series."""
        if not self.has_multimodal:
            return None
        if self._tracets_err is None:
            path = os.path.join(self.data_path, 'tracets_err.csv')
            if os.path.exists(path):
                self._tracets_err = pd.read_csv(path)
        return self._tracets_err
    
    def get_service_data(self, services: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get all modality data organized by service.
        
        Returns:
            {service: {'metrics': array, 'logs': array, 'traces': array}}
        """
        result = {}
        
        # Load all data
        metrics_df = self.load_metrics()
        logts_df = self.load_logts()
        tracets_lat_df = self.load_tracets_lat()
        tracets_err_df = self.load_tracets_err()
        
        for service in services:
            service_data = {}
            
            # === Metrics ===
            # Find columns for this service (handle different naming conventions)
            service_cols = [c for c in metrics_df.columns 
                          if c.startswith(service + '_') or c.startswith(service + '-')]
            if service_cols:
                data = metrics_df[service_cols].values.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                data = np.zeros((len(metrics_df), 1), dtype=np.float32)
            service_data['metrics'] = data
            
            # === Logs (from logts.csv) ===
            if logts_df is not None:
                # Column format: "service_templateID"
                log_cols = [c for c in logts_df.columns 
                           if c != 'time' and self._service_matches(c, service)]
                if log_cols:
                    data = logts_df[log_cols].values.astype(np.float32)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    data = np.zeros((len(logts_df), 1), dtype=np.float32)
                service_data['logs'] = data
            else:
                service_data['logs'] = None
            
            # === Traces (latency + errors) ===
            if tracets_lat_df is not None and tracets_err_df is not None:
                # Column format: "service_method" or "serviceservice_method"
                lat_cols = [c for c in tracets_lat_df.columns 
                           if c != 'time' and self._trace_service_matches(c, service)]
                err_cols = [c for c in tracets_err_df.columns 
                           if c != 'time' and self._trace_service_matches(c, service)]
                
                if lat_cols:
                    lat_data = tracets_lat_df[lat_cols].values.astype(np.float32)
                    lat_data = np.nan_to_num(lat_data, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    lat_data = np.zeros((len(tracets_lat_df), 1), dtype=np.float32)
                
                if err_cols:
                    err_data = tracets_err_df[err_cols].values.astype(np.float32)
                    err_data = np.nan_to_num(err_data, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    err_data = np.zeros((len(tracets_err_df), 1), dtype=np.float32)
                
                # Combine latency and error features
                # Pad to same feature count
                max_feat = max(lat_data.shape[1], err_data.shape[1])
                if lat_data.shape[1] < max_feat:
                    lat_data = np.pad(lat_data, ((0, 0), (0, max_feat - lat_data.shape[1])))
                if err_data.shape[1] < max_feat:
                    err_data = np.pad(err_data, ((0, 0), (0, max_feat - err_data.shape[1])))
                
                # Stack: (T, 2*feat) - latency then errors
                service_data['traces'] = np.concatenate([lat_data, err_data], axis=1)
            else:
                service_data['traces'] = None
            
            result[service] = service_data
        
        return result
    
    def _service_matches(self, col_name: str, service: str) -> bool:
        """Check if logts column belongs to service."""
        # Format: "servicename_templateID"
        parts = col_name.rsplit('_', 1)
        if len(parts) != 2:
            return False
        col_service = parts[0]
        return col_service.lower() == service.lower() or col_service.lower().replace('-', '') == service.lower().replace('-', '')
    
    def _trace_service_matches(self, col_name: str, service: str) -> bool:
        """Check if trace column belongs to service."""
        # Format: "serviceservice_method" (e.g., "currencyservice_Convert")
        # or "service_method" 
        col_lower = col_name.lower()
        service_lower = service.lower().replace('-', '')
        # Check if column starts with service name
        return col_lower.startswith(service_lower) or col_lower.startswith(service.lower())
    
    def clear_cache(self):
        """Clear cached data to free memory."""
        self._metrics = None
        self._logts = None
        self._tracets_lat = None
        self._tracets_err = None


def discover_multimodal_cases(data_root: Union[str, Path] = 'data/RCAEval',
                               require_multimodal: bool = True) -> List[MultimodalFailureCase]:
    """
    Discover all failure cases, optionally filtering for multimodal (RE2) only.
    
    Args:
        data_root: Root path to RCAEval dataset
        require_multimodal: If True, only return RE2 cases with logs/traces
    """
    cases = []
    data_root = Path(data_root) if isinstance(data_root, str) else data_root
    
    for system in ['TrainTicket', 'SockShop', 'OnlineBoutique']:
        sys_path = data_root / system
        if not sys_path.exists():
            continue
            
        for re_dir in sys_path.iterdir():
            if not re_dir.is_dir():
                continue
            
            # Check if this is RE2 (has multimodal data)
            is_re2 = 'RE2' in re_dir.name
            
            if require_multimodal and not is_re2:
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
                        if not case_dir.is_dir():
                            continue
                        # Check for required files
                        has_data = (case_dir / 'data.csv').exists() or (case_dir / 'metrics.csv').exists()
                        if not has_data:
                            continue
                        
                        # Check for multimodal files
                        has_multimodal = (
                            (case_dir / 'logts.csv').exists() and
                            (case_dir / 'tracets_lat.csv').exists() and
                            (case_dir / 'tracets_err.csv').exists()
                        )
                        
                        if require_multimodal and not has_multimodal:
                            continue
                        
                        case_id = f"{system}_{re_dir.name}_{fault_dir.name}_{case_dir.name}"
                        cases.append(MultimodalFailureCase(
                            case_id=case_id,
                            system=system,
                            root_cause=service,
                            fault_type=fault_type,
                            data_path=str(case_dir),
                            has_multimodal=has_multimodal
                        ))
    
    return cases


def get_all_services_multimodal(cases: List[MultimodalFailureCase]) -> List[str]:
    """Extract all unique service names from multimodal cases."""
    services = set()
    for case in cases:
        services.add(case.root_cause)
    return sorted(list(services))


def split_multimodal_cases(
    cases: List[MultimodalFailureCase], 
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[MultimodalFailureCase], List[MultimodalFailureCase], List[MultimodalFailureCase]]:
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


class MultimodalRCADataset(Dataset):
    """
    PyTorch dataset for multimodal RCA training.
    
    Each sample contains:
    - metrics: (n_services, seq_len, n_metric_features)
    - logs: (n_services, seq_len, n_log_features) or None
    - traces: (n_services, seq_len, n_trace_features) or None
    - target: service index
    """
    
    def __init__(self, 
                 cases: List[MultimodalFailureCase],
                 services: List[str],
                 seq_len: int = 60,  # Shorter for RE2 (15s intervals)
                 n_metric_features: int = 64,
                 n_log_features: int = 32,
                 n_trace_features: int = 32):
        self.cases = cases
        self.services = services
        self.service_to_idx = {s: i for i, s in enumerate(services)}
        self.seq_len = seq_len
        self.n_metric_features = n_metric_features
        self.n_log_features = n_log_features
        self.n_trace_features = n_trace_features
        
    def __len__(self):
        return len(self.cases)
    
    def _process_modality(self, data: Optional[np.ndarray], n_features: int) -> np.ndarray:
        """Process a single modality's data to standard shape."""
        if data is None:
            return np.zeros((self.seq_len, n_features), dtype=np.float32)
        
        # Take last seq_len timesteps
        if len(data) > self.seq_len:
            data = data[-self.seq_len:]
        
        # Truncate features if needed
        n_feat = min(data.shape[1] if len(data.shape) > 1 else 1, n_features)
        
        # Pad time if needed
        if len(data) < self.seq_len:
            pad = np.zeros((self.seq_len - len(data), n_feat), dtype=np.float32)
            if len(data.shape) > 1:
                data = np.vstack([pad, data[:, :n_feat]])
            else:
                data = np.vstack([pad, data.reshape(-1, 1)[:, :n_feat]])
        else:
            if len(data.shape) > 1:
                data = data[:, :n_feat]
            else:
                data = data.reshape(-1, 1)[:, :n_feat]
        
        # Pad features if needed
        if n_feat < n_features:
            data = np.pad(data, ((0, 0), (0, n_features - n_feat)))
        
        # Z-score normalize
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mean) / std
        
        # data is guaranteed to be ndarray at this point
        assert data is not None
        return data.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.cases[idx]
        service_data = case.get_service_data(self.services)
        
        n_services = len(self.services)
        
        # Initialize arrays
        metrics_arr = np.zeros((n_services, self.seq_len, self.n_metric_features), dtype=np.float32)
        logs_arr = np.zeros((n_services, self.seq_len, self.n_log_features), dtype=np.float32)
        traces_arr = np.zeros((n_services, self.seq_len, self.n_trace_features), dtype=np.float32)
        
        has_logs = False
        has_traces = False
        
        for i, service in enumerate(self.services):
            sdata = service_data.get(service, {})
            
            # Metrics
            metrics_arr[i] = self._process_modality(
                sdata.get('metrics'), self.n_metric_features
            )
            
            # Logs
            if sdata.get('logs') is not None:
                logs_arr[i] = self._process_modality(
                    sdata.get('logs'), self.n_log_features
                )
                has_logs = True
            
            # Traces
            if sdata.get('traces') is not None:
                traces_arr[i] = self._process_modality(
                    sdata.get('traces'), self.n_trace_features
                )
                has_traces = True
        
        # Target: index of root cause service
        target = self.service_to_idx.get(case.root_cause, 0)
        
        return {
            'metrics': torch.from_numpy(metrics_arr).float(),
            'logs': torch.from_numpy(logs_arr).float() if has_logs else None,
            'traces': torch.from_numpy(traces_arr).float() if has_traces else None,
            'target': torch.tensor(target, dtype=torch.long),
            'fault_type': case.fault_type,
            'system': case.system,
            'case_id': case.case_id,
            'has_multimodal': case.has_multimodal
        }


def multimodal_collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of multimodal samples."""
    # Stack tensors
    metrics = torch.stack([b['metrics'] for b in batch])
    targets = torch.stack([b['target'] for b in batch])
    
    # Handle optional modalities
    has_logs = any(b['logs'] is not None for b in batch)
    has_traces = any(b['traces'] is not None for b in batch)
    
    if has_logs:
        # Create zero tensors for samples without logs
        logs_list = []
        for b in batch:
            if b['logs'] is not None:
                logs_list.append(b['logs'])
            else:
                # Use zeros with same shape
                logs_list.append(torch.zeros_like(batch[0]['logs'] if batch[0]['logs'] is not None 
                                                   else torch.zeros(metrics.shape[1], 60, 32)))
        logs = torch.stack(logs_list)
    else:
        logs = None
    
    if has_traces:
        traces_list = []
        for b in batch:
            if b['traces'] is not None:
                traces_list.append(b['traces'])
            else:
                traces_list.append(torch.zeros_like(batch[0]['traces'] if batch[0]['traces'] is not None 
                                                     else torch.zeros(metrics.shape[1], 60, 32)))
        traces = torch.stack(traces_list)
    else:
        traces = None
    
    return {
        'metrics': metrics,
        'logs': logs,
        'traces': traces,
        'target': targets,
        'fault_type': [b['fault_type'] for b in batch],
        'system': [b['system'] for b in batch],
        'case_id': [b['case_id'] for b in batch],
        'has_multimodal': [b['has_multimodal'] for b in batch]
    }


def create_multimodal_loaders(
    data_root: str = 'data/RCAEval',
    batch_size: int = 16,
    seq_len: int = 60,
    n_metric_features: int = 64,
    n_log_features: int = 32,
    n_trace_features: int = 32,
    seed: int = 42,
    require_multimodal: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train/val/test data loaders for multimodal RCA.
    
    Returns:
        train_loader, val_loader, test_loader, services
    """
    # Discover and split cases
    cases = discover_multimodal_cases(data_root, require_multimodal=require_multimodal)
    services = get_all_services_multimodal(cases)
    train_cases, val_cases, test_cases = split_multimodal_cases(cases, seed=seed)
    
    print(f"Discovered {len(cases)} multimodal cases")
    print(f"Services: {len(services)} - {services[:5]}...")
    print(f"Split: {len(train_cases)} train, {len(val_cases)} val, {len(test_cases)} test")
    
    # Create datasets
    train_dataset = MultimodalRCADataset(
        train_cases, services, seq_len, n_metric_features, n_log_features, n_trace_features
    )
    val_dataset = MultimodalRCADataset(
        val_cases, services, seq_len, n_metric_features, n_log_features, n_trace_features
    )
    test_dataset = MultimodalRCADataset(
        test_cases, services, seq_len, n_metric_features, n_log_features, n_trace_features
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=multimodal_collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=multimodal_collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=multimodal_collate_fn, num_workers=0
    )
    
    return train_loader, val_loader, test_loader, services


if __name__ == '__main__':
    # Test the data loader
    cases = discover_multimodal_cases('data/RCAEval', require_multimodal=True)
    print(f"Found {len(cases)} multimodal cases")
    
    if cases:
        # Test loading one case
        case = cases[0]
        print(f"\nTest case: {case.case_id}")
        print(f"System: {case.system}, Root cause: {case.root_cause}, Fault: {case.fault_type}")
        
        services = get_all_services_multimodal(cases)
        print(f"\nAll services ({len(services)}): {services[:5]}...")
        
        data = case.get_service_data(services[:3])
        for svc, sdata in data.items():
            print(f"\n{svc}:")
            print(f"  metrics: {sdata['metrics'].shape}")
            print(f"  logs: {sdata['logs'].shape if sdata['logs'] is not None else None}")
            print(f"  traces: {sdata['traces'].shape if sdata['traces'] is not None else None}")
