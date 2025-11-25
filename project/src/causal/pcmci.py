"""
PCMCI causal discovery for root cause analysis.

Uses tigramite library for temporal causal discovery on metrics data.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


def compute_pcmci_weights(
    service_data: Dict[str, np.ndarray],
    services: List[str],
    max_lag: int = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Compute causal weights using PCMCI algorithm.
    
    Args:
        service_data: Dict mapping service name to (T, features) array
        services: List of service names (determines order)
        max_lag: Maximum lag for causal discovery
        alpha: Significance level
        
    Returns:
        (n_services, n_services) causal weight matrix
    """
    n_services = len(services)
    
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        # Return identity if tigramite not available
        return np.eye(n_services, dtype=np.float32)
    
    # Build service-level time series (mean of metrics per service)
    service_ts = []
    for service in services:
        data = service_data.get(service, np.zeros((100, 10)))
        # Aggregate to single time series per service (mean across features)
        mean_ts = data.mean(axis=1)
        service_ts.append(mean_ts)
    
    # Stack into (T, N) array
    data_array = np.column_stack(service_ts).astype(np.float32)
    
    # Handle NaN/Inf
    data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Subsample if too long (PCMCI is slow on long sequences)
    if len(data_array) > 200:
        step = len(data_array) // 200
        data_array = data_array[::step]
    
    # Create tigramite dataframe
    dataframe = pp.DataFrame(data_array, var_names=services)
    
    # Initialize PCMCI
    parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    
    # Run PCMCI
    try:
        results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=alpha)
        
        # Extract causal matrix (sum over lags)
        # results['val_matrix'] has shape (N, N, tau_max+1)
        val_matrix = results['val_matrix']
        
        # Sum absolute values over lags for overall causal strength
        causal_weights = np.abs(val_matrix).sum(axis=2)
        
        # Normalize to [0, 1]
        if causal_weights.max() > 0:
            causal_weights = causal_weights / causal_weights.max()
        
        return causal_weights.astype(np.float32)
        
    except Exception as e:
        # Return identity on failure
        return np.eye(n_services, dtype=np.float32)


def precompute_causal_weights(
    cases,  # List[FailureCase]
    services: List[str],
    cache_path: str = 'outputs/causal_cache.pkl',
    max_lag: int = 5,
    use_pcmci: bool = True
) -> Dict[str, np.ndarray]:
    """
    Precompute PCMCI weights for all cases.
    
    Args:
        cases: List of FailureCase objects
        services: List of service names
        cache_path: Path to save/load cache
        max_lag: Maximum lag for PCMCI
        use_pcmci: Whether to use PCMCI (False for quick testing)
        
    Returns:
        Dict mapping case_id to causal weight matrix
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        print(f"Loading cached causal weights from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    causal_cache = {}
    n_services = len(services)
    
    if not use_pcmci:
        print("Skipping PCMCI (using identity weights)")
        for case in cases:
            causal_cache[case.case_id] = np.eye(n_services, dtype=np.float32)
        return causal_cache
    
    print(f"Computing PCMCI causal weights for {len(cases)} cases...")
    print("This is slow (~1-5 sec/case) but improves accuracy significantly.")
    print("Progress will be saved incrementally.")
    
    # Process with periodic saving
    save_interval = 20
    
    for i, case in enumerate(tqdm(cases, desc="PCMCI")):
        service_data = case.get_service_metrics(services)
        weights = compute_pcmci_weights(service_data, services, max_lag=max_lag)
        causal_cache[case.case_id] = weights
        
        # Save periodically
        if (i + 1) % save_interval == 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(causal_cache, f)
    
    # Final save
    with open(cache_path, 'wb') as f:
        pickle.dump(causal_cache, f)
    print(f"Saved causal cache to {cache_path}")
    
    return causal_cache
