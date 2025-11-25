"""
Multimodal Root Cause Analysis System
=====================================
Complete end-to-end pipeline for microservice fault diagnosis.

Architecture:
- TCN encoder for service-level time series
- PCMCI causal discovery for causal relationships  
- Cross-attention fusion for multimodal integration
- Service ranking head for root cause prediction

Target: Beat SOTA (RUN: 63.1% AC@1) by trading speed for accuracy.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FailureCase:
    """Single failure case with all data."""
    case_id: str
    system: str  # TrainTicket, SockShop, OnlineBoutique
    root_cause: str  # Service that caused the fault
    fault_type: str  # cpu, mem, delay, loss, disk
    data_path: str
    
    # Loaded data (lazy)
    _metrics: Optional[pd.DataFrame] = None
    _service_metrics: Optional[Dict[str, np.ndarray]] = None
    
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


# ============================================================================
# DATA LOADING
# ============================================================================

def discover_cases(data_root: str = 'data/RCAEval') -> List[FailureCase]:
    """Discover all failure cases in the dataset."""
    cases = []
    data_root = Path(data_root)
    
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


# ============================================================================
# PYTORCH DATASET
# ============================================================================

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
            'metrics': torch.from_numpy(X),  # (n_services, seq_len, n_features)
            'target': torch.tensor(target, dtype=torch.long),
            'fault_type': case.fault_type,
            'system': case.system,
            'case_id': case.case_id
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of samples."""
    return {
        'metrics': torch.stack([b['metrics'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'fault_type': [b['fault_type'] for b in batch],
        'system': [b['system'] for b in batch],
        'case_id': [b['case_id'] for b in batch]
    }


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class TemporalBlock(nn.Module):
    """Single TCN block with dilated causal convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # Causal trimming
        self.trim = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, channels, seq_len)"""
        residual = x
        
        out = self.conv1(x)
        out = out[:, :, :-self.trim] if self.trim > 0 else out
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.trim] if self.trim > 0 else out
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Match lengths
        if residual.size(2) > out.size(2):
            residual = residual[:, :, -out.size(2):]
        
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder."""
    
    def __init__(self, 
                 in_channels: int = 64,
                 hidden_channels: int = 128,
                 out_channels: int = 256,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        dilations = [2**i for i in range(num_layers)]
        
        for i, dilation in enumerate(dilations):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, out_channels)
        """
        # (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return x


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention for service interactions."""
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                causal_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_services, embed_dim)
            causal_weights: (batch, n_services, n_services) optional attention bias
        Returns:
            (batch, n_services, embed_dim)
        """
        # Ensure x is 3D
        if x.dim() == 4:
            x = x.mean(dim=2)
        
        # Self-attention across services
        attn_out, _ = self.attention(x, x, x)
        
        # Apply causal weights if provided
        if causal_weights is not None:
            # Use causal weights to reweight service representations
            # causal_weights: (batch, n_services, n_services)
            # Each row i tells us how much service i is causally influenced by other services
            # Take row-wise mean as importance weight for each service
            service_importance = causal_weights.mean(dim=-1, keepdim=True)  # (batch, n_services, 1)
            attn_out = attn_out * (1.0 + service_importance)  # boost important services
        
        x = self.norm(x + self.dropout(attn_out))
        return x


class RCAModel(nn.Module):
    """
    Complete RCA model.
    
    Architecture:
    1. TCN encodes each service's time series
    2. Cross-attention models service interactions
    3. Optional causal weights from PCMCI
    4. Classification head predicts root cause
    """
    
    def __init__(self,
                 n_services: int,
                 n_features: int = 64,
                 tcn_hidden: int = 128,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_attn_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.n_services = n_services
        self.embed_dim = embed_dim
        
        # Service-level TCN encoder
        self.tcn = TCNEncoder(
            in_channels=n_features,
            hidden_channels=tcn_hidden,
            out_channels=embed_dim,
            num_layers=4,
            dropout=dropout
        )
        
        # Learnable service embeddings
        self.service_embed = nn.Embedding(n_services, embed_dim)
        
        # Cross-attention layers
        self.attention_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_attn_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, 
                metrics: torch.Tensor,
                causal_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            metrics: (batch, n_services, seq_len, n_features)
            causal_weights: (batch, n_services, n_services) from PCMCI
        Returns:
            Dict with logits, probs, ranking
        """
        batch_size, n_services, seq_len, n_features = metrics.shape
        
        # Encode each service's time series
        # Reshape: (batch * n_services, seq_len, n_features)
        metrics_flat = metrics.view(batch_size * n_services, seq_len, n_features)
        service_encodings = self.tcn(metrics_flat)  # (batch * n_services, embed_dim)
        service_encodings = service_encodings.view(batch_size, n_services, -1)
        
        # Add learnable service embeddings
        service_ids = torch.arange(n_services, device=metrics.device)
        service_emb = self.service_embed(service_ids)  # (n_services, embed_dim)
        service_encodings = service_encodings + service_emb.unsqueeze(0)
        
        # Cross-attention layers
        x = service_encodings
        for attn_layer in self.attention_layers:
            x = attn_layer(x, causal_weights)
        
        # Classification: score each service
        scores = self.classifier(x).squeeze(-1)  # (batch, n_services)
        
        # Softmax probabilities
        probs = F.softmax(scores, dim=-1)
        
        # Ranking (descending by score)
        ranking = torch.argsort(scores, dim=-1, descending=True)
        
        return {
            'logits': scores,
            'probs': probs,
            'ranking': ranking
        }


# ============================================================================
# PCMCI CAUSAL DISCOVERY
# ============================================================================

def compute_pcmci_weights(case: FailureCase, 
                          services: List[str],
                          max_lag: int = 5,
                          alpha: float = 0.05) -> np.ndarray:
    """
    Compute causal weights using PCMCI algorithm.
    
    Args:
        case: Failure case
        services: List of service names
        max_lag: Maximum lag for causal discovery
        alpha: Significance level
        
    Returns:
        (n_services, n_services) causal weight matrix
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        # Return identity if tigramite not available
        return np.eye(len(services), dtype=np.float32)
    
    service_metrics = case.get_service_metrics(services)
    n_services = len(services)
    
    # Build service-level time series (mean of metrics per service)
    service_ts = []
    for service in services:
        data = service_metrics.get(service, np.zeros((100, 10)))
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


def precompute_causal_weights(cases: List[FailureCase],
                               services: List[str],
                               cache_path: str = 'causal_cache.pkl',
                               max_lag: int = 5,
                               use_pcmci: bool = True) -> Dict[str, np.ndarray]:
    """Precompute PCMCI weights for all cases (slow but thorough)."""
    
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
        weights = compute_pcmci_weights(case, services, max_lag=max_lag)
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


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Training loop with evaluation."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 services: List[str],
                 causal_cache: Dict[str, np.ndarray],
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 weight_decay: float = 0.01):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.services = services
        self.causal_cache = causal_cache
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_ac1 = 0.0
        self.best_model_state = None
        
    def get_causal_weights(self, case_ids: List[str]) -> torch.Tensor:
        """Get causal weights for batch."""
        weights = []
        for case_id in case_ids:
            w = self.causal_cache.get(case_id, np.eye(len(self.services)))
            weights.append(w)
        return torch.tensor(np.stack(weights), dtype=torch.float32, device=self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in self.train_loader:
            metrics = batch['metrics'].to(self.device)
            targets = batch['target'].to(self.device)
            case_ids = batch['case_id']
            
            # Get causal weights
            causal_weights = self.get_causal_weights(case_ids)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(metrics, causal_weights)
            
            # Loss
            loss = self.criterion(outputs['logits'], targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * targets.size(0)
            preds = outputs['logits'].argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        
        all_rankings = []
        all_targets = []
        all_fault_types = []
        all_systems = []
        
        for batch in loader:
            metrics = batch['metrics'].to(self.device)
            targets = batch['target']
            case_ids = batch['case_id']
            
            causal_weights = self.get_causal_weights(case_ids)
            outputs = self.model(metrics, causal_weights)
            
            all_rankings.append(outputs['ranking'].cpu())
            all_targets.append(targets)
            all_fault_types.extend(batch['fault_type'])
            all_systems.extend(batch['system'])
        
        rankings = torch.cat(all_rankings, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        results = {}
        
        # AC@k
        for k in [1, 3, 5]:
            top_k = rankings[:, :k]
            hits = (top_k == targets.unsqueeze(1)).any(dim=1)
            results[f'AC@{k}'] = hits.float().mean().item()
        
        # MRR
        ranks = []
        for i in range(len(targets)):
            rank = (rankings[i] == targets[i]).nonzero()
            if len(rank) > 0:
                ranks.append(1.0 / (rank[0].item() + 1))
            else:
                ranks.append(0.0)
        results['MRR'] = np.mean(ranks)
        
        return results
    
    def train(self, epochs: int = 50, patience: int = 10) -> Dict:
        """Full training loop."""
        history = {'train': [], 'val': []}
        no_improve = 0
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            history['val'].append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Check improvement
            if val_metrics['AC@1'] > self.best_ac1:
                self.best_ac1 = val_metrics['AC@1']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val AC@1: {val_metrics['AC@1']:.4f} | "
                  f"Val AC@3: {val_metrics['AC@3']:.4f} | "
                  f"Val MRR: {val_metrics['MRR']:.4f}")
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return history
    
    def test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        return self.evaluate(self.test_loader)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    print("=" * 70)
    print("MULTIMODAL ROOT CAUSE ANALYSIS SYSTEM")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print("\n[1/5] Loading dataset...")
    cases = discover_cases('data/RCAEval')
    services = get_all_services(cases)
    print(f"  Total cases: {len(cases)}")
    print(f"  Services: {len(services)}")
    print(f"  Services: {services}")
    
    # Split data
    print("\n[2/5] Splitting data...")
    train_cases, val_cases, test_cases = split_cases(cases)
    print(f"  Train: {len(train_cases)}")
    print(f"  Val: {len(val_cases)}")
    print(f"  Test: {len(test_cases)}")
    
    # Compute PCMCI weights (slow but important)
    print("\n[3/5] Computing PCMCI causal weights...")
    all_cases = train_cases + val_cases + test_cases
    causal_cache = precompute_causal_weights(
        all_cases, services, 
        cache_path='outputs/causal_cache.pkl',
        max_lag=5,
        use_pcmci=True  # Set to False for quick testing without PCMCI
    )
    
    # Create datasets
    print("\n[4/5] Creating data loaders...")
    train_dataset = RCADataset(train_cases, services, seq_len=128, n_features=64)
    val_dataset = RCADataset(val_cases, services, seq_len=128, n_features=64)
    test_dataset = RCADataset(test_cases, services, seq_len=128, n_features=64)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    
    # Create model
    print("\n[5/5] Building model...")
    model = RCAModel(
        n_services=len(services),
        n_features=64,
        tcn_hidden=128,
        embed_dim=256,
        num_heads=8,
        num_attn_layers=3,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        services=services,
        causal_cache=causal_cache,
        device=device,
        lr=1e-4,
        weight_decay=0.01
    )
    
    history = trainer.train(epochs=50, patience=15)
    
    # Test
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    test_results = trainer.test()
    print(f"  AC@1: {test_results['AC@1']:.4f} ({test_results['AC@1']*100:.1f}%)")
    print(f"  AC@3: {test_results['AC@3']:.4f} ({test_results['AC@3']*100:.1f}%)")
    print(f"  AC@5: {test_results['AC@5']:.4f} ({test_results['AC@5']*100:.1f}%)")
    print(f"  MRR:  {test_results['MRR']:.4f}")
    
    # Save model
    os.makedirs('outputs/models', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/models/rca_model.pt')
    print("\nModel saved to outputs/models/rca_model.pt")
    
    # Save results
    with open('outputs/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    print("Results saved to outputs/test_results.json")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    
    return test_results


if __name__ == '__main__':
    main()
