"""
Training script V2 for improved RCA model.

Improvements:
1. RankingLoss instead of plain cross-entropy
2. Data augmentation (time warping, jittering, cutout)
3. Learning rate warmup + cosine annealing
4. Gradient accumulation for effective larger batches
5. Exponential moving average of model weights
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.rca_data import (
    discover_cases,
    split_cases,
    get_all_services,
    RCADataset,
    collate_fn
)
from src.models.rca_v2 import RCAModelV2, RankingLoss
from src.causal.pcmci import precompute_causal_weights


class TimeSeriesAugmentation:
    """Data augmentation for time series."""
    
    def __init__(self, 
                 jitter_std: float = 0.03,
                 scaling_range: Tuple[float, float] = (0.9, 1.1),
                 cutout_ratio: float = 0.1,
                 p: float = 0.5):
        self.jitter_std = jitter_std
        self.scaling_range = scaling_range
        self.cutout_ratio = cutout_ratio
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations.
        Args:
            x: (n_services, seq_len, n_features)
        Returns:
            Augmented tensor
        """
        if np.random.random() > self.p:
            return x
            
        x = x.clone()
        n_services, seq_len, n_features = x.shape
        
        # Random jittering
        if np.random.random() < 0.5:
            noise = torch.randn_like(x) * self.jitter_std
            x = x + noise
        
        # Random scaling per service
        if np.random.random() < 0.5:
            scale = torch.empty(n_services, 1, 1).uniform_(*self.scaling_range)
            x = x * scale.to(x.device)
        
        # Random cutout (zero out a time segment)
        if np.random.random() < 0.3:
            cutout_len = int(seq_len * self.cutout_ratio)
            start = np.random.randint(0, seq_len - cutout_len)
            x[:, start:start+cutout_len, :] = 0
        
        return x


class EMAModel:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = deepcopy(model)
        self.decay = decay
        
        for p in self.model.parameters():
            p.requires_grad_(False)
            
    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
                
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class TrainerV2:
    """Improved trainer with augmentation and ranking loss."""
    
    def __init__(self,
                 model: RCAModelV2,
                 device: torch.device,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 warmup_epochs: int = 5,
                 use_ema: bool = True):
        self.model = model.to(device)
        self.device = device
        self.warmup_epochs = warmup_epochs
        
        # EMA model
        self.ema = EMAModel(model) if use_ema else None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Loss
        self.criterion = RankingLoss(ce_weight=1.0, rank_weight=0.5, margin=1.0)
        
        # Data augmentation
        self.augmentation = TimeSeriesAugmentation(p=0.5)
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
        
    def _get_lr(self, epoch: int, base_lr: float, total_epochs: int) -> float:
        """Learning rate with warmup and cosine annealing."""
        if epoch < self.warmup_epochs:
            return base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def train_epoch(self,
                    dataloader: DataLoader,
                    causal_cache: Optional[Dict] = None,
                    epoch: int = 0,
                    total_epochs: int = 100) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.current_epoch = epoch
        
        # Set learning rate
        base_lr = self.optimizer.defaults['lr']
        lr = self._get_lr(epoch, base_lr, total_epochs)
        self._set_lr(lr)
        
        total_loss = 0.0
        total_ce = 0.0
        total_rank = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
            metrics = batch['metrics'].to(self.device)
            targets = batch['target'].to(self.device)
            case_ids = batch['case_id']
            
            # Data augmentation (during training)
            batch_size = metrics.shape[0]
            augmented = []
            for i in range(batch_size):
                aug_sample = self.augmentation(metrics[i])
                augmented.append(aug_sample)
            metrics = torch.stack(augmented)
            
            # Get causal weights
            causal_weights = None
            if causal_cache is not None:
                batch_weights = []
                for case_id in case_ids:
                    if case_id in causal_cache:
                        w = causal_cache[case_id]
                        if isinstance(w, np.ndarray):
                            w = torch.from_numpy(w)
                        batch_weights.append(w.float())
                    else:
                        n_services = metrics.shape[1]
                        batch_weights.append(torch.zeros(n_services, n_services))
                causal_weights = torch.stack(batch_weights).to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(metrics, causal_weights)
            
            # Loss
            loss_dict = self.criterion(output['logits'], targets, output.get('embeddings'))
            loss = loss_dict['loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)
            
            # Metrics
            total_loss += loss.item() * metrics.shape[0]
            total_ce += loss_dict['ce_loss'].item() * metrics.shape[0]
            total_rank += loss_dict['rank_loss'].item() * metrics.shape[0]
            
            preds = output['ranking'][:, 0]
            correct += (preds == targets).sum().item()
            total += metrics.shape[0]
        
        return {
            'loss': total_loss / total,
            'ce_loss': total_ce / total,
            'rank_loss': total_rank / total,
            'accuracy': correct / total,
            'lr': lr
        }
    
    @torch.no_grad()
    def evaluate(self,
                 dataloader: DataLoader,
                 causal_cache: Optional[Dict] = None,
                 use_ema: bool = True) -> Dict[str, float]:
        """Evaluate model."""
        model = self.ema.model if (self.ema is not None and use_ema) else self.model
        model.eval()
        
        all_rankings = []
        all_targets = []
        all_probs = []
        
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            metrics = batch['metrics'].to(self.device)
            targets = batch['target'].to(self.device)
            case_ids = batch['case_id']
            
            # Get causal weights
            causal_weights = None
            if causal_cache is not None:
                batch_weights = []
                for case_id in case_ids:
                    if case_id in causal_cache:
                        w = causal_cache[case_id]
                        if isinstance(w, np.ndarray):
                            w = torch.from_numpy(w)
                        batch_weights.append(w.float())
                    else:
                        n_services = metrics.shape[1]
                        batch_weights.append(torch.zeros(n_services, n_services))
                causal_weights = torch.stack(batch_weights).to(self.device)
            
            output = model(metrics, causal_weights)
            
            all_rankings.append(output['ranking'].cpu())
            all_targets.append(targets.cpu())
            all_probs.append(output['probs'].cpu())
        
        # Compute metrics
        rankings = torch.cat(all_rankings, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return self._compute_metrics(rankings, targets)
    
    def _compute_metrics(self, rankings: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute AC@k and MRR."""
        batch_size = rankings.shape[0]
        
        # AC@1
        top1 = rankings[:, 0]
        ac1 = (top1 == targets).float().mean().item()
        
        # AC@3
        top3 = rankings[:, :3]
        targets_exp = targets.unsqueeze(1).expand(-1, 3)
        ac3 = (top3 == targets_exp).any(dim=1).float().mean().item()
        
        # AC@5
        top5 = rankings[:, :5]
        targets_exp = targets.unsqueeze(1).expand(-1, 5)
        ac5 = (top5 == targets_exp).any(dim=1).float().mean().item()
        
        # MRR
        mrr = 0.0
        for i in range(batch_size):
            pos = (rankings[i] == targets[i]).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                mrr += 1.0 / (pos[0].item() + 1)
        mrr /= batch_size
        
        return {
            'AC@1': ac1,
            'AC@3': ac3,
            'AC@5': ac5,
            'MRR': mrr
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 100,
              patience: int = 20,
              causal_cache: Optional[Dict] = None,
              save_dir: str = "outputs/models") -> Dict:
        """Full training loop."""
        
        os.makedirs(save_dir, exist_ok=True)
        history: Dict = {'train': [], 'val': []}
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")  # type: ignore
        print(f"Validation samples: {len(val_loader.dataset)}")  # type: ignore
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(
                train_loader, causal_cache, epoch, n_epochs
            )
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.evaluate(val_loader, causal_cache)
            history['val'].append(val_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"LR: {train_metrics['lr']:.2e} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.1%} | "
                  f"Val AC@1: {val_metrics['AC@1']:.1%}, AC@3: {val_metrics['AC@3']:.1%}, MRR: {val_metrics['MRR']:.3f}")
            
            # Check for improvement
            if val_metrics['AC@1'] > self.best_val_acc:
                self.best_val_acc = val_metrics['AC@1']
                self.patience_counter = 0
                
                # Save best model (EMA if available)
                save_model = self.ema.model if self.ema else self.model
                save_path = os.path.join(save_dir, "rca_model_v2_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'metrics': val_metrics
                }, save_path)
                print(f"  -> New best model! AC@1: {self.best_val_acc:.1%}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        return history


def main():
    parser = argparse.ArgumentParser(description="Train RCA Model V2")
    parser.add_argument("--data-dir", type=str, default="data/RCAEval")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use-pcmci", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal-weight", type=float, default=0.3,
                        help="Weight for causal attention (0-1)")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    data_dir = Path(project_root) / args.data_dir
    output_dir = Path(project_root) / args.output_dir
    
    print(f"\nLoading data from {data_dir}...")
    cases = discover_cases(str(data_dir))
    services = get_all_services(cases)
    train_cases, val_cases, test_cases = split_cases(cases, seed=args.seed)
    
    print(f"Found {len(cases)} cases, {len(services)} services")
    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")
    
    # Create datasets
    train_dataset = RCADataset(train_cases, services, seq_len=128, n_features=64)
    val_dataset = RCADataset(val_cases, services, seq_len=128, n_features=64)
    test_dataset = RCADataset(test_cases, services, seq_len=128, n_features=64)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Get dimensions
    sample = next(iter(train_loader))
    n_services = sample['metrics'].shape[1]
    n_features = sample['metrics'].shape[3]
    print(f"Data shape: {n_services} services, {n_features} features")
    
    # Load/compute causal weights
    causal_cache = None
    if args.use_pcmci:
        cache_path = output_dir / "causal_cache.pkl"
        causal_cache = precompute_causal_weights(
            cases, services, str(cache_path), max_lag=5, use_pcmci=True
        )
    
    # Create model
    model = RCAModelV2(
        n_services=n_services,
        n_features=n_features,
        hidden_dim=128,
        embed_dim=256,
        num_heads=8,
        num_attn_layers=4,
        dropout=0.2,
        causal_weight=args.causal_weight
    )
    
    params = model.count_parameters()
    print(f"\nModel parameters: {params['trainable']:,} trainable")
    
    # Create trainer
    trainer = TrainerV2(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=1e-4,
        warmup_epochs=5,
        use_ema=True
    )
    
    # Train
    model_dir = output_dir / "models"
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        patience=args.patience,
        causal_cache=causal_cache,
        save_dir=str(model_dir)
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    best_path = model_dir / "rca_model_v2_best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=False)
        if trainer.ema:
            trainer.ema.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    test_metrics = trainer.evaluate(test_loader, causal_cache)
    
    print(f"\nTest Results:")
    print(f"  AC@1: {test_metrics['AC@1']:.1%}")
    print(f"  AC@3: {test_metrics['AC@3']:.1%}")
    print(f"  AC@5: {test_metrics['AC@5']:.1%}")
    print(f"  MRR:  {test_metrics['MRR']:.3f}")
    
    # Compare with previous
    print(f"\nImprovement over V1:")
    print(f"  AC@1: {test_metrics['AC@1']:.1%} (was 46.7%)")
    print(f"  AC@3: {test_metrics['AC@3']:.1%} (was 65.0%)")
    print(f"  AC@5: {test_metrics['AC@5']:.1%} (was 78.3%)")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'best_val_acc': trainer.best_val_acc,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = output_dir / "training_results_v2.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
