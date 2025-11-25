"""
Training script V3 - Focused on generalization.

Key changes:
1. Higher dropout and weight decay
2. No EMA (can hurt with small datasets)
3. Cosine annealing with restarts
4. Gradient clipping
5. Better data split validation
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.rca_data import (
    discover_cases,
    split_cases,
    get_all_services,
    RCADataset,
    collate_fn
)
from src.models.rca_v3 import RCAModelV3, BalancedLoss
from src.causal.pcmci import precompute_causal_weights


class TrainerV3:
    """Simple, effective trainer focused on generalization."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 lr: float = 1e-3,
                 weight_decay: float = 0.01):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.criterion = BalancedLoss(smoothing=0.1, rank_weight=0.3, margin=0.5)
        self.best_val_acc = 0.0
        self.best_val_mrr = 0.0
        self.patience_counter = 0
        
    def train_epoch(self,
                    dataloader: DataLoader,
                    causal_cache: Optional[Dict] = None) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            metrics = batch['metrics'].to(self.device)
            targets = batch['target'].to(self.device)
            case_ids = batch['case_id']
            
            # Get causal weights
            causal_weights = self._get_causal_weights(case_ids, causal_cache, metrics.shape[1])
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(metrics, causal_weights)
            
            loss_dict = self.criterion(output['logits'], targets)
            loss = loss_dict['loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * metrics.shape[0]
            preds = output['ranking'][:, 0]
            correct += (preds == targets).sum().item()
            total += metrics.shape[0]
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    @torch.no_grad()
    def evaluate(self,
                 dataloader: DataLoader,
                 causal_cache: Optional[Dict] = None) -> Dict[str, float]:
        self.model.eval()
        
        all_rankings = []
        all_targets = []
        
        for batch in dataloader:
            metrics = batch['metrics'].to(self.device)
            targets = batch['target'].to(self.device)
            case_ids = batch['case_id']
            
            causal_weights = self._get_causal_weights(case_ids, causal_cache, metrics.shape[1])
            output = self.model(metrics, causal_weights)
            
            all_rankings.append(output['ranking'].cpu())
            all_targets.append(targets.cpu())
        
        rankings = torch.cat(all_rankings, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return self._compute_metrics(rankings, targets)
    
    def _get_causal_weights(self, case_ids, causal_cache, n_services):
        if causal_cache is None:
            return None
            
        batch_weights = []
        for case_id in case_ids:
            if case_id in causal_cache:
                w = causal_cache[case_id]
                if isinstance(w, np.ndarray):
                    w = torch.from_numpy(w)
                batch_weights.append(w.float())
            else:
                batch_weights.append(torch.eye(n_services))
        
        return torch.stack(batch_weights).to(self.device)
    
    def _compute_metrics(self, rankings: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        batch_size = rankings.shape[0]
        
        # AC@1
        ac1 = (rankings[:, 0] == targets).float().mean().item()
        
        # AC@3
        top3 = rankings[:, :3]
        ac3 = (top3 == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        
        # AC@5
        top5 = rankings[:, :5]
        ac5 = (top5 == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        
        # MRR
        mrr = 0.0
        for i in range(batch_size):
            pos = (rankings[i] == targets[i]).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                mrr += 1.0 / (pos[0].item() + 1)
        mrr /= batch_size
        
        return {'AC@1': ac1, 'AC@3': ac3, 'AC@5': ac5, 'MRR': mrr}
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epochs: int = 100,
              patience: int = 25,
              causal_cache: Optional[Dict] = None,
              save_dir: str = "outputs/models") -> Dict:
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Learning rate scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        history: Dict = {'train': [], 'val': []}
        
        print(f"\nTraining for up to {n_epochs} epochs...")
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")  # type: ignore
        
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, causal_cache)
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.evaluate(val_loader, causal_cache)
            history['val'].append(val_metrics)
            
            # Step scheduler
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"LR: {lr:.2e} | "
                  f"Train: {train_metrics['accuracy']:.1%} | "
                  f"Val AC@1: {val_metrics['AC@1']:.1%}, AC@3: {val_metrics['AC@3']:.1%}, MRR: {val_metrics['MRR']:.3f}")
            
            # Check improvement (use MRR for early stopping as it's smoother)
            improved = False
            if val_metrics['MRR'] > self.best_val_mrr:
                self.best_val_mrr = val_metrics['MRR']
                improved = True
            if val_metrics['AC@1'] > self.best_val_acc:
                self.best_val_acc = val_metrics['AC@1']
                improved = True
                
            if improved:
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'best_val_mrr': self.best_val_mrr,
                    'metrics': val_metrics
                }, os.path.join(save_dir, "rca_model_v3_best.pt"))
                print(f"  -> Saved! Best AC@1: {self.best_val_acc:.1%}, MRR: {self.best_val_mrr:.3f}")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        return history


def main():
    parser = argparse.ArgumentParser(description="Train RCA Model V3")
    parser.add_argument("--data-dir", type=str, default="data/RCAEval")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    data_dir = Path(project_root) / args.data_dir
    output_dir = Path(project_root) / args.output_dir
    
    print(f"\nLoading data...")
    cases = discover_cases(str(data_dir))
    services = get_all_services(cases)
    train_cases, val_cases, test_cases = split_cases(cases, seed=args.seed)
    
    print(f"Cases: {len(cases)}, Services: {len(services)}")
    print(f"Split: {len(train_cases)} / {len(val_cases)} / {len(test_cases)}")
    
    # Datasets
    train_dataset = RCADataset(train_cases, services, seq_len=128, n_features=64)
    val_dataset = RCADataset(val_cases, services, seq_len=128, n_features=64)
    test_dataset = RCADataset(test_cases, services, seq_len=128, n_features=64)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    
    # Dimensions
    sample = next(iter(train_loader))
    n_services = sample['metrics'].shape[1]
    n_features = sample['metrics'].shape[3]
    
    # Load causal cache
    cache_path = output_dir / "causal_cache.pkl"
    causal_cache = precompute_causal_weights(cases, services, str(cache_path))
    
    # Model - smaller and more regularized
    model = RCAModelV3(
        n_services=n_services,
        n_features=n_features,
        hidden_dim=64,
        embed_dim=128,
        num_heads=4,
        num_attn_layers=2,
        dropout=args.dropout
    )
    
    params = model.count_parameters()
    print(f"\nModel: {params['trainable']:,} parameters")
    
    # Train
    trainer = TrainerV3(model, device, lr=args.lr, weight_decay=args.weight_decay)
    
    history = trainer.train(
        train_loader, val_loader,
        n_epochs=args.epochs,
        patience=args.patience,
        causal_cache=causal_cache,
        save_dir=str(output_dir / "models")
    )
    
    # Load best and evaluate on test
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    best_path = output_dir / "models" / "rca_model_v3_best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best model from epoch {ckpt['epoch']+1}")
    
    test_metrics = trainer.evaluate(test_loader, causal_cache)
    
    print(f"\nTest Results:")
    print(f"  AC@1: {test_metrics['AC@1']:.1%}")
    print(f"  AC@3: {test_metrics['AC@3']:.1%}")  
    print(f"  AC@5: {test_metrics['AC@5']:.1%}")
    print(f"  MRR:  {test_metrics['MRR']:.3f}")
    
    print(f"\nComparison with V1 (baseline):")
    print(f"  AC@1: {test_metrics['AC@1']:.1%} vs 46.7%")
    print(f"  AC@3: {test_metrics['AC@3']:.1%} vs 65.0%")
    print(f"  AC@5: {test_metrics['AC@5']:.1%} vs 78.3%")
    
    # Save
    with open(output_dir / "test_results_v3.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
