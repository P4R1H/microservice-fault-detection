"""
Training script for RCA model.

Usage:
    python scripts/train_rca.py --config config/model_config.yaml
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.rca_data import (
    FailureCase, 
    discover_cases, 
    split_cases, 
    RCADataset, 
    collate_fn,
    create_data_loaders
)
from src.encoders.tcn import TCNEncoder
from src.models.rca import RCAModel
from src.causal.pcmci import precompute_causal_weights


class Trainer:
    """Training manager for RCA model."""
    
    def __init__(self,
                 model: RCAModel,
                 device: torch.device,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def train_epoch(self,
                    dataloader: DataLoader,
                    causal_cache: Optional[Dict] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            metrics = batch['metrics'].to(self.device)
            targets = batch['target'].to(self.device)
            case_ids = batch['case_id']
            
            # Get causal weights
            causal_weights = None
            if causal_cache is not None:
                batch_weights = []
                for case_id in case_ids:
                    if case_id in causal_cache:
                        batch_weights.append(causal_cache[case_id])
                    else:
                        n_services = metrics.shape[1]
                        batch_weights.append(torch.zeros(n_services, n_services))
                causal_weights = torch.stack(batch_weights).to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(metrics, causal_weights)
            loss = self.criterion(output['logits'], targets)
            
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
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        
        all_rankings = []
        all_targets = []
        
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
                        batch_weights.append(causal_cache[case_id])
                    else:
                        n_services = metrics.shape[1]
                        batch_weights.append(torch.zeros(n_services, n_services))
                causal_weights = torch.stack(batch_weights).to(self.device)
            
            output = self.model(metrics, causal_weights)
            loss = self.criterion(output['logits'], targets)
            
            total_loss += loss.item() * metrics.shape[0]
            all_rankings.append(output['ranking'].cpu())
            all_targets.append(targets.cpu())
        
        # Compute metrics
        rankings = torch.cat(all_rankings, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        metrics = self._compute_metrics(rankings, targets)
        metrics['loss'] = total_loss / len(targets)
        
        return metrics
    
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
              patience: int = 15,
              causal_cache: Optional[Dict] = None,
              save_dir: str = "outputs/models") -> Dict:
        """Full training loop."""
        
        os.makedirs(save_dir, exist_ok=True)
        history: Dict = {'train': [], 'val': []}
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")  # type: ignore
        print(f"Validation samples: {len(val_loader.dataset)}")  # type: ignore
        
        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, causal_cache)
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.evaluate(val_loader, causal_cache)
            history['val'].append(val_metrics)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.1%} | "
                  f"Val AC@1: {val_metrics['AC@1']:.1%}, AC@3: {val_metrics['AC@3']:.1%}, MRR: {val_metrics['MRR']:.3f}")
            
            # Check for improvement
            if val_metrics['AC@1'] > self.best_val_acc:
                self.best_val_acc = val_metrics['AC@1']
                self.patience_counter = 0
                
                # Save best model
                save_path = os.path.join(save_dir, "rca_model_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'metrics': val_metrics
                }, save_path)
                print(f"  -> New best model saved! AC@1: {self.best_val_acc:.1%}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        return history


def main():
    parser = argparse.ArgumentParser(description="Train RCA Model")
    parser.add_argument("--data-dir", type=str, default="data/RCAEval",
                        help="Path to RCAEval data directory")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--use-pcmci", action="store_true", default=True,
                        help="Use PCMCI causal discovery")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Discover cases
    data_dir = Path(project_root) / args.data_dir
    output_dir = Path(project_root) / args.output_dir
    
    print(f"\nLoading data from {data_dir}...")
    
    # Create data loaders using the convenience function
    train_loader, val_loader, test_loader, services = create_data_loaders(
        data_root=str(data_dir),
        batch_size=args.batch_size,
        seq_len=128,
        n_features=64,
        seed=args.seed
    )
    
    # Get dimensions from data
    sample_batch = next(iter(train_loader))
    n_services = sample_batch['metrics'].shape[1]
    n_features = sample_batch['metrics'].shape[3]
    print(f"Found {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)} cases")  # type: ignore
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")  # type: ignore
    print(f"\nData shape: n_services={n_services}, n_features={n_features}")
    
    # Compute PCMCI weights
    causal_cache = None
    if args.use_pcmci:
        cache_path = output_dir / "causal_cache.pkl"
        if cache_path.exists():
            print(f"\nLoading cached causal weights from {cache_path}...")
            with open(cache_path, 'rb') as f:
                causal_cache = pickle.load(f)
            print(f"Loaded {len(causal_cache)} cached weights")
        else:
            print("\nComputing PCMCI causal weights (this may take a while)...")
            # Need to get all cases for PCMCI computation
            all_cases = discover_cases(str(data_dir))
            causal_cache = precompute_causal_weights(
                all_cases, 
                services=services,
                cache_path=str(cache_path),
                max_lag=5
            )
            print(f"Saved causal cache to {cache_path}")
    
    # Create model
    model = RCAModel(
        n_services=n_services,
        n_features=n_features,
        tcn_hidden=128,
        embed_dim=256,
        num_heads=8,
        num_attn_layers=3,
        dropout=0.2
    )
    
    params = model.count_parameters()
    print(f"\nModel parameters: {params['trainable']:,} trainable")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=1e-4
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
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Load best model
    best_model_path = model_dir / "rca_model_best.pt"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_metrics = trainer.evaluate(test_loader, causal_cache)
    
    print(f"\nTest Results:")
    print(f"  AC@1: {test_metrics['AC@1']:.1%}")
    print(f"  AC@3: {test_metrics['AC@3']:.1%}")
    print(f"  AC@5: {test_metrics['AC@5']:.1%}")
    print(f"  MRR:  {test_metrics['MRR']:.3f}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'best_val_acc': trainer.best_val_acc,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
