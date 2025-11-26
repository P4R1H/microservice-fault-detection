"""
Training script for Multimodal RCA Model (V4/V4.1/V4.2).

Versions:
- V4: Original TCN encoders for all modalities
- V4.1: TF-IDF logs encoder (fixes placeholder)
- V4.2: GCN traces encoder (coming soon)

Features:
- Multimodal data loading (metrics + logs + traces)
- PCMCI causal weight computation
- Configurable encoder types
- Cosine annealing with warm restarts
- Gradient clipping
- Early stopping
- Comprehensive evaluation metrics
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.multimodal_data import (
    create_multimodal_loaders,
    discover_multimodal_cases,
    get_all_services_multimodal
)
from src.models.rca_v4_multimodal import MultimodalRCAModel, MultimodalLoss, create_multimodal_model
from src.causal.pcmci import CausalWeightComputer


def compute_metrics(ranking: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute AC@k and MRR metrics."""
    batch_size = ranking.shape[0]
    
    # Find rank of true label
    ranks = []
    for i in range(batch_size):
        rank = (ranking[i] == targets[i]).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            ranks.append(rank[0].item() + 1)  # 1-indexed
        else:
            ranks.append(ranking.shape[1])  # Worst case
    
    ranks = np.array(ranks)
    
    return {
        'ac@1': np.mean(ranks == 1),
        'ac@3': np.mean(ranks <= 3),
        'ac@5': np.mean(ranks <= 5),
        'mrr': np.mean(1.0 / ranks),
        'avg_rank': np.mean(ranks)
    }


def train_epoch(model, loader, criterion, optimizer, device, causal_computer, 
                clip_grad=1.0, accumulate_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_rankings = []
    all_targets = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_idx, batch in enumerate(pbar):
        metrics = batch['metrics'].to(device)
        logs = batch['logs'].to(device) if batch['logs'] is not None else None
        traces = batch['traces'].to(device) if batch['traces'] is not None else None
        targets = batch['target'].to(device)
        
        # Get causal weights
        causal_weights = causal_computer.get_batch_weights(
            batch['case_id'], 
            metrics.shape[1],  # n_services
            device
        )
        
        # Forward
        outputs = model(metrics, logs, traces, causal_weights)
        losses = criterion(outputs['logits'], targets)
        loss = losses['loss'] / accumulate_steps
        
        # Backward
        loss.backward()
        
        if (batch_idx + 1) % accumulate_steps == 0:
            # Gradient clipping
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += losses['loss'].item()
        all_rankings.append(outputs['ranking'].detach().cpu())
        all_targets.append(targets.detach().cpu())
        
        pbar.set_postfix({'loss': f"{losses['loss'].item():.4f}"})
    
    # Final optimizer step if needed
    if len(loader) % accumulate_steps != 0:
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()
    
    all_rankings = torch.cat(all_rankings)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_rankings, all_targets)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, causal_computer):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_rankings = []
    all_targets = []
    
    for batch in tqdm(loader, desc='Evaluating', leave=False):
        metrics = batch['metrics'].to(device)
        logs = batch['logs'].to(device) if batch['logs'] is not None else None
        traces = batch['traces'].to(device) if batch['traces'] is not None else None
        targets = batch['target'].to(device)
        
        causal_weights = causal_computer.get_batch_weights(
            batch['case_id'], 
            metrics.shape[1],
            device
        )
        
        outputs = model(metrics, logs, traces, causal_weights)
        losses = criterion(outputs['logits'], targets)
        
        total_loss += losses['loss'].item()
        all_rankings.append(outputs['ranking'].cpu())
        all_targets.append(targets.cpu())
    
    all_rankings = torch.cat(all_rankings)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_rankings, all_targets)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics


def train(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"Multimodal RCA Training (V4)")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device)
    
    # Create data loaders
    print("Loading multimodal data...")
    train_loader, val_loader, test_loader, services = create_multimodal_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_metric_features=args.n_metric_features,
        n_log_features=args.n_log_features,
        n_trace_features=args.n_trace_features,
        seed=args.seed,
        require_multimodal=True  # Only use RE2 data with logs/traces
    )
    
    n_services = len(services)
    print(f"Services: {n_services}")
    # Dataset sizes
    train_size = len(list(train_loader.dataset))  # type: ignore
    val_size = len(list(val_loader.dataset))  # type: ignore  
    test_size = len(list(test_loader.dataset))  # type: ignore
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create model
    print("\nCreating model...")
    print(f"  Logs encoder: {args.logs_encoder}")
    
    model = create_multimodal_model(
        n_services=n_services,
        n_metric_features=args.n_metric_features,
        n_log_features=args.n_log_features,
        n_trace_features=args.n_trace_features,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        logs_encoder_type=args.logs_encoder
    ).to(device)
    
    # Set causal weight manually (not in factory)
    model.causal_weight = args.causal_weight
    
    params = model.count_parameters()
    print(f"Model parameters: {params['total']:,} ({params['trainable']:,} trainable)")
    for name, count in params.items():
        if name not in ['total', 'trainable']:
            print(f"  {name}: {count:,}")
    
    # Create causal weight computer
    print("\nInitializing causal weight computer...")
    causal_computer = CausalWeightComputer(
        cache_path=args.causal_cache,
        services=services
    )
    
    # Loss and optimizer
    criterion = MultimodalLoss(
        smoothing=args.label_smoothing,
        rank_weight=args.rank_weight,
        margin=args.margin
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t0,
        T_mult=2,
        eta_min=args.lr * 0.01
    )
    
    # Training loop
    best_val_ac1 = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': [], 'test': []}
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            causal_computer, clip_grad=args.clip_grad
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, causal_computer)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} ({epoch_time:.1f}s) | "
              f"Train: Loss={train_metrics['loss']:.4f}, AC@1={train_metrics['ac@1']*100:.1f}% | "
              f"Val: Loss={val_metrics['loss']:.4f}, AC@1={val_metrics['ac@1']*100:.1f}%, MRR={val_metrics['mrr']:.3f}")
        
        # Check for improvement
        if val_metrics['ac@1'] > best_val_ac1:
            best_val_ac1 = val_metrics['ac@1']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'services': services,
                'args': vars(args)
            }, args.save_path)
            print(f"  → Saved best model (AC@1: {best_val_ac1*100:.1f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break
    
    # Load best model and evaluate on test
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, causal_computer)
    history['test'].append(test_metrics)
    
    print(f"\nTest Results:")
    print(f"  AC@1: {test_metrics['ac@1']*100:.1f}%")
    print(f"  AC@3: {test_metrics['ac@3']*100:.1f}%")
    print(f"  AC@5: {test_metrics['ac@5']*100:.1f}%")
    print(f"  MRR:  {test_metrics['mrr']:.3f}")
    print(f"  Avg Rank: {test_metrics['avg_rank']:.2f}")
    
    # Save history
    history_path = args.save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(history), f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation AC@1: {best_val_ac1*100:.1f}% (epoch {best_epoch})")
    print(f"Model saved to: {args.save_path}")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal RCA Model V4')
    
    # Data
    parser.add_argument('--data-root', type=str, default='data/RCAEval')
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--n-metric-features', type=int, default=64)
    parser.add_argument('--n-log-features', type=int, default=32)
    parser.add_argument('--n-trace-features', type=int, default=32)
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-attn-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--causal-weight', type=float, default=0.3)
    parser.add_argument('--logs-encoder', type=str, default='tfidf',
                        choices=['tcn', 'tfidf', 'gemini'],
                        help='Logs encoder type: tcn (V4), tfidf (V4.1), gemini (V4.3)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--t0', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    
    # Loss
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--rank-weight', type=float, default=0.3)
    parser.add_argument('--margin', type=float, default=0.5)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-path', type=str, default='outputs/models/multimodal_v4.pt')
    parser.add_argument('--causal-cache', type=str, default='outputs/causal_cache_multimodal.pkl')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    train(args)


if __name__ == '__main__':
    main()
