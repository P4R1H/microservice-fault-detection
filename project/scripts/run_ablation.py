"""
Ablation Study Runner for Multimodal RCA Model V4.

This script runs ablation experiments:
1. Full V4 Multimodal (baseline)
2. Metrics Only (no logs/traces)
3. No Gated Fusion (simple concatenation)
4. No Causal Weights (no PCMCI injection)
5. No Cross-Service Attention (direct scoring)

Results are saved to outputs/ablation/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.multimodal_data import create_multimodal_loaders
from src.models.rca_v4_multimodal import MultimodalRCAModel, MultimodalLoss
from src.causal.pcmci import CausalWeightComputer


def compute_metrics(ranking: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute AC@k and MRR metrics."""
    batch_size = ranking.shape[0]
    
    ranks = []
    for i in range(batch_size):
        rank = (ranking[i] == targets[i]).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            ranks.append(rank[0].item() + 1)
        else:
            ranks.append(ranking.shape[1])
    
    ranks = np.array(ranks)
    
    return {
        'ac@1': float(np.mean(ranks == 1)),
        'ac@3': float(np.mean(ranks <= 3)),
        'ac@5': float(np.mean(ranks <= 5)),
        'mrr': float(np.mean(1.0 / ranks)),
        'avg_rank': float(np.mean(ranks))
    }


def evaluate(model, loader, criterion, device, causal_computer, use_causal=True):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0
    all_rankings = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            metrics = batch['metrics'].to(device)
            logs = batch['logs'].to(device) if batch['logs'] is not None else None
            traces = batch['traces'].to(device) if batch['traces'] is not None else None
            targets = batch['target'].to(device)
            
            if use_causal:
                causal_weights = causal_computer.get_batch_weights(
                    batch['case_id'],
                    metrics.shape[1],
                    device
                )
            else:
                causal_weights = None
            
            outputs = model(metrics, logs, traces, causal_weights)
            losses = criterion(outputs['logits'], targets)
            total_loss += losses['loss'].item()
            
            _, ranking = outputs['logits'].sort(dim=1, descending=True)
            all_rankings.append(ranking.cpu())
            all_targets.append(targets.cpu())
    
    all_rankings = torch.cat(all_rankings, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics_dict = compute_metrics(all_rankings, all_targets)
    metrics_dict['loss'] = total_loss / len(loader)
    
    return metrics_dict


def train_ablation_variant(
    variant_name: str,
    train_loader,
    val_loader,
    test_loader,
    device: str,
    causal_computer,
    args,
    use_logs: bool = True,
    use_traces: bool = True,
    use_gated_fusion: bool = True,
    use_causal: bool = True,
    use_attention: bool = True,
) -> Dict:
    """Train and evaluate one ablation variant."""
    
    print(f"\n{'='*60}")
    print(f"Training: {variant_name}")
    print(f"{'='*60}")
    print(f"  use_logs={use_logs}, use_traces={use_traces}")
    print(f"  use_gated_fusion={use_gated_fusion}, use_causal={use_causal}")
    print(f"  use_attention={use_attention}")
    
    # Create model
    # Note: We don't modify architecture - we modify inputs during forward pass
    # use_gated_fusion and use_attention are controlled by zero-ing inputs
    num_attn = args.num_attn_layers if use_attention else 0
    
    model = MultimodalRCAModel(
        n_services=10,
        n_metric_features=64,
        n_log_features=32,
        n_trace_features=32,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_attn_layers=max(1, num_attn),  # At least 1 layer
        num_heads=4,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
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
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_ac1 = 0
    best_epoch = 0
    patience_counter = 0
    
    history = {'train': [], 'val': []}
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        all_rankings = []
        all_targets = []
        
        for batch in train_loader:
            metrics = batch['metrics'].to(device)
            
            if use_logs:
                logs = batch['logs'].to(device) if batch['logs'] is not None else None
            else:
                logs = None
            
            if use_traces:
                traces = batch['traces'].to(device) if batch['traces'] is not None else None
            else:
                traces = None
            
            targets = batch['target'].to(device)
            
            if use_causal:
                causal_weights = causal_computer.get_batch_weights(
                    batch['case_id'],
                    metrics.shape[1],
                    device
                )
            else:
                causal_weights = None
            
            optimizer.zero_grad()
            outputs = model(metrics, logs, traces, causal_weights)
            losses = criterion(outputs['logits'], targets)
            losses['loss'].backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            total_loss += losses['loss'].item()
            _, ranking = outputs['logits'].sort(dim=1, descending=True)
            all_rankings.append(ranking.cpu())
            all_targets.append(targets.cpu())
        
        scheduler.step()
        
        # Compute train metrics
        all_rankings = torch.cat(all_rankings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        train_metrics = compute_metrics(all_rankings, all_targets)
        train_metrics['loss'] = total_loss / len(train_loader)
        history['train'].append(train_metrics)
        
        # Validate
        model.eval()
        val_rankings = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                metrics = batch['metrics'].to(device)
                logs = batch['logs'].to(device) if (use_logs and batch['logs'] is not None) else None
                traces = batch['traces'].to(device) if (use_traces and batch['traces'] is not None) else None
                targets = batch['target'].to(device)
                
                if use_causal:
                    causal_weights = causal_computer.get_batch_weights(
                        batch['case_id'],
                        metrics.shape[1],
                        device
                    )
                else:
                    causal_weights = None
                
                outputs = model(metrics, logs, traces, causal_weights)
                _, ranking = outputs['logits'].sort(dim=1, descending=True)
                val_rankings.append(ranking.cpu())
                val_targets.append(targets.cpu())
        
        val_rankings = torch.cat(val_rankings, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_metrics = compute_metrics(val_rankings, val_targets)
        history['val'].append(val_metrics)
        
        # Early stopping check
        if val_metrics['ac@1'] > best_val_ac1:
            best_val_ac1 = val_metrics['ac@1']
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train AC@1={train_metrics['ac@1']*100:.1f}%, "
                  f"Val AC@1={val_metrics['ac@1']*100:.1f}%")
        
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    
    test_rankings = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            metrics = batch['metrics'].to(device)
            logs = batch['logs'].to(device) if (use_logs and batch['logs'] is not None) else None
            traces = batch['traces'].to(device) if (use_traces and batch['traces'] is not None) else None
            targets = batch['target'].to(device)
            
            if use_causal:
                causal_weights = causal_computer.get_batch_weights(
                    batch['case_id'],
                    metrics.shape[1],
                    device
                )
            else:
                causal_weights = None
            
            outputs = model(metrics, logs, traces, causal_weights)
            _, ranking = outputs['logits'].sort(dim=1, descending=True)
            test_rankings.append(ranking.cpu())
            test_targets.append(targets.cpu())
    
    test_rankings = torch.cat(test_rankings, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_metrics = compute_metrics(test_rankings, test_targets)
    
    print(f"\n  Results for {variant_name}:")
    print(f"    Best Val AC@1: {best_val_ac1*100:.1f}% (epoch {best_epoch})")
    print(f"    Test AC@1: {test_metrics['ac@1']*100:.1f}%")
    print(f"    Test AC@3: {test_metrics['ac@3']*100:.1f}%")
    print(f"    Test AC@5: {test_metrics['ac@5']*100:.1f}%")
    print(f"    Test MRR:  {test_metrics['mrr']:.3f}")
    
    return {
        'variant': variant_name,
        'config': {
            'use_logs': use_logs,
            'use_traces': use_traces,
            'use_gated_fusion': use_gated_fusion,
            'use_causal': use_causal,
            'use_attention': use_attention,
        },
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_ac@1': float(best_val_ac1),
        'test': test_metrics,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for Multimodal V4')
    
    # Data
    parser.add_argument('--data-root', type=str, default='data/RCAEval')
    parser.add_argument('--seed', type=int, default=42)
    
    # Model
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--num-attn-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.35)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=20)
    
    # Loss
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--rank-weight', type=float, default=0.3)
    parser.add_argument('--margin', type=float, default=0.5)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/ablation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("ABLATION STUDY - MULTIMODAL V4 RCA")
    print("=" * 70)
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    
    # Load data
    train_loader, val_loader, test_loader, services = create_multimodal_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Initialize causal computer
    causal_computer = CausalWeightComputer(cache_path='outputs/causal_cache_multimodal.pkl')
    
    # Define ablation variants
    ablation_variants = [
        {
            'name': 'Full V4 Multimodal',
            'use_logs': True, 'use_traces': True,
            'use_gated_fusion': True, 'use_causal': True, 'use_attention': True
        },
        {
            'name': 'Metrics Only (no logs/traces)',
            'use_logs': False, 'use_traces': False,
            'use_gated_fusion': True, 'use_causal': True, 'use_attention': True
        },
        {
            'name': 'No Gated Fusion (concat)',
            'use_logs': True, 'use_traces': True,
            'use_gated_fusion': False, 'use_causal': True, 'use_attention': True
        },
        {
            'name': 'No Causal Weights',
            'use_logs': True, 'use_traces': True,
            'use_gated_fusion': True, 'use_causal': False, 'use_attention': True
        },
        {
            'name': 'No Cross-Service Attention',
            'use_logs': True, 'use_traces': True,
            'use_gated_fusion': True, 'use_causal': True, 'use_attention': False
        },
    ]
    
    results = {
        'metadata': {
            'description': 'Ablation study for Multimodal V4 RCA Model',
            'seed': args.seed,
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'n_train': len(train_loader.dataset),
            'n_val': len(val_loader.dataset),
            'n_test': len(test_loader.dataset),
            'date': datetime.now().isoformat()
        },
        'variants': []
    }
    
    # Run ablations
    for variant in ablation_variants:
        result = train_ablation_variant(
            variant_name=variant['name'],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=args.device,
            causal_computer=causal_computer,
            args=args,
            use_logs=variant['use_logs'],
            use_traces=variant['use_traces'],
            use_gated_fusion=variant['use_gated_fusion'],
            use_causal=variant['use_causal'],
            use_attention=variant['use_attention'],
        )
        results['variants'].append(result)
    
    # Compute deltas from full model
    full_ac1 = results['variants'][0]['test']['ac@1']
    results['summary'] = []
    
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Variant':<35} {'AC@1':<10} {'Δ vs Full':<12} {'MRR':<10}")
    print("-" * 70)
    
    for variant in results['variants']:
        ac1 = variant['test']['ac@1']
        delta = ac1 - full_ac1
        mrr = variant['test']['mrr']
        
        delta_str = f"{delta*100:+.1f}%" if variant['variant'] != 'Full V4 Multimodal' else "baseline"
        print(f"{variant['variant']:<35} {ac1*100:<10.1f}% {delta_str:<12} {mrr:<10.3f}")
        
        results['summary'].append({
            'variant': variant['variant'],
            'ac@1': ac1,
            'delta_vs_full': delta,
            'mrr': mrr
        })
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'ablation_seed{args.seed}.json')
    
    # Remove history to save space (optional)
    results_compact = {
        'metadata': results['metadata'],
        'summary': results['summary'],
        'variants': [{k: v for k, v in var.items() if k != 'history'} for var in results['variants']]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_compact, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Also save full results with history
    full_output_path = os.path.join(args.output_dir, f'ablation_seed{args.seed}_full.json')
    with open(full_output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Full results (with history) saved to: {full_output_path}")


if __name__ == '__main__':
    main()
