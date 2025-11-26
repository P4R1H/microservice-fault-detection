"""
Evaluation script for Multimodal RCA Model V4.

This script:
1. Loads trained checkpoints
2. Runs evaluation on test set
3. Outputs comprehensive JSON results with AC@1/AC@3/AC@5/MRR
4. Aggregates results across multiple seeds
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
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
        'avg_rank': float(np.mean(ranks)),
        'ranks': ranks.tolist()
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    data_root: str,
    seed: int,
    device: str = 'cuda',
    embed_dim: int = 128,
    hidden_dim: int = 32
) -> Dict:
    """Evaluate a single checkpoint on test set."""
    
    print(f"\nEvaluating: {checkpoint_path}")
    print(f"  Seed: {seed}, embed_dim: {embed_dim}, hidden_dim: {hidden_dim}")
    
    # Load data with same seed for consistent splits
    _, _, test_loader, services = create_multimodal_loaders(
        data_root=data_root,
        batch_size=8,
        seed=seed
    )
    
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = MultimodalRCAModel(
        n_services=10,
        n_metric_features=64,
        n_log_features=32,
        n_trace_features=32,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_attn_layers=2,
        num_heads=4,
        dropout=0.35
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Initialize causal computer
    causal_computer = CausalWeightComputer(cache_path='outputs/causal_cache_multimodal.pkl')
    
    # Evaluate
    all_rankings = []
    all_targets = []
    all_case_ids = []
    all_gate_values = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='  Evaluating', leave=False):
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
            
            # Get rankings
            _, ranking = outputs['logits'].sort(dim=1, descending=True)
            all_rankings.append(ranking.cpu())
            all_targets.append(targets.cpu())
            all_case_ids.extend(batch['case_id'])
            
            # Get gate values if available
            if 'gate_values' in outputs and outputs['gate_values'] is not None:
                all_gate_values.append(outputs['gate_values'].cpu())
    
    # Compute metrics
    all_rankings = torch.cat(all_rankings, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    results = compute_metrics(all_rankings, all_targets)
    
    # Add gate value statistics
    if all_gate_values:
        gate_values = torch.cat(all_gate_values, dim=0).numpy()
        results['gate_values'] = {
            'metrics_mean': float(np.mean(gate_values[:, 0])),
            'logs_mean': float(np.mean(gate_values[:, 1])),
            'traces_mean': float(np.mean(gate_values[:, 2])),
            'metrics_std': float(np.std(gate_values[:, 0])),
            'logs_std': float(np.std(gate_values[:, 1])),
            'traces_std': float(np.std(gate_values[:, 2])),
        }
    
    results['checkpoint'] = checkpoint_path
    results['seed'] = seed
    results['embed_dim'] = embed_dim
    results['hidden_dim'] = hidden_dim
    results['n_params'] = n_params
    results['n_trainable_params'] = n_trainable
    results['n_test_samples'] = len(test_loader.dataset)
    
    print(f"  Results: AC@1={results['ac@1']*100:.1f}%, AC@3={results['ac@3']*100:.1f}%, "
          f"AC@5={results['ac@5']*100:.1f}%, MRR={results['mrr']:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multimodal RCA V4 checkpoints')
    parser.add_argument('--data-root', type=str, default='data/RCAEval')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/models')
    parser.add_argument('--output', type=str, default='outputs/v4_final_results.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MULTIMODAL V4 EVALUATION")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    
    # Define checkpoints to evaluate
    checkpoints = [
        # Small model (324K params): embed_dim=128, hidden_dim=32
        {'path': 'multimodal_v4.pt', 'seed': 42, 'embed_dim': 128, 'hidden_dim': 32, 'config': 'Small'},
        {'path': 'multimodal_v4_seed123.pt', 'seed': 123, 'embed_dim': 128, 'hidden_dim': 32, 'config': 'Small'},
        {'path': 'multimodal_v4_seed456.pt', 'seed': 456, 'embed_dim': 128, 'hidden_dim': 32, 'config': 'Small'},
        {'path': 'multimodal_v4_seed789.pt', 'seed': 789, 'embed_dim': 128, 'hidden_dim': 32, 'config': 'Small'},
        {'path': 'multimodal_v4_seed2024.pt', 'seed': 2024, 'embed_dim': 128, 'hidden_dim': 32, 'config': 'Small'},
        
        # Large model (722K params): embed_dim=192, hidden_dim=48
        {'path': 'multimodal_v4_bigger.pt', 'seed': 42, 'embed_dim': 192, 'hidden_dim': 48, 'config': 'Large'},
        {'path': 'multimodal_v4_bigger_s123.pt', 'seed': 123, 'embed_dim': 192, 'hidden_dim': 48, 'config': 'Large'},
        {'path': 'multimodal_v4_bigger_s456.pt', 'seed': 456, 'embed_dim': 192, 'hidden_dim': 48, 'config': 'Large'},
    ]
    
    results = {
        'metadata': {
            'description': 'Multimodal V4 RCA Model Evaluation Results',
            'model': 'Depthwise Separable TCN + Gated Fusion + Cross-Service Attention + PCMCI',
            'dataset': 'RCAEval RE2 (OnlineBoutique, SockShop, TrainTicket)',
            'evaluation_date': datetime.now().isoformat(),
            'device': args.device
        },
        'individual_results': [],
        'summary': {}
    }
    
    # Evaluate each checkpoint
    for ckpt_info in checkpoints:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_info['path'])
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {ckpt_info['path']} - not found")
            continue
        
        try:
            result = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                data_root=args.data_root,
                seed=ckpt_info['seed'],
                device=args.device,
                embed_dim=ckpt_info['embed_dim'],
                hidden_dim=ckpt_info['hidden_dim']
            )
            result['config'] = ckpt_info['config']
            results['individual_results'].append(result)
        except Exception as e:
            print(f"\nError evaluating {ckpt_info['path']}: {e}")
            continue
    
    # Compute summary statistics
    if results['individual_results']:
        # By config
        for config in ['Small', 'Large']:
            config_results = [r for r in results['individual_results'] if r['config'] == config]
            if config_results:
                ac1_values = [r['ac@1'] for r in config_results]
                ac3_values = [r['ac@3'] for r in config_results]
                ac5_values = [r['ac@5'] for r in config_results]
                mrr_values = [r['mrr'] for r in config_results]
                
                results['summary'][config] = {
                    'n_seeds': len(config_results),
                    'n_params': config_results[0]['n_params'],
                    'ac@1_mean': float(np.mean(ac1_values)),
                    'ac@1_std': float(np.std(ac1_values)),
                    'ac@1_min': float(np.min(ac1_values)),
                    'ac@1_max': float(np.max(ac1_values)),
                    'ac@3_mean': float(np.mean(ac3_values)),
                    'ac@3_std': float(np.std(ac3_values)),
                    'ac@5_mean': float(np.mean(ac5_values)),
                    'ac@5_std': float(np.std(ac5_values)),
                    'mrr_mean': float(np.mean(mrr_values)),
                    'mrr_std': float(np.std(mrr_values)),
                }
        
        # Overall
        all_ac1 = [r['ac@1'] for r in results['individual_results']]
        all_mrr = [r['mrr'] for r in results['individual_results']]
        
        results['summary']['overall'] = {
            'n_checkpoints': len(results['individual_results']),
            'ac@1_mean': float(np.mean(all_ac1)),
            'ac@1_std': float(np.std(all_ac1)),
            'ac@1_best': float(np.max(all_ac1)),
            'mrr_mean': float(np.mean(all_mrr)),
            'mrr_best': float(np.max(all_mrr)),
        }
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if 'Small' in results['summary']:
        s = results['summary']['Small']
        print(f"\nV4-Small ({s['n_params']:,} params, {s['n_seeds']} seeds):")
        print(f"  AC@1: {s['ac@1_mean']*100:.1f}% ± {s['ac@1_std']*100:.1f}%")
        print(f"  AC@3: {s['ac@3_mean']*100:.1f}% ± {s['ac@3_std']*100:.1f}%")
        print(f"  AC@5: {s['ac@5_mean']*100:.1f}% ± {s['ac@5_std']*100:.1f}%")
        print(f"  MRR:  {s['mrr_mean']:.3f} ± {s['mrr_std']:.3f}")
    
    if 'Large' in results['summary']:
        s = results['summary']['Large']
        print(f"\nV4-Large ({s['n_params']:,} params, {s['n_seeds']} seeds):")
        print(f"  AC@1: {s['ac@1_mean']*100:.1f}% ± {s['ac@1_std']*100:.1f}%")
        print(f"  AC@3: {s['ac@3_mean']*100:.1f}% ± {s['ac@3_std']*100:.1f}%")
        print(f"  AC@5: {s['ac@5_mean']*100:.1f}% ± {s['ac@5_std']*100:.1f}%")
        print(f"  MRR:  {s['mrr_mean']:.3f} ± {s['mrr_std']:.3f}")
    
    if 'overall' in results['summary']:
        o = results['summary']['overall']
        print(f"\nOverall ({o['n_checkpoints']} checkpoints):")
        print(f"  AC@1 Mean: {o['ac@1_mean']*100:.1f}%")
        print(f"  AC@1 Best: {o['ac@1_best']*100:.1f}%")
        print(f"  MRR Mean:  {o['mrr_mean']:.3f}")
    
    print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
