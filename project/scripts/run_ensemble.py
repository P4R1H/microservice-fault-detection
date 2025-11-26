"""
Ensemble evaluation script for Multimodal RCA models.

Loads multiple trained models and performs soft voting (probability averaging)
to improve prediction accuracy.

Usage:
    python scripts/run_ensemble.py --model-paths outputs/models/v4_s42.pt outputs/models/v4_s123.pt ...
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.multimodal_data import create_multimodal_loaders
from src.models.rca_v4_multimodal import create_multimodal_model
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
        'ac@1': np.mean(ranks == 1),
        'ac@3': np.mean(ranks <= 3),
        'ac@5': np.mean(ranks <= 5),
        'mrr': np.mean(1.0 / ranks),
        'avg_rank': np.mean(ranks)
    }


def load_models(model_paths: list, device: torch.device) -> list:
    """Load all models from checkpoints."""
    models = []
    
    for path in model_paths:
        print(f"Loading model from {path}...")
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        
        args = checkpoint['args']
        services = checkpoint['services']
        n_services = len(services)
        
        # Create model with same architecture
        model = create_multimodal_model(
            n_services=n_services,
            n_metric_features=args.get('n_metric_features', 64),
            n_log_features=args.get('n_log_features', 32),
            n_trace_features=args.get('n_trace_features', 32),
            hidden_dim=args.get('hidden_dim', 32),
            embed_dim=args.get('embed_dim', 128),
            dropout=args.get('dropout', 0.35),
            logs_encoder_type=args.get('logs_encoder', 'tfidf')
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append({
            'model': model,
            'services': services,
            'args': args,
            'path': path,
            'val_metrics': checkpoint.get('val_metrics', {})
        })
        
        val_ac1 = checkpoint.get('val_metrics', {}).get('ac@1', 0) * 100
        print(f"  Loaded: {len(services)} services, Val AC@1: {val_ac1:.1f}%")
    
    return models


@torch.no_grad()
def ensemble_evaluate(models: list, loader, device, causal_computer, ensemble_type='soft'):
    """
    Evaluate ensemble on a dataset.
    
    Args:
        models: List of model dicts
        loader: Data loader
        device: torch device
        causal_computer: CausalWeightComputer
        ensemble_type: 'soft' (average probabilities) or 'hard' (majority vote)
    """
    all_targets = []
    all_ensemble_rankings = []
    
    # Also track individual model predictions for comparison
    individual_predictions = [[] for _ in models]
    
    for batch in tqdm(loader, desc='Evaluating ensemble'):
        metrics = batch['metrics'].to(device)
        logs = batch['logs'].to(device) if batch['logs'] is not None else None
        traces = batch['traces'].to(device) if batch['traces'] is not None else None
        targets = batch['target'].to(device)
        
        causal_weights = causal_computer.get_batch_weights(
            batch['case_id'],
            metrics.shape[1],
            device
        )
        
        # Collect predictions from all models
        all_probs = []
        for i, m in enumerate(models):
            model = m['model']
            outputs = model(metrics, logs, traces, causal_weights)
            probs = F.softmax(outputs['logits'], dim=-1)
            all_probs.append(probs)
            individual_predictions[i].append(outputs['ranking'].cpu())
        
        # Ensemble: average probabilities (soft voting)
        if ensemble_type == 'soft':
            ensemble_probs = torch.stack(all_probs).mean(dim=0)
        else:  # hard voting not implemented yet
            ensemble_probs = torch.stack(all_probs).mean(dim=0)
        
        # Get ranking from ensemble probabilities
        ensemble_ranking = torch.argsort(ensemble_probs, dim=-1, descending=True)
        
        all_ensemble_rankings.append(ensemble_ranking.cpu())
        all_targets.append(targets.cpu())
    
    # Compute ensemble metrics
    all_ensemble_rankings = torch.cat(all_ensemble_rankings)
    all_targets = torch.cat(all_targets)
    ensemble_metrics = compute_metrics(all_ensemble_rankings, all_targets)
    
    # Compute individual model metrics
    individual_metrics = []
    for i, preds in enumerate(individual_predictions):
        preds = torch.cat(preds)
        individual_metrics.append(compute_metrics(preds, all_targets))
    
    return ensemble_metrics, individual_metrics


def main():
    parser = argparse.ArgumentParser(description='Ensemble evaluation for Multimodal RCA')
    
    parser.add_argument('--model-paths', type=str, nargs='+', required=True,
                        help='Paths to trained model checkpoints')
    parser.add_argument('--data-root', type=str, default='data/RCAEval')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--n-metric-features', type=int, default=64)
    parser.add_argument('--n-log-features', type=int, default=32)
    parser.add_argument('--n-trace-features', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data split (use same as training for fair comparison)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--causal-cache', type=str, default='outputs/causal_cache_multimodal.pkl')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print("Multimodal RCA Ensemble Evaluation")
    print(f"{'='*60}")
    print(f"Models: {len(args.model_paths)}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load models
    models = load_models(args.model_paths, device)
    
    # Get services from first model (should be same for all)
    services = models[0]['services']
    n_services = len(services)
    print(f"\nServices ({n_services}): {services[:5]}...")
    
    # Create data loader with same seed to get same split
    print("\nLoading test data...")
    _, _, test_loader, _ = create_multimodal_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_metric_features=args.n_metric_features,
        n_log_features=args.n_log_features,
        n_trace_features=args.n_trace_features,
        seed=args.seed,
        require_multimodal=True
    )
    test_size = len(test_loader.dataset)  # type: ignore
    print(f"Test samples: {test_size}")
    
    # Initialize causal weight computer
    causal_computer = CausalWeightComputer(
        cache_path=args.causal_cache,
        services=services
    )
    
    # Evaluate ensemble
    print("\nEvaluating...")
    ensemble_metrics, individual_metrics = ensemble_evaluate(
        models, test_loader, device, causal_computer  # type: ignore
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    
    print("\nIndividual Model Performance:")
    print("-" * 50)
    for i, (m, metrics) in enumerate(zip(models, individual_metrics)):
        seed = m['args'].get('seed', 'unknown')
        print(f"  Model {i+1} (seed={seed}): AC@1={metrics['ac@1']*100:.1f}%, "
              f"AC@3={metrics['ac@3']*100:.1f}%, MRR={metrics['mrr']:.3f}")
    
    avg_ac1 = np.mean([m['ac@1'] for m in individual_metrics])
    print(f"\n  Individual Average: AC@1={avg_ac1*100:.1f}%")
    
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS (Soft Voting)")
    print(f"{'='*60}")
    print(f"  AC@1:     {ensemble_metrics['ac@1']*100:.1f}%")
    print(f"  AC@3:     {ensemble_metrics['ac@3']*100:.1f}%")
    print(f"  AC@5:     {ensemble_metrics['ac@5']*100:.1f}%")
    print(f"  MRR:      {ensemble_metrics['mrr']:.3f}")
    print(f"  Avg Rank: {ensemble_metrics['avg_rank']:.2f}")
    
    improvement = (ensemble_metrics['ac@1'] - avg_ac1) * 100
    print(f"\n  Improvement over avg: {improvement:+.1f}%")
    
    # Check if target reached
    target_ac1 = 0.70
    if ensemble_metrics['ac@1'] >= target_ac1:
        print(f"\n✓ TARGET REACHED: AC@1 >= {target_ac1*100:.0f}%")
    else:
        gap = (target_ac1 - ensemble_metrics['ac@1']) * 100
        print(f"\n✗ Target not reached. Gap: {gap:.1f}%")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
