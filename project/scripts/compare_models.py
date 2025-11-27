"""
Compare multimodal V4 models - Speed AND Accuracy benchmarks.

Compares:
- Baseline model (PCMCI only)
- LLM Prior model (PCMCI + LLM causal prior)
- Ensemble inference (4 models)

Usage:
    python scripts/compare_models.py --mode baseline
    python scripts/compare_models.py --mode llm-prior
    python scripts/compare_models.py --mode ensemble
    python scripts/compare_models.py --mode all

Outputs results to outputs/model_comparison.json
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
import time
import numpy as np
from src.models.rca_v4_multimodal import MultimodalRCAModel, create_multimodal_model
from src.data.multimodal_data import create_multimodal_loaders
from src.causal.pcmci import CausalWeightComputer


# ============================================================================
# Accuracy Evaluation
# ============================================================================

def evaluate_model_accuracy(model, test_loader, device, causal_computer=None, services=None):
    """
    Evaluate model accuracy on test set.
    
    Returns:
        Dict with AC@1, AC@3, AC@5, MRR metrics
    """
    model.eval()
    
    all_ranks = []
    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0
    total = 0
    
    n_services = len(services) if services else 10
    
    with torch.no_grad():
        for batch in test_loader:
            metrics = batch['metrics'].to(device)
            logs = batch['logs'].to(device)
            traces = batch['traces'].to(device)
            targets = batch['target'].to(device)
            case_ids = batch.get('case_id', ['case_001'] * metrics.shape[0])
            
            # Get causal weights if available
            if causal_computer is not None:
                causal_weights = causal_computer.get_batch_weights(
                    case_ids, metrics.shape[1], device
                )
                outputs = model(metrics, logs, traces, causal_weights)
            else:
                outputs = model(metrics, logs, traces)
            
            probs = outputs['probs']
            
            for i in range(probs.shape[0]):
                pred_ranking = torch.argsort(probs[i], descending=True)
                gt = targets[i].item()
                
                rank = (pred_ranking == gt).nonzero(as_tuple=True)[0][0].item() + 1
                all_ranks.append(rank)
                
                if rank == 1:
                    correct_at_1 += 1
                if rank <= 3:
                    correct_at_3 += 1
                if rank <= 5:
                    correct_at_5 += 1
                total += 1
    
    return {
        'ac@1': correct_at_1 / total * 100,
        'ac@3': correct_at_3 / total * 100,
        'ac@5': correct_at_5 / total * 100,
        'mrr': float(np.mean([1.0 / r for r in all_ranks])),
        'avg_rank': float(np.mean(all_ranks)),
        'total_samples': total
    }


def evaluate_ensemble_accuracy(model_paths, test_loader, device, services):
    """Evaluate ensemble accuracy using soft voting."""
    models = []
    
    for path in model_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False, map_location=device)
            args = checkpoint['args']
            
            model = create_multimodal_model(
                n_services=len(services),
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
            models.append(model)
    
    if not models:
        return {'error': 'No models found'}
    
    all_ranks = []
    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            metrics = batch['metrics'].to(device)
            logs = batch['logs'].to(device)
            traces = batch['traces'].to(device)
            targets = batch['target'].to(device)
            
            # Ensemble prediction (soft voting)
            all_probs = []
            for model in models:
                outputs = model(metrics, logs, traces)
                probs = F.softmax(outputs['logits'], dim=-1)
                all_probs.append(probs)
            
            ensemble_probs = torch.stack(all_probs).mean(dim=0)
            
            for i in range(ensemble_probs.shape[0]):
                pred_ranking = torch.argsort(ensemble_probs[i], descending=True)
                gt = targets[i].item()
                
                rank = (pred_ranking == gt).nonzero(as_tuple=True)[0][0].item() + 1
                all_ranks.append(rank)
                
                if rank == 1:
                    correct_at_1 += 1
                if rank <= 3:
                    correct_at_3 += 1
                if rank <= 5:
                    correct_at_5 += 1
                total += 1
    
    return {
        'ac@1': correct_at_1 / total * 100,
        'ac@3': correct_at_3 / total * 100,
        'ac@5': correct_at_5 / total * 100,
        'mrr': float(np.mean([1.0 / r for r in all_ranks])),
        'avg_rank': float(np.mean(all_ranks)),
        'num_models': len(models),
        'total_samples': total
    }


# ============================================================================
# Speed Measurement
# ============================================================================

def measure_model_speed(model, dataloader, device, causal_computer=None, 
                       num_warmup=5, num_runs=100):
    """Measure average inference time per sample."""
    model.eval()
    
    batch = next(iter(dataloader))
    metrics = batch['metrics'].to(device)
    logs = batch['logs'].to(device)
    traces = batch['traces'].to(device)
    case_ids = batch.get('case_id', ['case_001'])
    
    batch_size = metrics.shape[0]
    n_services = metrics.shape[1]
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if causal_computer is not None:
                causal_weights = causal_computer.get_batch_weights(case_ids, n_services, device)
                _ = model(metrics, logs, traces, causal_weights)
            else:
                _ = model(metrics, logs, traces)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            if causal_computer is not None:
                causal_weights = causal_computer.get_batch_weights(case_ids, n_services, device)
                _ = model(metrics, logs, traces, causal_weights)
            else:
                _ = model(metrics, logs, traces)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) / batch_size)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def measure_ensemble_speed(model_paths, dataloader, device, num_warmup=3, num_runs=50):
    """Measure ensemble inference time."""
    models = []
    for path in model_paths:
        if os.path.exists(path):
            checkpoint = torch.load(path, weights_only=False, map_location=device)
            args = checkpoint['args']
            services = checkpoint['services']
            
            model = create_multimodal_model(
                n_services=len(services),
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
            models.append(model)
    
    if not models:
        return {'error': 'No models found'}
    
    batch = next(iter(dataloader))
    metrics = batch['metrics'].to(device)
    logs = batch['logs'].to(device)
    traces = batch['traces'].to(device)
    batch_size = metrics.shape[0]
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            all_probs = []
            for model in models:
                outputs = model(metrics, logs, traces)
                probs = F.softmax(outputs['logits'], dim=-1)
                all_probs.append(probs)
            _ = torch.stack(all_probs).mean(dim=0)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            all_probs = []
            for model in models:
                outputs = model(metrics, logs, traces)
                probs = F.softmax(outputs['logits'], dim=-1)
                all_probs.append(probs)
            _ = torch.stack(all_probs).mean(dim=0)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) / batch_size)
    
    return {
        'num_models': len(models),
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'per_model_ms': np.mean(times) * 1000 / len(models)
    }


# ============================================================================
# Benchmark Runners
# ============================================================================

def load_trained_model(model_path, device, services):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    args = checkpoint['args']
    
    model = create_multimodal_model(
        n_services=len(services),
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
    return model


def run_baseline_comparison(device, test_loader, speed_loader, services):
    """Compare baseline models (PCMCI only)."""
    print("\n" + "=" * 70)
    print("BASELINE MODELS (PCMCI only)")
    print("=" * 70)
    
    model_paths = [
        ('v4_s42', 'outputs/models/v4_s42.pt'),
        ('v4_s123', 'outputs/models/v4_s123.pt'),
        ('v4_s456', 'outputs/models/v4_s456.pt'),
        ('v4_s789', 'outputs/models/v4_s789.pt'),
    ]
    
    causal_computer = CausalWeightComputer(
        cache_path='outputs/causal_cache_multimodal.pkl',
        services=services
    )
    
    results = {}
    
    for name, path in model_paths:
        if not os.path.exists(path):
            print(f"  ⚠️  {name}: Not found")
            continue
            
        print(f"\n  📊 {name}:")
        model = load_trained_model(path, device, services)
        
        # Accuracy (use batch loader for efficiency)
        accuracy = evaluate_model_accuracy(model, test_loader, device, causal_computer, services)
        print(f"    AC@1: {accuracy['ac@1']:.1f}%  |  AC@3: {accuracy['ac@3']:.1f}%  |  MRR: {accuracy['mrr']:.3f}")
        
        # Speed (use single-sample loader for realistic timing)
        speed = measure_model_speed(model, speed_loader, device, causal_computer)
        print(f"    Speed: {speed['mean_ms']:.2f}ms ± {speed['std_ms']:.2f}ms")
        
        results[name] = {
            'accuracy': accuracy,
            'speed': speed
        }
    
    return results


def run_llm_prior_comparison(device, test_loader, speed_loader, services):
    """Compare LLM prior models."""
    print("\n" + "=" * 70)
    print("LLM PRIOR MODELS (PCMCI + LLM)")
    print("=" * 70)
    
    model_paths = [
        ('v4_llm_s42', 'outputs/models/v4_llm_s42.pt'),
        ('v4_llm_s123', 'outputs/models/v4_llm_s123.pt'),
        ('v4_llm_s456', 'outputs/models/v4_llm_s456.pt'),
        ('v4_llm_s789', 'outputs/models/v4_llm_s789.pt'),
    ]
    
    causal_computer = CausalWeightComputer(
        cache_path='outputs/causal_cache_multimodal.pkl',
        services=services
    )
    
    results = {}
    
    for name, path in model_paths:
        if not os.path.exists(path):
            print(f"  ⚠️  {name}: Not found")
            continue
            
        print(f"\n  📊 {name}:")
        model = load_trained_model(path, device, services)
        
        # Accuracy (use batch loader for efficiency)
        accuracy = evaluate_model_accuracy(model, test_loader, device, causal_computer, services)
        print(f"    AC@1: {accuracy['ac@1']:.1f}%  |  AC@3: {accuracy['ac@3']:.1f}%  |  MRR: {accuracy['mrr']:.3f}")
        
        # Speed (use single-sample loader for realistic timing)
        speed = measure_model_speed(model, speed_loader, device, causal_computer)
        print(f"    Speed: {speed['mean_ms']:.2f}ms ± {speed['std_ms']:.2f}ms")
        
        results[name] = {
            'accuracy': accuracy,
            'speed': speed
        }
    
    return results


def run_ensemble_comparison(device, test_loader, speed_loader, services):
    """Compare ensemble configurations."""
    print("\n" + "=" * 70)
    print("ENSEMBLE MODELS (4 models each)")
    print("=" * 70)
    
    results = {}
    
    # Baseline ensemble
    baseline_paths = [
        'outputs/models/v4_s42.pt',
        'outputs/models/v4_s123.pt',
        'outputs/models/v4_s456.pt',
        'outputs/models/v4_s789.pt'
    ]
    
    existing_baseline = [p for p in baseline_paths if os.path.exists(p)]
    if existing_baseline:
        print(f"\n  📊 Baseline Ensemble ({len(existing_baseline)} models):")
        accuracy = evaluate_ensemble_accuracy(baseline_paths, test_loader, device, services)
        speed = measure_ensemble_speed(baseline_paths, speed_loader, device)
        
        print(f"    AC@1: {accuracy['ac@1']:.1f}%  |  AC@3: {accuracy['ac@3']:.1f}%  |  MRR: {accuracy['mrr']:.3f}")
        print(f"    Speed: {speed['mean_ms']:.2f}ms (total) | {speed['per_model_ms']:.2f}ms (per model)")
        
        results['baseline_ensemble'] = {
            'accuracy': accuracy,
            'speed': speed
        }
    
    # LLM prior ensemble
    llm_paths = [
        'outputs/models/v4_llm_s42.pt',
        'outputs/models/v4_llm_s123.pt',
        'outputs/models/v4_llm_s456.pt',
        'outputs/models/v4_llm_s789.pt'
    ]
    
    existing_llm = [p for p in llm_paths if os.path.exists(p)]
    if existing_llm:
        print(f"\n  📊 LLM Prior Ensemble ({len(existing_llm)} models):")
        accuracy = evaluate_ensemble_accuracy(llm_paths, test_loader, device, services)
        speed = measure_ensemble_speed(llm_paths, speed_loader, device)
        
        print(f"    AC@1: {accuracy['ac@1']:.1f}%  |  AC@3: {accuracy['ac@3']:.1f}%  |  MRR: {accuracy['mrr']:.3f}")
        print(f"    Speed: {speed['mean_ms']:.2f}ms (total) | {speed['per_model_ms']:.2f}ms (per model)")
        
        results['llm_prior_ensemble'] = {
            'accuracy': accuracy,
            'speed': speed
        }
    
    return results


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # SOTA reference
    sota_acc = 63.1  # RUN (AAAI 2024)
    sota_time = 892.0  # ms
    
    print(f"\n{'Model':<25} {'AC@1':<10} {'AC@3':<10} {'MRR':<10} {'Speed (ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    # Print SOTA reference
    print(f"{'RUN (SOTA, AAAI 2024)':<25} {sota_acc:<10.1f} {'N/A':<10} {'N/A':<10} {sota_time:<12.1f} {'1.0x':<10}")
    print("-" * 80)
    
    # Collect and sort results
    rows = []
    
    for category, results in all_results.items():
        if category == 'metadata':
            continue
        if isinstance(results, dict):
            for name, data in results.items():
                if isinstance(data, dict) and 'accuracy' in data:
                    acc = data['accuracy']
                    spd = data['speed']
                    speedup = sota_time / spd['mean_ms']
                    rows.append({
                        'name': name,
                        'ac1': acc['ac@1'],
                        'ac3': acc['ac@3'],
                        'mrr': acc['mrr'],
                        'speed': spd['mean_ms'],
                        'speedup': speedup
                    })
    
    # Sort by AC@1 descending
    rows.sort(key=lambda x: x['ac1'], reverse=True)
    
    for row in rows:
        print(f"{row['name']:<25} {row['ac1']:<10.1f} {row['ac3']:<10.1f} {row['mrr']:<10.3f} {row['speed']:<12.2f} {row['speedup']:.0f}x")
    
    # Print best results
    if rows:
        best_acc = max(rows, key=lambda x: x['ac1'])
        best_speed = min(rows, key=lambda x: x['speed'])
        
        print("\n" + "-" * 80)
        print(f"🏆 Best Accuracy: {best_acc['name']} ({best_acc['ac1']:.1f}% AC@1)")
        print(f"⚡ Fastest:       {best_speed['name']} ({best_speed['speed']:.2f}ms, {best_speed['speedup']:.0f}x vs SOTA)")


def main():
    parser = argparse.ArgumentParser(description='Compare model speed and accuracy')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['baseline', 'llm-prior', 'ensemble', 'all'],
                       help='What to compare')
    parser.add_argument('--output', type=str, default='outputs/model_comparison.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("MODEL COMPARISON - Speed & Accuracy")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data - two loaders: batch_size=8 for accuracy, batch_size=1 for speed
    data_root = "data/RCAEval"
    _, _, test_loader, services = create_multimodal_loaders(
        data_root=data_root,
        batch_size=8,  # Larger batch for accurate evaluation
        seed=42
    )
    _, _, speed_loader, _ = create_multimodal_loaders(
        data_root=data_root,
        batch_size=1,  # Single sample for realistic speed measurement
        seed=42
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")  # type: ignore 
    
    all_results = {
        'metadata': {
            'description': 'Model comparison - speed and accuracy',
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
            'date': datetime.now().isoformat()
        }
    }
    
    # Run requested comparisons
    if args.mode in ['baseline', 'all']:
        all_results['baseline'] = run_baseline_comparison(device, test_loader, speed_loader, services)
    
    if args.mode in ['llm-prior', 'all']:
        all_results['llm_prior'] = run_llm_prior_comparison(device, test_loader, speed_loader, services)
    
    if args.mode in ['ensemble', 'all']:
        all_results['ensemble'] = run_ensemble_comparison(device, test_loader, speed_loader, services)
    
    # Print summary
    print_comparison_table(all_results)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    return all_results


if __name__ == "__main__":
    main()
