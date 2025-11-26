"""
Inference script with Gemini explanations.

Loads a trained model, runs inference on test cases, and generates
natural language explanations for the predictions.

Usage:
    python scripts/inference_with_explanations.py --model-path outputs/models/multimodal_v4_seed42.pt
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multimodal_data import create_multimodal_loaders
from src.models.rca_v4_multimodal import MultimodalRCAModel, create_multimodal_model
from src.llm.explainer import GeminiExplainer


def load_model(model_path: str, device: torch.device, n_services: Optional[int] = None) -> Tuple[MultimodalRCAModel, Dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    
    # Override n_services if provided
    actual_n_services = n_services if n_services is not None else config.get('n_services', 10)
    
    model = create_multimodal_model(
        n_services=actual_n_services,
        n_metric_features=config.get('n_metric_features', 64),
        n_log_features=config.get('n_log_features', 32),
        n_trace_features=config.get('n_trace_features', 32),
        hidden_dim=config.get('hidden_dim', 32),
        embed_dim=config.get('embed_dim', 128),
        dropout=config.get('dropout', 0.35),
        logs_encoder_type=config.get('logs_encoder_type', 'tfidf')
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def extract_context_from_batch(
    batch: Dict[str, torch.Tensor],
    service_names: List[str],
    idx: int = 0
) -> Dict:
    """
    Extract human-readable context from a data batch.
    
    Args:
        batch: Data batch from dataloader
        service_names: List of service names
        idx: Index within batch to extract
    
    Returns:
        Context dict with metrics, logs, traces summaries
    """
    context = {
        'services': service_names,
        'metrics': {},
        'logs': [],
        'traces': {}
    }
    
    # Extract metrics
    if 'metrics' in batch:
        metrics = batch['metrics'][idx].cpu().numpy()  # (n_services, seq_len, features)
        for i, service in enumerate(service_names):
            service_metrics = metrics[i]  # (seq_len, features)
            
            # Compute simple anomaly indicators
            mean_vals = service_metrics.mean(axis=0)
            std_vals = service_metrics.std(axis=0)
            max_vals = service_metrics.max(axis=0)
            
            # Simple anomaly score based on variance
            anomaly_score = float(std_vals.mean())
            
            if anomaly_score > 0.3:  # Threshold for reporting
                context['metrics'][service] = {
                    'anomaly_score': anomaly_score,
                    'details': f'High variance in metrics (std={std_vals.mean():.2f})'
                }
    
    # Extract traces
    if 'traces' in batch:
        traces = batch['traces'][idx].cpu().numpy()  # (n_services, seq_len, features)
        for i, service in enumerate(service_names):
            service_traces = traces[i]
            
            # Estimate latency and error rate from features
            avg_latency = float(service_traces[:, 0].mean() * 1000)  # Scale to ms
            error_rate = float(max(0, min(1, service_traces[:, 1].mean())))
            
            if avg_latency > 100 or error_rate > 0.1:
                context['traces'][service] = {
                    'avg_latency': avg_latency,
                    'error_rate': error_rate
                }
    
    return context


def run_inference_with_explanations(
    model: MultimodalRCAModel,
    dataloader: torch.utils.data.DataLoader,
    explainer: GeminiExplainer,
    service_names: List[str],
    device: torch.device,
    max_samples: int = 5,
    explain_all: bool = False
) -> List[Dict]:
    """
    Run inference and generate explanations.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        explainer: GeminiExplainer instance
        service_names: List of service names
        device: Torch device
        max_samples: Max samples to explain (if not explain_all)
        explain_all: Whether to explain all samples
    
    Returns:
        List of result dicts with predictions and explanations
    """
    results = []
    sample_count = 0
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            metrics = batch['metrics'].to(device)
            logs = batch.get('logs')
            if logs is not None:
                logs = logs.to(device)
            traces = batch.get('traces')
            if traces is not None:
                traces = traces.to(device)
            targets = batch['target'].to(device)
            causal = batch.get('causal_scores')
            if causal is not None:
                causal = causal.to(device)
            
            # Forward pass
            outputs = model(metrics, logs, traces, causal)
            probs = outputs['probs']
            
            batch_size = metrics.shape[0]
            
            for i in range(batch_size):
                if not explain_all and sample_count >= max_samples:
                    break
                
                # Get prediction
                sample_probs = probs[i].cpu().numpy()
                pred_idx = int(np.argmax(sample_probs))
                confidence = float(sample_probs[pred_idx])
                
                # Get ground truth
                gt_idx = int(targets[i].cpu().item())
                is_correct = pred_idx == gt_idx
                
                # Build ranking
                sorted_indices = np.argsort(sample_probs)[::-1]
                ranking = [
                    (service_names[idx], float(sample_probs[idx]))
                    for idx in sorted_indices[:5]
                ]
                
                # Extract context
                context = extract_context_from_batch(batch, service_names, i)
                
                # Build prediction dict
                prediction = {
                    'root_cause': service_names[pred_idx],
                    'confidence': confidence,
                    'ranking': ranking,
                    'ground_truth': service_names[gt_idx],
                    'correct': is_correct
                }
                
                # Generate explanation
                print(f"\n{'='*60}")
                print(f"Sample {sample_count + 1}")
                print(f"Predicted: {prediction['root_cause']} ({confidence:.1%})")
                print(f"Actual: {prediction['ground_truth']}")
                print(f"Correct: {'✅' if is_correct else '❌'}")
                print(f"{'='*60}")
                
                explanation_result = explainer.explain_with_comparison(
                    prediction, context, prediction['ground_truth']
                )
                
                print(f"\n{explanation_result['explanation']}")
                if 'comparison' in explanation_result:
                    print(f"\n{explanation_result['comparison']}")
                
                # Store result
                results.append({
                    'sample_id': sample_count,
                    'prediction': prediction,
                    'context_summary': {
                        'services_with_metric_anomalies': list(context['metrics'].keys()),
                        'services_with_trace_issues': list(context['traces'].keys())
                    },
                    'explanation': explanation_result['explanation'],
                    'comparison': explanation_result.get('comparison', '')
                })
                
                sample_count += 1
            
            if not explain_all and sample_count >= max_samples:
                break
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inference with Gemini explanations')
    parser.add_argument('--model-path', type=str, 
                        default='outputs/models/multimodal_v4.pt',
                        help='Path to trained model')
    parser.add_argument('--data-path', type=str,
                        default='data/RCAEval',
                        help='Path to RCAEval data')
    parser.add_argument('--max-samples', type=int, default=3,
                        help='Maximum samples to explain')
    parser.add_argument('--explain-all', action='store_true',
                        help='Explain all test samples')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    
    # Get data loaders and service names
    _, _, test_loader, service_names = create_multimodal_loaders(
        data_root=args.data_path,
        batch_size=8,
        seed=args.seed,
        require_multimodal=True
    )
    
    n_services = len(service_names)
    print(f"Services: {n_services}")
    print(f"Test samples: {len(list(test_loader.dataset))}")  # type: ignore
    
    # Load model - use discovered n_services
    print(f"\nLoading model from {args.model_path}...")
    model, config = load_model(args.model_path, device, n_services=n_services)
    print(f"Model loaded: {model.count_parameters()['total']:,} parameters")
    
    # Create explainer
    print("\nInitializing Gemini explainer...")
    explainer = GeminiExplainer()
    
    # Run inference with explanations
    print(f"\nRunning inference on {args.max_samples if not args.explain_all else 'all'} samples...")
    results = run_inference_with_explanations(
        model=model,
        dataloader=test_loader,
        explainer=explainer,
        service_names=service_names,
        device=device,
        max_samples=args.max_samples,
        explain_all=args.explain_all
    )
    
    # Summary
    correct = sum(1 for r in results if r['prediction']['correct'])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {correct}/{total} correct ({correct/total:.1%})")
    print(f"{'='*60}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = f"outputs/explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'timestamp': datetime.now().isoformat(),
            'accuracy': correct / total if total > 0 else 0,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
