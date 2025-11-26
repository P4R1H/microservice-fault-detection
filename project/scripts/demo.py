#!/usr/bin/env python3
"""
=============================================================================
MULTIMODAL ROOT CAUSE ANALYSIS - DEMO SCRIPT
=============================================================================

This script provides a unified interface to demonstrate all capabilities of
the Multimodal RCA system. Designed for project presentations.

Usage:
    python scripts/demo.py                    # Interactive menu
    python scripts/demo.py --mode evaluate    # Evaluate best model
    python scripts/demo.py --mode inference   # Run inference with explanations  
    python scripts/demo.py --mode speed       # Benchmark inference speed
    python scripts/demo.py --mode all         # Run all demos
    python scripts/demo.py --mode quick       # Quick demo (3 samples)

Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
Date: November 2025
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

# Best model checkpoint (V4.1 TF-IDF, seed 123 = 85.2% AC@1)
BEST_MODEL_PATH = "outputs/models/multimodal_v41_tfidf_s123.pt"
# Alternative: V4.0 seed 123 (81.5% AC@1)
ALT_MODEL_PATH = "outputs/models/multimodal_v4_seed123.pt"

DATA_ROOT = "data/RCAEval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Helper Functions
# ============================================================================

def print_header(title: str, char: str = "="):
    """Print a styled header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.1f}%"


def get_model_path() -> str:
    """Get the best available model path."""
    best_path = PROJECT_ROOT / BEST_MODEL_PATH
    alt_path = PROJECT_ROOT / ALT_MODEL_PATH
    
    if best_path.exists():
        return str(best_path)
    elif alt_path.exists():
        return str(alt_path)
    else:
        # Find any v4 model
        models_dir = PROJECT_ROOT / "outputs" / "models"
        for f in models_dir.glob("multimodal_v4*.pt"):
            return str(f)
    
    raise FileNotFoundError("No trained model found! Please train a model first.")


# ============================================================================
# Demo Modules
# ============================================================================

def demo_evaluate(verbose: bool = True) -> Dict:
    """
    Evaluate the best model on test set.
    
    Returns:
        Dict with evaluation metrics
    """
    from src.data.multimodal_data import create_multimodal_loaders
    from src.models.rca_v4_multimodal import create_multimodal_model
    from src.causal.pcmci import CausalWeightComputer
    
    print_header("MODEL EVALUATION")
    
    model_path = get_model_path()
    print(f"📁 Model: {Path(model_path).name}")
    print(f"🖥️  Device: {DEVICE}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    
    print(f"📊 Model config: embed_dim={config.get('embed_dim', 128)}, "
          f"hidden_dim={config.get('hidden_dim', 32)}")
    
    # Count parameters
    n_params = sum(p.numel() for p in checkpoint['model_state_dict'].values() 
                   if isinstance(p, torch.Tensor) and p.dtype in [torch.float32, torch.float16])
    print(f"📐 Parameters: {n_params:,}")
    
    # Load data
    _, _, test_loader, services = create_multimodal_loaders(
        data_root=str(PROJECT_ROOT / DATA_ROOT),
        batch_size=8,
        seed=123
    )
    
    print(f"🧪 Test samples: {len(test_loader.dataset)}")  # type: ignore
    print(f"🔧 Services: {len(services)}")
    
    # Create model
    model = create_multimodal_model(
        n_services=len(services),
        n_metric_features=config.get('n_metric_features', 64),
        n_log_features=config.get('n_log_features', 32),
        n_trace_features=config.get('n_trace_features', 32),
        hidden_dim=config.get('hidden_dim', 32),
        embed_dim=config.get('embed_dim', 128),
        dropout=config.get('dropout', 0.35),
        logs_encoder_type=config.get('logs_encoder_type', 'tfidf')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Evaluate
    print_section("Running Evaluation")
    
    causal_computer = CausalWeightComputer(
        cache_path=str(PROJECT_ROOT / 'outputs/causal_cache_multimodal.pkl')
    )
    
    all_ranks = []
    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            metrics = batch['metrics'].to(DEVICE)
            logs = batch['logs'].to(DEVICE) if batch['logs'] is not None else None
            traces = batch['traces'].to(DEVICE) if batch['traces'] is not None else None
            targets = batch['target'].to(DEVICE)
            
            causal_weights = causal_computer.get_batch_weights(
                batch['case_id'],
                metrics.shape[1],
                DEVICE
            )
            
            outputs = model(metrics, logs, traces, causal_weights)
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
    
    # Compute metrics
    results = {
        'ac@1': correct_at_1 / total,
        'ac@3': correct_at_3 / total,
        'ac@5': correct_at_5 / total,
        'mrr': float(np.mean([1.0 / r for r in all_ranks])),
        'avg_rank': float(np.mean(all_ranks)),
        'total_samples': total
    }
    
    # Print results
    print_section("Results")
    print(f"  ✅ AC@1: {format_percentage(results['ac@1'])} (correct on first prediction)")
    print(f"  ✅ AC@3: {format_percentage(results['ac@3'])} (correct in top 3)")
    print(f"  ✅ AC@5: {format_percentage(results['ac@5'])} (correct in top 5)")
    print(f"  📈 MRR:  {results['mrr']:.3f}")
    print(f"  📊 Average Rank: {results['avg_rank']:.2f}")
    
    # Comparison with SOTA
    print_section("Comparison with State-of-the-Art")
    print(f"  📊 RUN (AAAI 2024 SOTA): 63.1% AC@1")
    print(f"  📊 Our Model:            {format_percentage(results['ac@1'])} AC@1")
    improvement = (results['ac@1'] - 0.631) / 0.631 * 100
    print(f"  🚀 Improvement:          {improvement:+.1f}%")
    
    return results


def demo_inference(num_samples: int = 5, use_llm: bool = True) -> List[Dict]:
    """
    Run inference on test samples with optional LLM explanations.
    
    Args:
        num_samples: Number of samples to process
        use_llm: Whether to generate LLM explanations
    
    Returns:
        List of inference results
    """
    from src.data.multimodal_data import create_multimodal_loaders
    from src.models.rca_v4_multimodal import create_multimodal_model
    
    print_header("INFERENCE DEMO")
    
    model_path = get_model_path()
    print(f"📁 Model: {Path(model_path).name}")
    print(f"🔢 Samples: {num_samples}")
    print(f"🤖 LLM Explanations: {'Yes' if use_llm else 'No'}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Load data
    _, _, test_loader, services = create_multimodal_loaders(
        data_root=str(PROJECT_ROOT / DATA_ROOT),
        batch_size=1,
        seed=123
    )
    
    model = create_multimodal_model(
        n_services=len(services),
        n_metric_features=config.get('n_metric_features', 64),
        n_log_features=config.get('n_log_features', 32),
        n_trace_features=config.get('n_trace_features', 32),
        hidden_dim=config.get('hidden_dim', 32),
        embed_dim=config.get('embed_dim', 128),
        dropout=config.get('dropout', 0.35),
        logs_encoder_type=config.get('logs_encoder_type', 'tfidf')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Initialize explainer if needed
    explainer = None
    if use_llm:
        try:
            from src.llm.explainer import GeminiExplainer
            explainer = GeminiExplainer()
            print("✅ Gemini Explainer initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize Gemini: {e}")
            print("   Continuing without LLM explanations...")
            use_llm = False
    
    results = []
    sample_idx = 0
    
    print_section("Processing Samples")
    
    with torch.no_grad():
        for batch in test_loader:
            if sample_idx >= num_samples:
                break
            
            metrics = batch['metrics'].to(DEVICE)
            logs = batch['logs'].to(DEVICE) if batch['logs'] is not None else None
            traces = batch['traces'].to(DEVICE) if batch['traces'] is not None else None
            targets = batch['target'].to(DEVICE)
            
            outputs = model(metrics, logs, traces, None)
            probs = outputs['probs'][0].cpu().numpy()
            
            pred_idx = int(np.argmax(probs))
            gt_idx = int(targets[0].cpu().item())
            confidence = float(probs[pred_idx])
            
            # Get top 5 ranking
            sorted_idx = np.argsort(probs)[::-1][:5]
            ranking = [(services[i], float(probs[i])) for i in sorted_idx]
            
            result = {
                'sample': sample_idx + 1,
                'prediction': services[pred_idx],
                'confidence': confidence,
                'ground_truth': services[gt_idx],
                'correct': pred_idx == gt_idx,
                'ranking': ranking
            }
            
            # Print result
            status = "✅" if result['correct'] else "❌"
            print(f"\n  Sample {sample_idx + 1}: {status}")
            print(f"    Predicted:    {result['prediction']} ({confidence*100:.1f}%)")
            print(f"    Ground Truth: {result['ground_truth']}")
            print(f"    Top 3: {', '.join([f'{s}({p*100:.0f}%)' for s, p in ranking[:3]])}")
            
            # Generate explanation
            if use_llm and explainer is not None:
                try:
                    prediction_dict = {
                        'root_cause': result['prediction'],
                        'confidence': confidence,
                        'ranking': ranking
                    }
                    explanation = explainer.explain(
                        prediction=prediction_dict,
                        context={}
                    )
                    result['explanation'] = explanation
                    print(f"\n    💡 LLM Explanation:")
                    for line in explanation.split('\n')[:6]:
                        if line.strip():
                            print(f"       {line}")
                except Exception as e:
                    print(f"    ⚠️  Explanation failed: {e}")
            
            results.append(result)
            sample_idx += 1
    
    # Summary
    correct = sum(1 for r in results if r['correct'])
    print_section("Summary")
    print(f"  Correct: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    
    return results


def demo_speed() -> Dict:
    """
    Benchmark inference speed.
    
    Returns:
        Dict with speed metrics
    """
    from src.data.multimodal_data import create_multimodal_loaders
    from src.models.rca_v4_multimodal import create_multimodal_model
    
    print_header("SPEED BENCHMARK")
    
    model_path = get_model_path()
    print(f"📁 Model: {Path(model_path).name}")
    print(f"🖥️  Device: {DEVICE}")
    
    if DEVICE == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Load one batch for benchmarking
    _, _, test_loader, services = create_multimodal_loaders(
        data_root=str(PROJECT_ROOT / DATA_ROOT),
        batch_size=1,
        seed=123
    )
    
    model = create_multimodal_model(
        n_services=len(services),
        n_metric_features=config.get('n_metric_features', 64),
        n_log_features=config.get('n_log_features', 32),
        n_trace_features=config.get('n_trace_features', 32),
        hidden_dim=config.get('hidden_dim', 32),
        embed_dim=config.get('embed_dim', 128),
        dropout=config.get('dropout', 0.35),
        logs_encoder_type=config.get('logs_encoder_type', 'tfidf')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Get sample batch
    batch = next(iter(test_loader))
    metrics = batch['metrics'].to(DEVICE)
    logs = batch['logs'].to(DEVICE) if batch['logs'] is not None else None
    traces = batch['traces'].to(DEVICE) if batch['traces'] is not None else None
    
    # Warmup
    print_section("Warmup (10 iterations)")
    with torch.no_grad():
        for _ in range(10):
            _ = model(metrics, logs, traces, None)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    num_runs = 100
    print_section(f"Benchmarking ({num_runs} iterations)")
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(metrics, logs, traces, None)
            
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
    
    # Results
    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p99_ms': float(np.percentile(times, 99)),
    }
    
    print_section("Results")
    print(f"  ⚡ Mean:   {results['mean_ms']:.2f} ms")
    print(f"  📊 Std:    {results['std_ms']:.2f} ms")
    print(f"  🔽 Min:    {results['min_ms']:.2f} ms")
    print(f"  🔼 Max:    {results['max_ms']:.2f} ms")
    print(f"  📈 P50:    {results['p50_ms']:.2f} ms")
    print(f"  📈 P99:    {results['p99_ms']:.2f} ms")
    
    # Throughput
    throughput = 1000 / results['mean_ms']
    results['throughput'] = throughput
    print(f"\n  🚀 Throughput: {throughput:.0f} samples/second")
    
    # SOTA comparison
    sota_time = 892  # RUN method
    speedup = sota_time / results['mean_ms']
    results['speedup_vs_sota'] = speedup
    
    print_section("SOTA Comparison")
    print(f"  📊 RUN (SOTA): 892 ms")
    print(f"  📊 Our Model:  {results['mean_ms']:.2f} ms")
    print(f"  🚀 Speedup:    {speedup:.0f}× faster")
    
    return results


def demo_architecture():
    """Display model architecture information."""
    from src.models.rca_v4_multimodal import create_multimodal_model
    
    print_header("MODEL ARCHITECTURE")
    
    model_path = get_model_path()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    print("🏗️  MULTIMODAL RCA V4 ARCHITECTURE")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Input Modalities                                          │")
    print("  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │")
    print("  │  │  Metrics    │ │  Logs       │ │  Traces     │          │")
    print("  │  │  (64 feat)  │ │  (32 feat)  │ │  (32 feat)  │          │")
    print("  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │")
    print("  │         │               │               │                  │")
    print("  │         ▼               ▼               ▼                  │")
    print("  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │")
    print("  │  │ TCN Encoder │ │ TF-IDF+TCN  │ │ TCN Encoder │          │")
    print("  │  │ (depthwise) │ │  Encoder    │ │ (depthwise) │          │")
    print("  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │")
    print("  │         └───────────────┼───────────────┘                  │")
    print("  │                         ▼                                  │")
    print("  │              ┌─────────────────────┐                       │")
    print("  │              │    Gated Fusion     │                       │")
    print("  │              │  g_m·M + g_l·L + g_t·T                      │")
    print("  │              └──────────┬──────────┘                       │")
    print("  │                         ▼                                  │")
    print("  │              ┌─────────────────────┐                       │")
    print("  │              │ Cross-Service Attn  │                       │")
    print("  │              │ + PCMCI Causal Bias │                       │")
    print("  │              └──────────┬──────────┘                       │")
    print("  │                         ▼                                  │")
    print("  │              ┌─────────────────────┐                       │")
    print("  │              │  Scoring Head (MLP) │                       │")
    print("  │              └──────────┬──────────┘                       │")
    print("  │                         ▼                                  │")
    print("  │                   Service Ranking                          │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    
    print_section("Configuration")
    print(f"  Embedding Dim:     {config.get('embed_dim', 128)}")
    print(f"  Hidden Dim:        {config.get('hidden_dim', 32)}")
    print(f"  Attention Layers:  2")
    print(f"  Attention Heads:   4")
    print(f"  Dropout:           {config.get('dropout', 0.35)}")
    print(f"  Logs Encoder:      {config.get('logs_encoder_type', 'tfidf')}")
    
    # Count parameters
    n_params = sum(p.numel() for p in checkpoint['model_state_dict'].values() 
                   if isinstance(p, torch.Tensor) and p.dtype in [torch.float32, torch.float16])
    
    print_section("Model Size")
    print(f"  Total Parameters:  {n_params:,}")
    print(f"  Model Size:        ~{n_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    print_section("Key Innovations")
    print("  1. ⚡ Depthwise Separable TCN - 3× fewer parameters")
    print("  2. 🎯 Gated Fusion - learns modality importance per-case")
    print("  3. 🔗 PCMCI Causal Injection - distinguishes cause from effect")
    print("  4. 📝 TF-IDF Logs Encoder - learnable template weights")


def interactive_menu():
    """Show interactive menu for demo selection."""
    while True:
        print_header("MULTIMODAL RCA DEMO", "═")
        print("  Select a demo to run:")
        print()
        print("  [1] 📊 Evaluate Model       - Test accuracy on benchmark")
        print("  [2] 🔍 Inference Demo       - See predictions with explanations")
        print("  [3] ⚡ Speed Benchmark      - Measure inference speed")
        print("  [4] 🏗️  Architecture         - View model structure")
        print("  [5] 🎯 Quick Demo           - Fast evaluation (3 samples)")
        print("  [6] 🚀 Full Demo            - Run all demos")
        print("  [0] ❌ Exit")
        print()
        
        try:
            choice = input("  Enter choice [0-6]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break
        
        if choice == '0':
            print("\nGoodbye! 👋")
            break
        elif choice == '1':
            demo_evaluate()
        elif choice == '2':
            demo_inference(num_samples=5, use_llm=True)
        elif choice == '3':
            demo_speed()
        elif choice == '4':
            demo_architecture()
        elif choice == '5':
            demo_inference(num_samples=3, use_llm=False)
        elif choice == '6':
            demo_architecture()
            demo_evaluate()
            demo_speed()
            demo_inference(num_samples=3, use_llm=True)
        else:
            print("  Invalid choice. Please try again.")
        
        print()
        input("  Press Enter to continue...")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multimodal RCA Demo Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo.py                    # Interactive menu
  python scripts/demo.py --mode evaluate    # Evaluate model
  python scripts/demo.py --mode inference   # Run inference  
  python scripts/demo.py --mode speed       # Speed benchmark
  python scripts/demo.py --mode quick       # Quick demo
  python scripts/demo.py --mode all         # All demos
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['interactive', 'evaluate', 'inference', 'speed', 'architecture', 'quick', 'all'],
        default='interactive',
        help='Demo mode to run'
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=5,
        help='Number of samples for inference demo'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM explanations'
    )
    
    args = parser.parse_args()
    
    print_header("MULTIMODAL ROOT CAUSE ANALYSIS", "═")
    print("  Authors: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan")
    print("  Course:  B.Tech Major Project")
    print("  Date:    November 2025")
    print(f"  Device:  {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    
    try:
        if args.mode == 'interactive':
            interactive_menu()
        elif args.mode == 'evaluate':
            demo_evaluate()
        elif args.mode == 'inference':
            demo_inference(num_samples=args.samples, use_llm=not args.no_llm)
        elif args.mode == 'speed':
            demo_speed()
        elif args.mode == 'architecture':
            demo_architecture()
        elif args.mode == 'quick':
            demo_architecture()
            demo_inference(num_samples=3, use_llm=False)
        elif args.mode == 'all':
            demo_architecture()
            demo_evaluate()
            demo_speed()
            demo_inference(num_samples=5, use_llm=not args.no_llm)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please ensure you have trained models in outputs/models/")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
    
    print_header("DEMO COMPLETE", "═")


if __name__ == '__main__':
    main()
