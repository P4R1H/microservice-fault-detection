"""Test script for V4.1 model with TF-IDF logs encoder."""
import sys
import os

# Add project src to path properly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
from src.models.rca_v4_multimodal import create_multimodal_model, MultimodalLoss

def test_model():
    print("=" * 60)
    print("Testing V4.1 Model with TF-IDF Logs Encoder")
    print("=" * 60)
    
    # Configuration
    batch_size = 4
    n_services = 15
    seq_len = 60
    n_metric_feat = 64
    n_log_feat = 32
    n_trace_feat = 32
    
    # Create model with TF-IDF logs encoder (V4.1)
    print("\n1. Creating model with logs_encoder_type='tfidf'...")
    model = create_multimodal_model(
        n_services=n_services,
        n_metric_features=n_metric_feat,
        n_log_features=n_log_feat,
        n_trace_features=n_trace_feat,
        logs_encoder_type='tfidf',
        traces_encoder_type='tcn'
    )
    
    print(f"   Logs encoder type: {model.logs_encoder_type}")
    print(f"   Traces encoder type: {model.traces_encoder_type}")
    
    # Count parameters
    print("\n2. Model parameters by component:")
    params = model.count_parameters()
    for name, count in params.items():
        print(f"   {name}: {count:,}")
    
    # Create test inputs
    print("\n3. Creating test tensors...")
    metrics = torch.randn(batch_size, n_services, seq_len, n_metric_feat)
    logs = torch.randn(batch_size, n_services, seq_len, n_log_feat)
    traces = torch.randn(batch_size, n_services, seq_len, n_trace_feat)
    causal = torch.rand(batch_size, n_services, n_services)
    targets = torch.randint(0, n_services, (batch_size,))
    
    print(f"   metrics: {metrics.shape}")
    print(f"   logs: {logs.shape}")
    print(f"   traces: {traces.shape}")
    print(f"   causal: {causal.shape}")
    print(f"   targets: {targets.shape}")
    
    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        out = model(metrics, logs, traces, causal)
    
    print(f"   logits: {out['logits'].shape}")
    print(f"   probs: {out['probs'].shape}")
    print(f"   ranking: {out['ranking'].shape}")
    print(f"   Top prediction per sample: {out['ranking'][:, 0].tolist()}")
    
    # Test loss
    print("\n5. Testing loss computation...")
    loss_fn = MultimodalLoss()
    losses = loss_fn(out['logits'], targets)
    for name, val in losses.items():
        print(f"   {name}: {val.item():.4f}")
    
    # Compare with TCN logs encoder
    print("\n6. Creating model with logs_encoder_type='tcn' for comparison...")
    model_tcn = create_multimodal_model(
        n_services=n_services,
        n_metric_features=n_metric_feat,
        n_log_features=n_log_feat,
        n_trace_features=n_trace_feat,
        logs_encoder_type='tcn',
        traces_encoder_type='tcn'
    )
    params_tcn = model_tcn.count_parameters()
    
    print(f"\n   Parameter comparison:")
    print(f"   TF-IDF logs encoder: {params['logs_encoder']:,}")
    print(f"   TCN logs encoder:    {params_tcn['logs_encoder']:,}")
    print(f"   Difference: {params['logs_encoder'] - params_tcn['logs_encoder']:,}")
    
    print("\n" + "=" * 60)
    print("✓ V4.1 Model Test PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    test_model()
