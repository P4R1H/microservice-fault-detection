#!/usr/bin/env python3
"""
Quick smoke test for the full RCA pipeline.

Tests all components together:
- Encoders (metrics, logs, traces)
- Multimodal fusion
- RCA model
- Loss computation
- Top-k prediction
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.encoders.metrics_encoder import TCNEncoder
from src.encoders.traces_encoder import GCNEncoder
from src.encoders.logs_encoder import create_logs_encoder
from src.fusion import create_multimodal_fusion
from src.models import RCAModel

print("=" * 70)
print("FULL PIPELINE SMOKE TEST")
print("=" * 70)

try:
    # 1. Create encoders
    print("\n1. Creating encoders...")
    metrics_encoder = TCNEncoder(in_channels=7, embedding_dim=256)
    traces_encoder = GCNEncoder(in_channels=8, embedding_dim=128)
    logs_encoder = create_logs_encoder(embedding_dim=256, use_dummy=True)
    print("✓ Encoders created successfully")

    # 2. Create fusion
    print("\n2. Creating multimodal fusion...")
    fusion = create_multimodal_fusion(
        metrics_encoder=metrics_encoder,
        logs_encoder=logs_encoder,
        traces_encoder=traces_encoder,
        fusion_strategy='intermediate',
        fusion_dim=512,
        num_heads=8,
        use_modality_dropout=True
    )
    print("✓ Fusion created successfully")
    print(f"  - Strategy: intermediate (cross-modal attention)")
    print(f"  - Fusion dim: 512")
    print(f"  - Attention heads: 8")

    # 3. Create RCA model
    print("\n3. Creating RCA model...")
    rca_model = RCAModel(
        fusion_model=fusion,
        num_services=50,
        fusion_dim=512,
        hidden_dim=256,
        use_service_embedding=True
    )
    print("✓ RCA model created successfully")
    print(f"  - Number of services: 50")
    print(f"  - Using service embeddings: True")

    # Count parameters
    n_params = sum(p.numel() for p in rca_model.parameters())
    print(f"  - Total parameters: {n_params:,} (~{n_params/1e6:.1f}M)")

    # 4. Test forward pass with dummy data
    print("\n4. Testing forward pass...")
    batch_size = 2
    metrics = torch.randn(batch_size, 12, 7)  # (batch, seq_len, features)
    logs = None  # Skip logs for now (will use dummy encoder)
    traces = (
        torch.randn(10, 8),  # node features: (num_nodes, node_dim)
        torch.randint(0, 10, (2, 20))  # edge index: (2, num_edges)
    )

    rca_model.eval()
    with torch.no_grad():
        output = rca_model(metrics=metrics, logs=logs, traces=traces)

    print(f"✓ Forward pass successful!")
    print(f"  - Input shapes:")
    print(f"    - Metrics: {metrics.shape}")
    print(f"    - Traces: nodes={traces[0].shape}, edges={traces[1].shape}")
    print(f"  - Output shapes:")
    print(f"    - Logits: {output['logits'].shape}")
    print(f"    - Probs: {output['probs'].shape}")
    print(f"    - Ranking: {output['ranking'].shape}")

    # 5. Test top-k prediction
    print("\n5. Testing top-k prediction...")
    top_k_services, top_k_scores = rca_model.predict_top_k(output['logits'], k=5)
    print(f"✓ Top-k prediction successful!")
    print(f"  - Top-5 services shape: {top_k_services.shape}")
    print(f"  - Top-5 scores shape: {top_k_scores.shape}")
    print(f"  - Sample predictions:")
    for i in range(min(2, batch_size)):
        print(f"    Batch {i}: services={top_k_services[i].tolist()[:3]}... scores={top_k_scores[i].tolist()[:3]}")

    # 6. Test loss computation
    print("\n6. Testing loss computation...")
    target_services = torch.randint(0, 50, (batch_size,))
    loss_ce = rca_model.compute_loss(
        output['logits'],
        target_services,
        loss_type='cross_entropy'
    )
    loss_rank = rca_model.compute_loss(
        output['logits'],
        target_services,
        loss_type='ranking_loss'
    )
    print(f"✓ Loss computation successful!")
    print(f"  - Cross-entropy loss: {loss_ce.item():.4f}")
    print(f"  - Ranking loss: {loss_rank.item():.4f}")
    print(f"  - Target services: {target_services.tolist()}")

    # 7. Test evaluation metrics
    print("\n7. Testing evaluation metrics...")
    from src.models import compute_accuracy_at_k, compute_mrr

    # Create dummy predictions
    predictions = output['ranking']  # (batch, num_services)
    targets = target_services

    ac_1 = compute_accuracy_at_k(predictions, targets, k=1)
    ac_3 = compute_accuracy_at_k(predictions, targets, k=3)
    ac_5 = compute_accuracy_at_k(predictions, targets, k=5)
    mrr = compute_mrr(predictions, targets)

    print(f"✓ Evaluation metrics computed!")
    print(f"  - AC@1: {ac_1:.4f}")
    print(f"  - AC@3: {ac_3:.4f}")
    print(f"  - AC@5: {ac_5:.4f}")
    print(f"  - MRR: {mrr:.4f}")

    # 8. Test with service embeddings disabled
    print("\n8. Testing without service embeddings...")
    rca_model_no_emb = RCAModel(
        fusion_model=fusion,
        num_services=50,
        fusion_dim=512,
        use_service_embedding=False
    )
    rca_model_no_emb.eval()
    with torch.no_grad():
        output_no_emb = rca_model_no_emb(metrics=metrics, logs=logs, traces=traces)
    print(f"✓ Model without service embeddings works!")
    print(f"  - Logits shape: {output_no_emb['logits'].shape}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nFull pipeline is working correctly:")
    print("  ✓ Encoders (TCN, GCN, Dummy Logs)")
    print("  ✓ Multimodal fusion with cross-attention")
    print("  ✓ RCA model with service ranking")
    print("  ✓ Service embeddings (with/without)")
    print("  ✓ Loss computation (CE + Ranking)")
    print("  ✓ Top-k prediction")
    print("  ✓ Evaluation metrics (AC@k, MRR)")
    print("\nReady for training and evaluation!")
    print("=" * 70)

    sys.exit(0)

except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
