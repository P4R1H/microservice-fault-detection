"""
Test Script: End-to-End LLM Causal Prior Integration

Tests the complete integration of LLM Causal Prior with the
multimodal RCA model, including:
1. Data loading
2. Causal weight generation (PCMCI + LLM)
3. Model forward pass
4. Comparison with PCMCI-only baseline

Usage:
    cd project
    python tests/test_llm_prior_integration.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_integration():
    """Test full integration of LLM prior with model."""
    
    print("=" * 70)
    print("END-TO-END LLM CAUSAL PRIOR INTEGRATION TEST")
    print("=" * 70)
    
    # =========================================================================
    # Test 1: Import all required modules
    # =========================================================================
    print("\n[1/5] Testing imports...")
    
    try:
        from src.data.multimodal_data import (
            create_multimodal_loaders,
            get_all_services_multimodal,
            discover_multimodal_cases
        )
        from src.models.rca_v4_multimodal import create_multimodal_model
        from src.causal.pcmci import CausalWeightComputer
        from src.causal.llm_prior import LLMCausalPrior, CausalWeightManager, get_system_type
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # =========================================================================
    # Test 2: Initialize components
    # =========================================================================
    print("\n[2/5] Initializing components...")
    
    try:
        # Load data
        train_loader, val_loader, test_loader, services = create_multimodal_loaders(
            data_root="data/RCAEval",
            batch_size=4,
            seq_len=60,
            seed=42,
            require_multimodal=True
        )
        n_services = len(services)
        print(f"   Services: {n_services} ({', '.join(services[:3])}...)")
        print(f"   Train: {len(train_loader.dataset)} cases")
        
        # Create model
        model = create_multimodal_model(
            n_services=n_services,
            n_metric_features=64,
            n_log_features=32,
            n_trace_features=32,
            hidden_dim=32,
            embed_dim=128,
            dropout=0.35,
            logs_encoder_type='tfidf'
        )
        print(f"   Model: {sum(p.numel() for p in model.parameters()):,} params")
        
        # Create PCMCI-only computer
        pcmci_computer = CausalWeightComputer(
            cache_path="outputs/test_pcmci_cache.pkl",
            services=services
        )
        
        # Create PCMCI+LLM computer
        llm_manager = CausalWeightManager(
            pcmci_cache_path="outputs/test_pcmci_cache.pkl",
            llm_cache_path="outputs/test_llm_cache.pkl",
            lambda_pcmci=0.7,
            lambda_prior=0.3,
            use_llm_prior=True
        )
        
        print("✅ All components initialized")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Test 3: Compare causal weights (PCMCI-only vs PCMCI+LLM)
    # =========================================================================
    print("\n[3/5] Comparing causal weight methods...")
    
    try:
        # Get a sample batch
        batch = next(iter(train_loader))
        case_ids = batch['case_id']
        systems = batch['system']
        
        print(f"   Sample batch: {len(case_ids)} cases")
        print(f"   Systems: {set(systems)}")
        
        # PCMCI-only weights
        pcmci_weights = pcmci_computer.get_batch_weights(
            case_ids, n_services, 'cpu'
        )
        
        # PCMCI + LLM weights
        system_type = get_system_type(systems[0])
        llm_weights = llm_manager.get_batch_weights(
            case_ids, services, system_type, 'cpu'
        )
        
        print(f"   PCMCI weights shape: {pcmci_weights.shape}")
        print(f"   LLM+PCMCI weights shape: {llm_weights.shape}")
        
        # Compute difference
        diff = (llm_weights - pcmci_weights).abs().mean().item()
        print(f"   Mean absolute difference: {diff:.4f}")
        
        if diff > 0.01:
            print("✅ LLM prior is adding meaningful signal (diff > 0.01)")
        else:
            print("⚠️ LLM prior has minimal effect (using cached/fallback values)")
        
    except Exception as e:
        print(f"❌ Weight comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Test 4: Model forward pass with both methods
    # =========================================================================
    print("\n[4/5] Testing model forward pass...")
    
    try:
        model.eval()
        
        metrics = batch['metrics']
        logs = batch['logs']
        traces = batch['traces']
        targets = batch['target']
        
        # Forward with PCMCI-only
        with torch.no_grad():
            out_pcmci = model(metrics, logs, traces, pcmci_weights)
        
        # Forward with PCMCI+LLM
        with torch.no_grad():
            out_llm = model(metrics, logs, traces, llm_weights)
        
        print(f"   PCMCI-only output shape: {out_pcmci['logits'].shape}")
        print(f"   LLM+PCMCI output shape: {out_llm['logits'].shape}")
        
        # Check if predictions differ
        pcmci_preds = out_pcmci['ranking'][:, 0]
        llm_preds = out_llm['ranking'][:, 0]
        
        same_preds = (pcmci_preds == llm_preds).sum().item()
        total_preds = len(pcmci_preds)
        
        print(f"   Same predictions: {same_preds}/{total_preds}")
        
        # Compare with ground truth
        pcmci_correct = (pcmci_preds == targets).sum().item()
        llm_correct = (llm_preds == targets).sum().item()
        
        print(f"   PCMCI-only correct: {pcmci_correct}/{total_preds}")
        print(f"   LLM+PCMCI correct: {llm_correct}/{total_preds}")
        
        print("✅ Model forward pass successful")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Test 5: Gradient flow (ensure LLM prior doesn't break training)
    # =========================================================================
    print("\n[5/5] Testing gradient flow...")
    
    try:
        model.train()
        
        # Forward
        outputs = model(metrics, logs, traces, llm_weights)
        logits = outputs['logits']
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits, targets)
        
        # Backward
        loss.backward()
        
        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        avg_grad = np.mean(grad_norms)
        max_grad = np.max(grad_norms)
        
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Avg gradient norm: {avg_grad:.6f}")
        print(f"   Max gradient norm: {max_grad:.6f}")
        
        if avg_grad > 0 and not np.isnan(avg_grad):
            print("✅ Gradients flow correctly")
        else:
            print("❌ Gradient flow issue detected")
            return False
            
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    
    print("\nTo train with LLM Causal Prior, run:")
    print("  python scripts/train_multimodal_v4.py --use-llm-prior --lambda-pcmci 0.7 --lambda-prior 0.3")
    
    return True


if __name__ == '__main__':
    success = test_integration()
    sys.exit(0 if success else 1)
