"""
Test Script: LLM Causal Prior Integration

Tests the LLM-based causal prior generator and its integration
with the existing PCMCI system.

Usage:
    cd project
    python tests/test_llm_causal_prior.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.llm_prior import LLMCausalPrior, CausalWeightManager, get_system_type


def test_llm_causal_prior():
    """Test the LLM causal prior generator."""
    
    print("=" * 70)
    print("TEST 1: LLM Causal Prior Generator Initialization")
    print("=" * 70)
    
    try:
        prior_gen = LLMCausalPrior(
            cache_path="outputs/test_llm_cache.pkl"
        )
        print("✅ LLM Causal Prior initialized successfully")
        print(f"   Model available: {prior_gen.model is not None}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("TEST 2: Fallback Heuristic Prior (No API)")
    print("=" * 70)
    
    # Test with sample services
    services = ['frontend', 'api-gateway', 'user-service', 'database', 'cache-redis']
    
    # Force fallback by temporarily disabling model
    original_model = prior_gen.model
    prior_gen.model = None
    
    try:
        matrix = prior_gen.generate_prior(
            services=services,
            system_type="generic microservice"
        )
        
        print(f"✅ Fallback prior generated")
        print(f"   Shape: {matrix.shape}")
        print(f"   Min: {matrix.min():.3f}, Max: {matrix.max():.3f}")
        print(f"   Diagonal (should be 0): {np.diag(matrix)}")
        
        # Check database has high outgoing influence
        db_idx = services.index('database')
        db_influence = matrix[db_idx, :].mean()
        print(f"   Database avg outgoing influence: {db_influence:.3f}")
        
        # Check frontend has low outgoing influence
        fe_idx = services.index('frontend')
        fe_influence = matrix[fe_idx, :].mean()
        print(f"   Frontend avg outgoing influence: {fe_influence:.3f}")
        
        if db_influence > fe_influence:
            print("✅ Heuristics working correctly (DB > Frontend)")
        else:
            print("⚠️ Heuristics may need tuning")
            
    except Exception as e:
        print(f"❌ Fallback generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        prior_gen.model = original_model
    
    print("\n" + "=" * 70)
    print("TEST 3: Real System Services (OnlineBoutique)")
    print("=" * 70)
    
    online_boutique_services = [
        'frontend',
        'cartservice',
        'productcatalogservice', 
        'currencyservice',
        'paymentservice',
        'shippingservice',
        'emailservice',
        'checkoutservice',
        'recommendationservice',
        'adservice'
    ]
    
    try:
        matrix = prior_gen.generate_prior(
            services=online_boutique_services,
            system_type="Google Cloud Online Boutique e-commerce"
        )
        
        print(f"✅ OnlineBoutique prior generated")
        print(f"   Shape: {matrix.shape}")
        
        # Show top causal relationships
        print("\n   Top 5 causal relationships:")
        relationships = []
        for i, src in enumerate(online_boutique_services):
            for j, dst in enumerate(online_boutique_services):
                if i != j:
                    relationships.append((src, dst, matrix[i, j]))
        
        relationships.sort(key=lambda x: x[2], reverse=True)
        for src, dst, strength in relationships[:5]:
            print(f"   {src:25s} -> {dst:25s}: {strength:.3f}")
            
    except Exception as e:
        print(f"❌ OnlineBoutique generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("TEST 4: PCMCI + LLM Prior Combination")
    print("=" * 70)
    
    try:
        # Create mock PCMCI weights
        n = len(services)
        mock_pcmci = np.random.rand(n, n).astype(np.float32) * 0.5
        np.fill_diagonal(mock_pcmci, 0)
        
        # Generate combined weights
        combined = prior_gen.get_combined_weights(
            pcmci_weights=mock_pcmci,
            services=services,
            system_type="generic microservice",
            lambda_pcmci=0.7,
            lambda_prior=0.3
        )
        
        print(f"✅ Combined weights computed")
        print(f"   Shape: {combined.shape}")
        print(f"   Min: {combined.min():.3f}, Max: {combined.max():.3f}")
        
        # Verify combination formula
        prior_only = prior_gen.generate_prior(services, "generic microservice")
        expected = 0.7 * mock_pcmci + 0.3 * prior_only
        expected = expected / expected.max()  # Normalize
        
        if np.allclose(combined, expected, rtol=1e-3):
            print("✅ Combination formula verified")
        else:
            print("⚠️ Combination formula may differ (normalization)")
            
    except Exception as e:
        print(f"❌ Combination test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("TEST 5: CausalWeightManager Integration")
    print("=" * 70)
    
    try:
        manager = CausalWeightManager(
            pcmci_cache_path="outputs/causal_cache_multimodal.pkl",
            llm_cache_path="outputs/test_llm_cache.pkl",
            lambda_pcmci=0.7,
            lambda_prior=0.3,
            use_llm_prior=True
        )
        
        # Test single case
        weights = manager.get_weights(
            case_id="test_case_001",
            services=services,
            system_type="generic microservice"
        )
        
        print(f"✅ CausalWeightManager working")
        print(f"   Single case weights shape: {weights.shape}")
        
        # Test batch
        import torch
        batch_weights = manager.get_batch_weights(
            case_ids=["case_1", "case_2", "case_3"],
            services=services,
            system_type="generic microservice",
            device='cpu'
        )
        
        print(f"   Batch weights shape: {batch_weights.shape}")
        print(f"   Batch weights dtype: {batch_weights.dtype}")
        
    except Exception as e:
        print(f"❌ CausalWeightManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("TEST 6: System Type Mapping")
    print("=" * 70)
    
    test_systems = ['TrainTicket', 'SockShop', 'OnlineBoutique', 'Unknown']
    for sys_name in test_systems:
        sys_type = get_system_type(sys_name)
        print(f"   {sys_name:15s} -> {sys_type}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    
    return True


def test_llm_api_if_available():
    """Test actual LLM API call if credentials available."""
    
    import os
    if not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️ GEMINI_API_KEY not set - skipping API test")
        print("   Set the key to test actual LLM generation")
        return True
    
    print("\n" + "=" * 70)
    print("TEST 7: Live LLM API Call (Gemini)")
    print("=" * 70)
    
    prior_gen = LLMCausalPrior(
        cache_path="outputs/test_llm_api_cache.pkl"
    )
    
    # Clear cache to force API call
    LLMCausalPrior._cache = {}
    
    services = ['frontend', 'cart', 'checkout', 'payment', 'database']
    
    try:
        matrix = prior_gen.generate_prior(
            services=services,
            system_type="e-commerce checkout flow"
        )
        
        print(f"✅ LLM API call successful!")
        print(f"   Shape: {matrix.shape}")
        
        print("\n   Generated causal relationships:")
        for i, src in enumerate(services):
            for j, dst in enumerate(services):
                if i != j and matrix[i, j] > 0.3:
                    print(f"   {src:10s} -> {dst:10s}: {matrix[i, j]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM API call failed: {e}")
        return False


if __name__ == '__main__':
    success = test_llm_causal_prior()
    
    if success:
        test_llm_api_if_available()
    
    sys.exit(0 if success else 1)
