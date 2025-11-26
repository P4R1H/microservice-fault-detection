"""
Test Gemini logs encoder integration.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.encoders.logs_encoder import GeminiLogsEncoder, TFIDFLogsEncoder

def test_gemini_encoder():
    """Test GeminiLogsEncoder forward pass."""
    print("=" * 60)
    print("Testing GeminiLogsEncoder")
    print("=" * 60)
    
    # Create encoder
    encoder = GeminiLogsEncoder(
        n_log_features=32,
        hidden_dim=64,
        embed_dim=64,
        cache_dir="outputs/llm_cache"
    )
    
    # Set template names (simulating real log template names)
    template_names = [
        "frontend_error_connection",
        "db_slow_query",
        "auth_login_success",
        "payment_timeout",
        "cart_update_failed",
        "inventory_low_stock",
        "shipping_rate_error",
        "email_send_success",
    ] + [f"template_{i}" for i in range(8, 32)]
    
    encoder.set_template_names(template_names)
    print(f"Set {len(template_names)} template names")
    
    # Test forward pass
    batch_size = 4
    n_services = 10
    seq_len = 60
    n_features = 32
    
    x = torch.randn(batch_size * n_services, seq_len, n_features).abs()  # Template counts are positive
    print(f"Input shape: {x.shape}")
    
    encoder.eval()
    with torch.no_grad():
        output = encoder(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size * n_services}, 64)")
    
    assert output.shape == (batch_size * n_services, 64), f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    
    print("✓ GeminiLogsEncoder forward pass OK")
    
    # Check if embeddings were cached
    print(f"Cache size: {len(GeminiLogsEncoder._embedding_cache)} embeddings")
    
    return True

def test_model_with_gemini():
    """Test full model with Gemini encoder."""
    print("\n" + "=" * 60)
    print("Testing MultimodalRCAModel with Gemini")
    print("=" * 60)
    
    from src.models.rca_v4_multimodal import MultimodalRCAModel
    
    # Create model with Gemini logs encoder
    model = MultimodalRCAModel(
        n_services=10,
        n_metric_features=64,
        n_log_features=32,
        n_trace_features=32,
        hidden_dim=32,
        embed_dim=128,
        logs_encoder_type='gemini'
    )
    
    print(f"Model created with gemini logs encoder")
    
    # Test forward pass
    batch_size = 2
    metrics = torch.randn(batch_size, 10, 60, 64)
    logs = torch.randn(batch_size, 10, 60, 32).abs()
    traces = torch.randn(batch_size, 10, 60, 32)
    
    model.eval()
    with torch.no_grad():
        output = model(metrics, logs, traces)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    
    assert output['logits'].shape == (batch_size, 10), f"Logits shape mismatch"
    assert output['probs'].shape == (batch_size, 10), f"Probs shape mismatch"
    assert output['probs'].sum(dim=1).allclose(torch.ones(batch_size)), "Probs don't sum to 1"
    
    # Count parameters
    params = model.count_parameters()
    print(f"\nParameter counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    
    print("\n✓ Full model with Gemini encoder OK")
    return True

def compare_encoders():
    """Compare output characteristics of TF-IDF vs Gemini encoders."""
    print("\n" + "=" * 60)
    print("Comparing TF-IDF vs Gemini Encoders")
    print("=" * 60)
    
    # Same input
    x = torch.randn(40, 60, 32).abs()
    
    # TF-IDF encoder
    tfidf_enc = TFIDFLogsEncoder(n_log_features=32, embed_dim=64)
    tfidf_enc.eval()
    with torch.no_grad():
        tfidf_out = tfidf_enc(x)
    
    # Gemini encoder
    gemini_enc = GeminiLogsEncoder(n_log_features=32, embed_dim=64)
    gemini_enc.eval()
    with torch.no_grad():
        gemini_out = gemini_enc(x)
    
    print(f"TF-IDF output - mean: {tfidf_out.mean():.4f}, std: {tfidf_out.std():.4f}")
    print(f"Gemini output - mean: {gemini_out.mean():.4f}, std: {gemini_out.std():.4f}")
    
    # Check output diversity
    tfidf_var = tfidf_out.var(dim=0).mean()
    gemini_var = gemini_out.var(dim=0).mean()
    print(f"TF-IDF variance across samples: {tfidf_var:.4f}")
    print(f"Gemini variance across samples: {gemini_var:.4f}")
    
    print("\n✓ Encoder comparison complete")

if __name__ == '__main__':
    try:
        test_gemini_encoder()
        test_model_with_gemini()
        compare_encoders()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
