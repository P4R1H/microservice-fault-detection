"""Quick test of multimodal data loading."""
import sys
sys.path.insert(0, '.')

from src.data.multimodal_data import create_multimodal_loaders

print("Creating data loaders...")
train_loader, val_loader, test_loader, services = create_multimodal_loaders(
    batch_size=4, seq_len=60, seed=42
)

print(f"\nServices: {services}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

print("\nTesting batch loading...")
batch = next(iter(train_loader))

print(f"\nBatch contents:")
print(f"  metrics: {batch['metrics'].shape}")
print(f"  logs: {batch['logs'].shape if batch['logs'] is not None else None}")
print(f"  traces: {batch['traces'].shape if batch['traces'] is not None else None}")
print(f"  target: {batch['target']}")
print(f"  case_ids: {batch['case_id'][:2]}...")

# Test model
print("\n\nTesting model forward pass...")
from src.models.rca_v4_multimodal import MultimodalRCAModel
import torch

model = MultimodalRCAModel(n_services=len(services))
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    out = model(batch['metrics'], batch['logs'], batch['traces'])
    print(f"\nOutput:")
    print(f"  logits: {out['logits'].shape}")
    print(f"  probs: {out['probs'].shape}")
    print(f"  ranking: {out['ranking']}")

print("\n✓ All tests passed!")
