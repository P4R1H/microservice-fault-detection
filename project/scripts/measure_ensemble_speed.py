"""Quick script to measure ensemble inference time."""
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multimodal_data import create_multimodal_loaders
from src.models.rca_v4_multimodal import create_multimodal_model
from src.causal.pcmci import CausalWeightComputer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load 4 models
paths = [
    'outputs/models/v4_s42.pt',
    'outputs/models/v4_s123.pt', 
    'outputs/models/v4_s456.pt',
    'outputs/models/v4_s789.pt'
]

print("Loading 4 models...")
models = []
services = None

for p in paths:
    ckpt = torch.load(p, weights_only=False, map_location=device)
    args = ckpt['args']
    services = ckpt['services']
    m = create_multimodal_model(
        n_services=len(services),
        n_metric_features=64,
        n_log_features=32,
        n_trace_features=32,
        hidden_dim=32,
        embed_dim=128,
        dropout=0.35,
        logs_encoder_type='tfidf'
    ).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    models.append(m)

print(f"Loaded {len(models)} models")

# Get test data
_, _, test_loader, _ = create_multimodal_loaders(
    data_root='data/RCAEval',
    batch_size=1,
    seq_len=60,
    n_metric_features=64,
    n_log_features=32,
    n_trace_features=32,
    seed=42,
    require_multimodal=True
)

causal = CausalWeightComputer(
    cache_path='outputs/causal_cache_multimodal.pkl',
    services=services
)

batch = next(iter(test_loader))
metrics = batch['metrics'].to(device)
logs = batch['logs'].to(device)
traces = batch['traces'].to(device)
cw = causal.get_batch_weights(batch['case_id'], metrics.shape[1], device)  # type: ignore

# Warmup
print("\nWarmup...")
with torch.no_grad():
    for m in models:
        m(metrics, logs, traces, cw)
torch.cuda.synchronize()

# Measure single model
print("\nMeasuring single model inference (100 runs)...")
single_times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        out = models[0](metrics, logs, traces, cw)
        probs = torch.softmax(out['logits'], dim=-1)
    torch.cuda.synchronize()
    single_times.append((time.perf_counter() - start) * 1000)

print(f"Single model: {sum(single_times)/len(single_times):.2f} ms")

# Measure ensemble
print("\nMeasuring ensemble inference (100 runs)...")
times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        probs = []
        for m in models:
            out = m(metrics, logs, traces, cw)
            probs.append(torch.softmax(out['logits'], dim=-1))
        avg = torch.stack(probs).mean(0)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - start) * 1000)

mean_time = sum(times) / len(times)
print(f"\n{'='*60}")
print("ENSEMBLE INFERENCE TIME (4 models)")
print(f"{'='*60}")
print(f"  Mean: {mean_time:.2f} ms")
print(f"  Min:  {min(times):.2f} ms")
print(f"  Max:  {max(times):.2f} ms")
print(f"\n  Speedup vs SOTA (892ms): {892/mean_time:.1f}x faster")
print(f"  Still real-time capable: {'YES' if mean_time < 100 else 'NO'}")
print(f"{'='*60}")
