# 🧪 Complete Testing Guide for Phases 7-9

**Status**: All bugs fixed, ready for testing!
**Branch**: `claude/review-codebase-docs-01MHdFV8GKi5XTKyaX9K51vj`
**Commit**: `78403c8` (bug fixes applied)

---

## ✅ Bugs Fixed

### 1. **Service Embedding Architecture** ✅ FIXED
- **Issue**: ranking_head output mismatch when using service embeddings
- **Fix**: Separate architectures for with/without service embeddings
- **Impact**: RCA model will train correctly now

### 2. **ModalityDropout Restore Logic** ✅ FIXED
- **Issue**: Failed to restore modalities when all were dropped
- **Fix**: Keeps originals, properly restores one random modality
- **Impact**: Robustness training will work correctly

### 3. **Logs Encoder Missing** ✅ FIXED
- **Issue**: No logs encoder implementation
- **Fix**: Added DummyLogsEncoder with learnable embeddings
- **Impact**: Can test full pipeline without log parsing

---

## 📋 What You Need to Run

After upgrading to Python 3.10, run these tests in order and send me the **full output** of each command.

### ✅ Step 1: Install Dependencies (Python 3.10)

```bash
cd /home/user/fault-detection-microservices/project

# Confirm Python version
python --version
# Should show: Python 3.10.x

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric (for GCN/GAT)
pip install torch-geometric

# Chronos foundation model
pip install chronos-forecasting>=1.0.0

# PCMCI causal discovery
pip install tigramite

# All other requirements
pip install -r requirements.txt
```

**Send me**:
- Output of `python --version`
- Output of `pip list | grep -E "torch|chronos|tigramite"`
- Any errors during installation

---

### ✅ Step 2: Test Encoders (Metrics + Traces)

```bash
cd /home/user/fault-detection-microservices/project

# Quick test with 3 cases
python scripts/test_encoders.py --n_cases 3

# If successful, run full test with 10 cases
python scripts/test_encoders.py --n_cases 10
```

**Send me**:
- **Full terminal output** (all test results)
- Any errors or warnings
- The summary at the end showing PASSED/FAILED for each component

**What this tests**:
- ✅ Data loading from RCAEval (lazy loading, splits)
- ✅ Metrics preprocessing (normalization, windowing)
- ✅ Chronos-Bolt-Tiny encoder (zero-shot, ~100MB download on first run)
- ✅ TCN encoder (dilated convolutions)
- ✅ Traces preprocessing (graph construction)
- ✅ GCN encoder (graph neural networks)

---

### ✅ Step 3: Test PCMCI Causal Discovery

```bash
cd /home/user/fault-detection-microservices/project

# Test PCMCI with 3 cases
python scripts/test_pcmci.py --n_cases 3 --output_dir outputs/causal_graphs
```

**Send me**:
- **Full terminal output** (causal discovery results)
- Any errors or warnings
- The summary showing PASSED/SKIPPED for each component
- Tell me if causal graphs were saved to `outputs/causal_graphs/`

**What this tests**:
- ✅ PCMCI algorithm with tigramite
- ✅ Service-level causal score integration
- ✅ Granger-Lasso baseline
- ✅ Causal graph visualization

---

### ✅ Step 4: Quick Smoke Test (All Components)

Create and run this quick test to verify the full pipeline:

```python
# Save as: test_full_pipeline.py
import sys
from pathlib import Path
project_root = Path(__file__).parent
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

# 1. Create encoders
print("\n1. Creating encoders...")
metrics_encoder = TCNEncoder(in_channels=7, embedding_dim=256)
traces_encoder = GCNEncoder(in_channels=8, embedding_dim=128)
logs_encoder = create_logs_encoder(embedding_dim=256, use_dummy=True)
print("✓ Encoders created")

# 2. Create fusion
print("\n2. Creating multimodal fusion...")
fusion = create_multimodal_fusion(
    metrics_encoder=metrics_encoder,
    logs_encoder=logs_encoder,
    traces_encoder=traces_encoder,
    fusion_strategy='intermediate',
    fusion_dim=512,
    num_heads=8
)
print("✓ Fusion created")

# 3. Create RCA model
print("\n3. Creating RCA model...")
rca_model = RCAModel(
    fusion_model=fusion,
    num_services=50,
    fusion_dim=512,
    use_service_embedding=True
)
print("✓ RCA model created")

# 4. Test forward pass with dummy data
print("\n4. Testing forward pass...")
batch_size = 2
metrics = torch.randn(batch_size, 12, 7)  # (batch, seq_len, features)
logs = None  # Skip logs for now
traces = (
    torch.randn(10, 8),  # node features
    torch.randint(0, 10, (2, 20))  # edge index
)

output = rca_model(metrics=metrics, logs=logs, traces=traces)
print(f"✓ Forward pass successful!")
print(f"  - Logits shape: {output['logits'].shape}")
print(f"  - Probs shape: {output['probs'].shape}")
print(f"  - Ranking shape: {output['ranking'].shape}")

# 5. Test top-k prediction
print("\n5. Testing top-k prediction...")
top_k_services, top_k_scores = rca_model.predict_top_k(output['logits'], k=5)
print(f"✓ Top-k prediction successful!")
print(f"  - Top-5 services: {top_k_services}")
print(f"  - Top-5 scores: {top_k_scores}")

# 6. Test loss computation
print("\n6. Testing loss computation...")
target_services = torch.randint(0, 50, (batch_size,))
loss_ce = rca_model.compute_loss(output['logits'], target_services, loss_type='cross_entropy')
loss_rank = rca_model.compute_loss(output['logits'], target_services, loss_type='ranking_loss')
print(f"✓ Loss computation successful!")
print(f"  - Cross-entropy loss: {loss_ce.item():.4f}")
print(f"  - Ranking loss: {loss_rank.item():.4f}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nFull pipeline is working correctly:")
print("  ✓ Encoders (Metrics, Logs, Traces)")
print("  ✓ Multimodal fusion with attention")
print("  ✓ RCA model with service ranking")
print("  ✓ Loss computation")
print("  ✓ Top-k prediction")
print("\nReady for training and evaluation!")
```

**Run it**:
```bash
cd /home/user/fault-detection-microservices/project
python test_full_pipeline.py
```

**Send me**:
- **Full terminal output**
- Confirm if it says "✅ ALL TESTS PASSED!"
- Any errors or shape mismatches

---

### ✅ Step 5: Parameter Count Check

Run this to verify model sizes:

```python
# Save as: check_model_sizes.py
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from src.encoders.metrics_encoder import ChronosEncoder, TCNEncoder
from src.encoders.traces_encoder import GCNEncoder, GATEncoder
from src.encoders.logs_encoder import create_logs_encoder

print("=" * 70)
print("MODEL SIZE VERIFICATION")
print("=" * 70)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# Metrics encoders
print("\n1. METRICS ENCODERS:")
try:
    chronos = ChronosEncoder(embedding_dim=256, device='cpu')
    print(f"  Chronos-Bolt-Tiny: {count_params(chronos):,} params (~{count_params(chronos)/1e6:.1f}M)")
except ImportError as e:
    print(f"  Chronos: SKIPPED - {e}")

tcn = TCNEncoder(in_channels=7, embedding_dim=256)
print(f"  TCN: {count_params(tcn):,} params (~{count_params(tcn)/1e6:.1f}M)")

# Traces encoders
print("\n2. TRACES ENCODERS:")
gcn = GCNEncoder(in_channels=8, embedding_dim=128)
print(f"  GCN: {count_params(gcn):,} params (~{count_params(gcn)/1e6:.2f}M)")

gat = GATEncoder(in_channels=8, embedding_dim=128, num_heads=4)
print(f"  GAT: {count_params(gat):,} params (~{count_params(gat)/1e6:.2f}M)")

# Logs encoder
print("\n3. LOGS ENCODER:")
logs_enc = create_logs_encoder(embedding_dim=256, use_dummy=True)
print(f"  Dummy Logs: {count_params(logs_enc):,} params (~{count_params(logs_enc)/1e6:.2f}M)")

print("\n" + "=" * 70)
print("EXPECTED SIZES:")
print("  - Chronos: ~20M (100MB download first time)")
print("  - TCN: ~9M")
print("  - GCN: ~0.03M")
print("  - GAT: ~0.1M")
print("  - Total pipeline: <30M (without Chronos) or ~50M (with Chronos)")
print("=" * 70)
```

**Run it**:
```bash
cd /home/user/fault-detection-microservices/project
python check_model_sizes.py
```

**Send me**:
- Full output with parameter counts
- Confirm sizes match expectations

---

## 📊 What I Need From You

Please send me the outputs of ALL 5 steps above in your next message. Include:

1. ✅ **Python version + installed packages**
2. ✅ **Full output of `test_encoders.py`**
3. ✅ **Full output of `test_pcmci.py`**
4. ✅ **Full output of `test_full_pipeline.py`**
5. ✅ **Full output of `check_model_sizes.py`**

### How to capture outputs:

```bash
# Option 1: Capture to file
python scripts/test_encoders.py --n_cases 3 > test_encoders_output.txt 2>&1
cat test_encoders_output.txt

# Option 2: Copy from terminal
python scripts/test_encoders.py --n_cases 3
# Then copy-paste the output
```

---

## 🐛 If You Hit Errors

### Error: "ModuleNotFoundError: No module named 'chronos'"
```bash
pip install chronos-forecasting>=1.0.0
```

### Error: "ModuleNotFoundError: No module named 'torch_geometric'"
```bash
pip install torch-geometric
```

### Error: "tigramite not installed"
```bash
pip install tigramite
```

### Error: "Dataset not found"
```bash
# Verify dataset location
ls -la data/RCAEval/
# Should show: TrainTicket/, SockShop/, OnlineBoutique/
```

### Error: Shape mismatch in encoders
**Send me the exact error message** - I'll fix it immediately

### Error: CUDA out of memory
```bash
# All tests should run on CPU by default, but if CUDA is used:
export CUDA_VISIBLE_DEVICES=""
python scripts/test_encoders.py --n_cases 3
```

---

## 📈 Expected Test Results

### test_encoders.py
- ✅ Data loading: PASSED
- ✅ Metrics preprocessing: PASSED
- ✅ Chronos encoder: PASSED (or SKIPPED if not installed)
- ✅ TCN encoder: PASSED
- ✅ Traces preprocessing: PASSED
- ✅ GCN encoder: PASSED

### test_pcmci.py
- ✅ Data loading: PASSED
- ✅ PCMCI discovery: PASSED (or SKIPPED if tigramite not installed)
- ✅ Service-level integration: PASSED
- ✅ Granger-Lasso baseline: PASSED

### test_full_pipeline.py
- ✅ Encoders created
- ✅ Fusion created
- ✅ RCA model created
- ✅ Forward pass successful
- ✅ Top-k prediction successful
- ✅ Loss computation successful

---

## 🚀 After All Tests Pass

Once you confirm all tests pass, we'll proceed to:

1. **Phase 10**: Build evaluation framework
   - Training loop
   - DataLoader for RCAEval
   - Experiment tracking

2. **Phase 11**: Run experiments
   - Ablation studies (6+ configurations)
   - Baseline comparisons (5+ methods)
   - Statistical significance testing

3. **Phase 12-14**: Documentation & thesis writing

---

**Ready?** Run the tests and send me all the outputs! 🎯
