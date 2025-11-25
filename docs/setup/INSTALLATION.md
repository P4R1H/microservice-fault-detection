# Installation Guide

Complete setup instructions for the Multimodal RCA system.

---

## Prerequisites

- **Python**: 3.10+ (tested with 3.10, 3.11)
- **GPU**: NVIDIA with CUDA 11.8+ (optional, CPU supported)
- **Disk**: ~10GB (dataset + models)
- **RAM**: 16GB recommended

---

## Quick Setup (10 minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/P4R1H/fault-detection-microservices.git
cd fault-detection-microservices/project
```

### Step 2: Create Environment

**Option A - Conda (Recommended):**
```bash
conda create -n rca python=3.10
conda activate rca
```

**Option B - venv:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (choose one based on your setup)

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install Chronos foundation model
pip install chronos-forecasting

# Install Tigramite for PCMCI
pip install tigramite

# Install all other requirements
pip install -r requirements.txt
```

### Step 4: Download Dataset

**Minimal download for testing (~500MB):**
```bash
python scripts/download_dataset.py --systems TrainTicket --reversions RE2
```

**Full dataset (~4GB):**
```bash
python scripts/download_dataset.py --all
```

**Main benchmark (RE2 across all systems):**
```bash
python scripts/download_dataset.py --all --reversions RE2
```

### Step 5: Verify Installation

```bash
# Quick test with 1 case (~30 seconds)
python scripts/test_encoders.py --n_cases 1

# If successful, try 3 cases (~2 minutes)
python scripts/test_encoders.py --n_cases 3
```

---

## Dataset Download Options

| Command | Downloads | Size |
|---------|-----------|------|
| `--systems TrainTicket --reversions RE2` | 90 cases | ~500MB |
| `--all --reversions RE2` | 270 cases | ~1.5GB |
| `--all` | 810 cases | ~4GB |

**Flags:**
- `--force` - Re-download even if exists
- `--no-extract` - Download only, don't extract

---

## Expected Test Output

When tests pass, you'll see:

```
==================================================
TEST: Data Loading
==================================================
✅ Loaded 270 cases total
✅ Train: 162, Val: 54, Test: 54

==================================================
TEST: Metrics Encoder - Chronos-Bolt-Tiny
==================================================
✅ Chronos encoder initialized
✅ Encoded 3 cases successfully
✅ Output shape: torch.Size([3, 60, 64])

==================================================
TEST: Traces Encoder - GCN
==================================================
✅ Service graphs built: 3 cases
✅ GCN encoder initialized
✅ Output shape: torch.Size([3, num_services, 64])

==================================================
✅ ALL TESTS PASSED!
==================================================
```

---

## Test Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `test_encoders.py` | Test all encoders (Chronos, TCN, GCN) | ~30s-5min |
| `test_pcmci.py` | Test causal discovery | ~2-5min |
| `test_full_pipeline.py` | End-to-end RCA pipeline | ~5-10min |
| `test_data_loading.py` | Verify dataset extraction | ~10sec |
| `test_baselines.py` | Test statistical baselines | ~2-5min |

---

## Troubleshooting

### ModuleNotFoundError: No module named 'torch'
```bash
pip install torch torchvision torchaudio
```

### ModuleNotFoundError: No module named 'chronos'
```bash
pip install chronos-forecasting
```

### FileNotFoundError: RCAEval dataset not found
```bash
python scripts/download_dataset.py --systems TrainTicket --reversions RE2
```

### CUDA out of memory
Edit `config/experiment_config.yaml`:
```yaml
device: 'cpu'
batch_size: 4
```

### Chronos model download slow
First run downloads ~100MB model from HuggingFace. Be patient!

---

## GPU-Specific Setup (CUDA 11.8)

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric with CUDA support
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Next Steps

Once installation is verified:

1. **Run experiments**: `python scripts/train_rca_model.py`
2. **Run ablations**: `python scripts/run_all_ablations.py`
3. **Compare baselines**: `python scripts/run_baseline_comparisons.py`
4. **Generate figures**: `python scripts/visualization/generate_all_figures.py`
