# Environment Setup & Configuration

**Last Updated**: 2025-11-14

## Development Environment

### Current Sandbox Environment
- **Python**: 3.11.14 ✅
- **Platform**: Linux (containerized/sandboxed)
- **GPU**: Not available in current environment (CPU-only)
- **Purpose**: Code development, EDA, architecture design, CPU-based experiments

### Target Production Environment (User's Local Machine)
- **GPU**: NVIDIA RTX 4070 Mobile (80W, 8GB VRAM)
- **CPU**: 16 cores / 22 threads
- **RAM**: 16GB
- **CUDA**: 11.8 or 12.1 (to be confirmed)
- **Purpose**: GPU training (Chronos, TCN, GCN), full experiments

## Development Strategy

Given the environment constraints, we'll adopt a **hybrid development approach**:

### Phase 1: Sandbox Development (Current Environment)
✅ **What we CAN do here**:
- Install RCAEval and download dataset
- Exploratory Data Analysis (EDA) on all modalities
- Data preprocessing pipelines
- Architecture design and module structure
- Baseline implementations (statistical, Granger-Lasso)
- Code organization and documentation
- CloudWatch adapter interfaces
- Unit tests

❌ **What we CANNOT do here**:
- GPU-accelerated model training (Chronos, TCN, GCN)
- Large-scale experiments
- Final performance benchmarking

### Phase 2: Local GPU Training (User's Machine)
**Transfer approach**:
1. Export well-structured code from sandbox
2. User clones repository on local RTX 4070 machine
3. Install GPU-specific dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```
4. Run training scripts with GPU acceleration
5. Execute full ablation studies and baseline comparisons

## Installation Instructions

### For Sandbox (Current) - CPU-only
```bash
cd /home/user/fault-detection-microservices/project

# Install core dependencies (CPU versions)
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install networkx tqdm pyyaml joblib click python-dotenv

# Install RCAEval
pip install RCAEval[default]

# Install causal inference libraries
pip install tigramite causal-learn

# Install log parsing
pip install drain3

# Install baselines
pip install catboost statsmodels lightgbm

# Install visualization
pip install plotly graphviz pydot

# Install statistical testing
pip install pingouin

# Install experiment tracking
pip install tensorboard wandb
```

### For Local Machine (GPU) - Full Installation
```bash
# 1. Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install PyTorch Geometric with CUDA support
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 3. Install Chronos foundation model
pip install chronos-forecasting transformers accelerate

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Verify CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Dataset Setup

### RCAEval RE2-TrainTicket Dataset
- **Size**: 4.21GB compressed
- **Location**: Downloaded via RCAEval package
- **Systems**: TrainTicket, SockShop, Online Boutique
- **Cases**: 270 multimodal failure scenarios

**Download command**:
```python
from RCAEval import download_dataset
download_dataset('RE2-TrainTicket', target_dir='project/data/')
```

## Compute Resource Allocation

### Memory Requirements (RTX 4070 Mobile - 8GB VRAM)
| Component | VRAM Usage | Status |
|-----------|------------|--------|
| Chronos-Bolt-Tiny | 100MB | ✅ Fits easily |
| TCN (7 layers, 64-128 channels) | 50-100MB | ✅ Fits easily |
| GCN (2-layer, hidden=64) | 10-30MB | ✅ Fits easily |
| Data batches (batch_size=32) | 100-200MB | ✅ Acceptable |
| **Total Active** | ~500MB | ✅ Comfortable |
| **Buffer** | 7.5GB free | ✅ Plenty of headroom |

### CPU Requirements (16 cores / 22 threads)
- **PCMCI causal discovery**: Parallelizable across CPU cores
- **Data preprocessing**: Batch processing with joblib
- **Granger-Lasso**: Fast CPU execution
- **Drain3 log parsing**: CPU-intensive but efficient

## Development Workflow

### Current Phase: Setup & EDA (Sandbox)
1. ✅ Create project structure
2. ✅ Write requirements.txt
3. ✅ Document environment
4. 🔄 Install RCAEval
5. ⏳ Download dataset
6. ⏳ Run EDA on all modalities
7. ⏳ Design module interfaces
8. ⏳ Implement data loaders

### Next Phase: Implementation (Local GPU)
1. Train Chronos-Bolt-Tiny (zero-shot inference)
2. Train TCN on metrics data
3. Train GCN on service graphs
4. Run PCMCI causal discovery
5. Implement multimodal fusion
6. Execute comprehensive ablations
7. Run all baseline comparisons

### Final Phase: Documentation & Polish (Sandbox + Local)
1. Generate all visualizations
2. Write academic report
3. Create architecture diagrams
4. Compile results tables
5. Statistical significance testing
6. Final code cleanup

## Code Portability

All code written in sandbox environment will be **GPU-agnostic** where possible:

```python
# Good: Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Good: Configurable batch size
batch_size = 32 if torch.cuda.is_available() else 8

# Good: Conditional mixed precision
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
```

This ensures seamless transition between sandbox development and local GPU training.

## Verification Checklist

### Sandbox Environment ✅
- [x] Python 3.11.14 installed
- [x] Project structure created
- [x] requirements.txt written
- [ ] RCAEval installed
- [ ] Dataset downloaded
- [ ] EDA notebooks ready

### Local GPU Environment (To be verified by user)
- [ ] CUDA 11.8+ installed
- [ ] NVIDIA driver updated
- [ ] PyTorch with CUDA support installed
- [ ] PyTorch Geometric with CUDA installed
- [ ] GPU memory available (8GB)
- [ ] Chronos model downloaded
- [ ] Training scripts execute without errors

---

**Next Steps**:
1. Install RCAEval in sandbox
2. Download dataset (4.21GB)
3. Begin EDA and module design
4. Prepare training scripts for local GPU execution
