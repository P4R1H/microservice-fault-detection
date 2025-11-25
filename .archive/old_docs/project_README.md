# AIOps Microservice Anomaly Detection & Root Cause Analysis

**Status**: Initialization Phase
**Last Updated**: 2025-11-14

## Project Overview

This project implements a comprehensive multimodal AIOps system for microservice anomaly detection and root cause analysis. The system integrates three data modalities (metrics, logs, traces) using modern deep learning techniques including foundation models, graph neural networks, and causal inference.

### Key Features
- **Multimodal Fusion**: Combines metrics, logs, and traces for robust RCA
- **Foundation Models**: Leverages Chronos-Bolt-Tiny or TCN for time-series forecasting
- **Causal Discovery**: Uses PCMCI algorithm for causal graph inference
- **Graph Learning**: 2-layer GNN on service dependency graphs
- **Comprehensive Evaluation**: 10+ ablations and 5+ baseline comparisons
- **Production-Ready**: Modular, reproducible, well-documented

## Project Structure

```
/fault-detection-microservices
├── .workspace/              # Working memory & tracking (NOT in docs/)
│   ├── memory.md            # Long-term project understanding
│   ├── context.md           # Current session tracking
│   ├── todo.md              # Task management
│   ├── decisions.md         # Decision log
│   └── notes.md             # Scratch space
├── reference/               # Academic references & archived docs
│   ├── literature-review.txt
│   ├── midsem-report.txt
│   └── research-results.txt
├── project/                 # Main codebase
│   ├── config/              # YAML configurations
│   │   ├── model_config.yaml
│   │   ├── experiment_config.yaml
│   │   └── data_config.yaml
│   ├── docs/                # FORMAL DOCUMENTATION ONLY
│   │   ├── MODULE_INTERFACES.md
│   │   └── PHASE2_SETUP.md
│   ├── src/                 # Source code
│   │   ├── data/            # Data loading & preprocessing
│   │   ├── encoders/        # Metrics, Logs, Traces encoders
│   │   ├── causal/          # PCMCI causal discovery
│   │   ├── fusion/          # Multimodal fusion
│   │   ├── models/          # RCA models
│   │   ├── evaluation/      # Metrics & ablations
│   │   ├── baselines/       # Statistical baselines
│   │   └── utils/           # General utilities
│   ├── scripts/             # Executable scripts
│   ├── tests/               # Test suite
│   ├── experiments/         # Experiment runners (gitignored)
│   ├── outputs/             # Results (gitignored)
│   │   ├── models/
│   │   ├── results/
│   │   ├── figures/
│   │   └── logs/
│   ├── setup.py             # Package installation
│   └── requirements.txt
└── data/                    # Local datasets (gitignored)
    └── RCAEval/
```

## Quick Start

### Prerequisites
- **Python**: 3.8+ (tested with 3.11.14)
- **GPU**: NVIDIA RTX 4070 Mobile (8GB VRAM) or equivalent
- **CPU**: 16 cores / 22 threads recommended
- **RAM**: 16GB minimum
- **Disk**: ~10GB free space (dataset + models)

### Installation

**Step 1: Clone Repository**
```bash
git clone <repository-url>
cd fault-detection-microservices/project
```

**Step 2: Install Dependencies**

For GPU systems (recommended):
```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric with CUDA
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

For CPU-only systems (development):
```bash
pip install -r requirements.txt
```

**Step 3: Download RCAEval Dataset**
```bash
# Download TrainTicket dataset (~1.5GB)
python scripts/download_dataset.py

# Or download all three systems (~4.2GB)
python scripts/download_dataset.py --all
```

This downloads from Zenodo (DOI: 10.5281/zenodo.14590730):
- 270 multimodal failure cases (90 per system)
- Metrics, logs, and traces with ground truth labels
- Systems: TrainTicket, SockShop, Online Boutique

**Step 4: Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from src.utils.data_loader import RCAEvalDataLoader; print('✅ Data loader ready')"
```

### Usage

**Load Dataset**:
```python
from src.utils.data_loader import RCAEvalDataLoader

# Initialize loader
loader = RCAEvalDataLoader('project/data/RCAEval')

# Get train/val/test splits (162/54/54)
train, val, test = loader.load_splits()

# Access a failure case
case = train[0]
print(f"System: {case.system}")
print(f"Fault Type: {case.fault_type}")
print(f"Root Cause: {case.root_cause_service}")
print(f"Metrics shape: {case.metrics.shape}")
```

**Run Experiments** (Coming soon):
```bash
# Baseline experiments
python experiments/run_baselines.py

# Full multimodal system
python experiments/run_full_system.py --config configs/default.yaml

# Ablation studies
python experiments/run_ablations.py
```

## Dataset

This project uses the **RCAEval TrainTicket dataset**, a standard benchmark for microservice root cause analysis containing:
- System metrics (CPU, memory, latency, etc.)
- Application logs
- Distributed traces
- Ground-truth fault labels

## Methodology

### Architecture Components

1. **Metrics Module**: Time-series forecasting using Chronos-Bolt-Tiny (zero-shot) or TCN
2. **Logs Module**: Drain3 parser + template embeddings + anomaly scoring
3. **Traces Module**: Service dependency graph construction + 2-layer GNN
4. **Causal Module**: PCMCI-based causal graph discovery
5. **Fusion Module**: Cross-modal attention for intermediate fusion

### Ablation Studies

The project includes comprehensive ablation studies:
- Single modality: Metrics-only, Logs-only, Traces-only
- Pairwise: Metrics+Logs, Metrics+Traces, Logs+Traces
- Full system: All modalities
- Architecture: With/without GNN, with/without PCMCI
- Foundation models: With/without pretrained models

### Baselines

Comparison against 5+ state-of-the-art baselines including:
- BARO
- Statistical methods
- Deep learning approaches
- (Additional baselines TBD)

## Results

Results will be added as experiments are completed.

## Documentation

Full academic documentation is available in the `/docs` folder:
- **final_report.md**: Complete project report
- **architecture.md**: System architecture and design
- **methodology.md**: Detailed methodology and experimental setup
- **results.md**: Experimental results and analysis
- **ablations.md**: Ablation study results
- **literature_review.md**: Related work and references

## Contributing

This is a university research project. For questions or collaboration inquiries, please contact the project maintainer.

## License

TBD

## Acknowledgments

- RCAEval benchmark dataset
- Chronos foundation model by Amazon
- PCMCI algorithm by Jakob Runge et al.
- Drain3 log parser

---

**Project Status**: Awaiting initial materials and dataset access
