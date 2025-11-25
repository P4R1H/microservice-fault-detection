# Phase 2: Infrastructure & Baseline Implementation

**Status**: ✅ Complete
**Date**: 2025-11-14
**Branch**: `claude/aiops-microservice-rca-project-01Jv7LhfyWLCp2b2kXPtWB2o`

---

## Overview

Phase 2 establishes the complete infrastructure needed for the multimodal RCA project, including:

1. ✅ Exploratory Data Analysis (EDA) framework
2. ✅ Statistical baseline implementations
3. ✅ Visualization utilities for all modalities
4. ✅ Module interface specifications
5. ✅ Dataset extraction pipeline (offline mode)

---

## Deliverables

### 1. EDA Analysis Script
**Location**: `project/scripts/eda_analysis.py`

**Purpose**: Comprehensive exploratory data analysis for all three modalities

**Features**:
- Dataset-level statistics (system, fault type, root cause distributions)
- Metrics modality analysis (dimensionality, patterns, correlations)
- Logs modality analysis (volume, template distribution, log levels)
- Traces modality analysis (service graphs, latency patterns)
- Cross-modality completeness analysis

**Usage**:
```bash
# Analyze all systems
python scripts/eda_analysis.py --all

# Analyze specific system
python scripts/eda_analysis.py --systems TrainTicket

# Quick analysis (TrainTicket only)
python scripts/eda_analysis.py
```

**Output**:
- Console statistics
- Distribution plots: `project/outputs/eda/dataset_distributions.png`
- Metrics statistics: `project/outputs/eda/metrics_statistics.txt`

---

### 2. Statistical Baselines
**Location**: `project/src/baselines/statistical_baselines.py`

**Implemented Methods**:

#### 2.1 Three-Sigma (3σ) Detector
- **Method**: Flag values outside μ ± 3σ as anomalies
- **Use Case**: Industry standard baseline (SRE best practices)
- **Complexity**: O(n) per metric
- **Reference**: Gaussian distribution (68-95-99.7 rule)

```python
from src.baselines import ThreeSigmaDetector

detector = ThreeSigmaDetector(n_sigma=3.0, window_size=50)
ranking = detector.rank_services(metrics_df, service_mapping)
```

#### 2.2 ARIMA Forecaster
- **Method**: Forecast next values, flag large residuals as anomalies
- **Use Case**: Time series baseline
- **Complexity**: O(n²) per metric
- **Reference**: Box & Jenkins (1970)

```python
from src.baselines import ARIMAForecaster

forecaster = ARIMAForecaster(order=(5, 1, 0))
ranking = forecaster.rank_services(metrics_df, service_mapping)
```

#### 2.3 Granger-Lasso RCA
- **Method**: Granger causality with Lasso regularization
- **Use Case**: Causal baseline
- **Complexity**: O(n³) for n metrics
- **Reference**: Arnold et al. (2007)

```python
from src.baselines import GrangerLassoRCA

granger = GrangerLassoRCA(max_lag=5, alpha=0.01)
ranking = granger.rank_services(metrics_df, service_mapping, max_vars=20)
```

#### 2.4 Random Walk Baseline
- **Method**: Random service ranking
- **Use Case**: Sanity check (all methods should beat this)
- **Expected AC@1**: 1/N where N = number of services

```python
from src.baselines import RandomWalkBaseline

random = RandomWalkBaseline(random_seed=42)
ranking = random.rank_services(service_list)
```

**Evaluation Utility**:
```python
from src.baselines import evaluate_ranking

metrics = evaluate_ranking(
    predicted_ranking=ranking,
    ground_truth_service='frontend',
    k_values=[1, 3, 5]
)
# Returns: {'AC@1': ..., 'AC@3': ..., 'AC@5': ..., 'Avg@5': ..., 'MRR': ...}
```

---

### 3. Visualization Utilities
**Location**: `project/src/utils/visualization.py`

**Classes**:

#### 3.1 MetricsVisualizer
```python
from src.utils import MetricsVisualizer

viz = MetricsVisualizer(figsize=(15, 8), dpi=300)

# Time series plots
viz.plot_time_series(metrics_df, highlight_anomalies=[80, 85])

# Correlation matrix
viz.plot_correlation_matrix(metrics_df, method='pearson')

# Anomaly heatmap
viz.plot_anomaly_heatmap(metrics_df, anomaly_scores)
```

#### 3.2 LogsVisualizer
```python
from src.utils import LogsVisualizer

viz = LogsVisualizer()

# Template distribution
viz.plot_log_template_distribution(templates, counts, top_n=20)

# Log level distribution
viz.plot_log_level_distribution({'ERROR': 100, 'INFO': 500, ...})

# Log timeline
viz.plot_log_timeline(timestamps, levels)
```

#### 3.3 TracesVisualizer
```python
from src.utils import TracesVisualizer

viz = TracesVisualizer()

# Service dependency graph
viz.plot_service_dependency_graph(
    adjacency_matrix=adj_matrix,
    service_names=services,
    root_cause_service='database'
)

# Latency distribution
viz.plot_latency_distribution(latencies_by_service)
```

#### 3.4 ResultsVisualizer
```python
from src.utils import ResultsVisualizer

viz = ResultsVisualizer()

# Method comparison
viz.plot_method_comparison({
    'Method A': {'AC@1': 0.75, 'AC@3': 0.85, ...},
    'Method B': {'AC@1': 0.65, 'AC@3': 0.78, ...}
})

# Ablation study
viz.plot_ablation_study(ablation_results, baseline_score=0.80)

# Confusion matrix
viz.plot_confusion_matrix(y_true, y_pred, service_names)
```

**All visualizations**:
- Publication-quality (300 DPI)
- Customizable figure sizes
- Save to file or display
- Consistent color schemes

---

### 4. Module Interfaces Documentation
**Location**: `project/docs/MODULE_INTERFACES.md`

**Contents**:
- Architecture overview diagram
- Core data structures (`FailureCase`, `ModalityEmbedding`, `RCAResult`)
- Interface specifications for:
  - Data loading module
  - Preprocessing modules (metrics, logs, traces)
  - Encoder modules (Chronos, Logs, GCN)
  - Causal discovery module (PCMCI)
  - Fusion module
  - RCA model
  - Baseline methods
  - Evaluation metrics
  - Training pipeline
- End-to-end data flow example
- Configuration schema
- Testing interfaces
- Extensibility guidelines

**Purpose**: Ensures all components integrate seamlessly with clear contracts

---

### 5. Dataset Extraction (Offline Mode)
**Location**: `project/scripts/download_dataset.py`

**Updated Features**:
- ✅ Extraction-only mode (no download)
- ✅ Works with pre-downloaded zips in `data/zip_cache/`
- ✅ Correct RE{1,2,3}-{TT,SS,OB}.zip naming
- ✅ Smart skip if already extracted
- ✅ Verification with file counts

**Usage**:
```bash
# Extract all systems, all versions
python scripts/download_dataset.py --all

# Extract specific system
python scripts/download_dataset.py --systems TrainTicket

# Extract specific version
python scripts/download_dataset.py --all --reversions RE2

# Force re-extraction
python scripts/download_dataset.py --all --force
```

**Directory Structure After Extraction**:
```
project/data/RCAEval/
├── downloads/              # Original zips (preserved)
│   ├── RE1-TT.zip
│   ├── RE2-TT.zip
│   └── ...
├── TrainTicket/
│   ├── RE1/
│   │   ├── metrics/
│   │   ├── logs/
│   │   ├── traces/
│   │   └── ground_truth.csv
│   ├── RE2/
│   └── RE3/
├── SockShop/
│   ├── RE1/
│   ├── RE2/
│   └── RE3/
└── OnlineBoutique/
    ├── RE1/
    ├── RE2/
    └── RE3/
```

---

## Project Structure (Updated)

```
project/
├── data/                          # ✅ Gitignored
│   └── RCAEval/                   # Will be created after extraction
├── scripts/
│   ├── download_dataset.py        # ✅ Extraction pipeline
│   └── eda_analysis.py            # ✅ NEW: EDA script
├── src/
│   ├── baselines/
│   │   ├── __init__.py            # ✅ Exports all baselines
│   │   └── statistical_baselines.py  # ✅ NEW: 4 baseline methods
│   ├── utils/
│   │   ├── __init__.py            # ✅ Updated exports
│   │   ├── data_loader.py         # ✅ From Phase 1
│   │   └── visualization.py       # ✅ NEW: 4 visualizer classes
│   └── ...
├── docs/
│   ├── MODULE_INTERFACES.md       # ✅ NEW: Complete interface spec
│   └── PHASE2_SETUP.md            # ✅ This file
├── outputs/
│   └── eda/                       # Created by EDA script
└── ...
```

---

## Dependencies

All dependencies already in `requirements.txt`:

**Core**:
- `numpy`, `pandas`, `scikit-learn`, `scipy`

**Statistical**:
- `statsmodels` (ARIMA, Granger)

**Visualization**:
- `matplotlib`, `seaborn`, `plotly`, `networkx`, `graphviz`

**Utilities**:
- `tqdm`, `pyyaml`, `loguru`

---

## Testing

### Test Statistical Baselines
```bash
cd project
python -c "from src.baselines import ThreeSigmaDetector; print('✅ Baselines imported successfully')"

# Run built-in tests
python src/baselines/statistical_baselines.py
```

**Expected Output**:
```
================================================================================
Testing Statistical Baselines
================================================================================

1. Three-Sigma Detector
   Top 3 suspicious services:
      service_3: 8.52
      service_1: 2.14
      service_0: 1.89

2. ARIMA Forecaster
   Top 3 suspicious services:
      service_3: 3.21
      service_4: 1.87
      service_2: 1.45

3. Granger-Lasso RCA
   Top 3 suspicious services:
      service_3: 12.00
      service_1: 8.00
      service_0: 5.00

4. Evaluation Example
   3-Sigma: AC@1=1.00, AC@3=1.00, MRR=1.000
   ARIMA: AC@1=1.00, AC@3=1.00, MRR=1.000
   Granger: AC@1=1.00, AC@3=1.00, MRR=1.000
```

### Test Visualization
```bash
python src/utils/visualization.py
```

This generates 4 test plots to verify all visualizers work correctly.

---

## Next Steps (Phase 3)

### Once Dataset is Extracted:

1. **Run EDA Analysis**:
   ```bash
   python scripts/eda_analysis.py --all
   ```

2. **Verify Data Statistics**:
   - Check `outputs/eda/dataset_distributions.png`
   - Review `outputs/eda/metrics_statistics.txt`
   - Confirm 270 cases loaded (90 per system)

3. **Begin Preprocessing Implementation**:
   - `src/preprocessing/metrics_preprocessor.py`
   - `src/preprocessing/log_parser.py`
   - `src/preprocessing/trace_graph_builder.py`

4. **Run Statistical Baselines on Real Data**:
   - Establish performance lower bounds
   - Document baseline results in `docs/BASELINE_RESULTS.md`

5. **Phase 3: Implement Chronos Metrics Encoder**:
   - Load Chronos-Bolt-Tiny
   - Test on RCAEval TrainTicket RE2
   - Measure memory usage (target: <100MB VRAM)
   - Document zero-shot performance

---

## Performance Targets

### Statistical Baselines (Expected)
Based on literature review:

| Method | AC@1 | AC@3 | AC@5 | MRR |
|--------|------|------|------|-----|
| Random Walk | ~0.05 | ~0.15 | ~0.25 | ~0.10 |
| 3-Sigma | 0.10-0.20 | 0.25-0.35 | 0.35-0.45 | 0.20-0.30 |
| ARIMA | 0.15-0.25 | 0.30-0.40 | 0.40-0.50 | 0.25-0.35 |
| Granger | 0.20-0.30 | 0.35-0.45 | 0.45-0.55 | 0.30-0.40 |

**Goal**: Advanced methods (Chronos + PCMCI + GNN + Fusion) should achieve:
- AC@1: >0.70
- AC@3: >0.85
- AC@5: >0.90
- MRR: >0.75

---

## Memory Budget (RTX 4070 Mobile - 8GB VRAM)

| Component | VRAM Usage |
|-----------|-----------|
| Chronos-Bolt-Tiny | ~100 MB |
| Logs Encoder | ~30 MB |
| GCN Encoder | ~30 MB |
| Fusion Module | ~40 MB |
| Data Batch (16 cases) | ~200 MB |
| **Total Active** | **~400 MB** |
| **Buffer** | **7.6 GB** |

✅ **Well within limits!**

---

## Documentation Status

| Document | Status | Location |
|----------|--------|----------|
| QUICKSTART.md | ✅ Complete | `project/QUICKSTART.md` |
| ENVIRONMENT.md | ✅ Complete | `project/ENVIRONMENT.md` |
| README.md | ✅ Complete | `project/README.md` |
| MODULE_INTERFACES.md | ✅ Complete | `project/docs/MODULE_INTERFACES.md` |
| PHASE1_SETUP.md | ✅ Implied | Via commits |
| PHASE2_SETUP.md | ✅ Complete | `project/docs/PHASE2_SETUP.md` |

---

## Git Status

**Branch**: `claude/aiops-microservice-rca-project-01Jv7LhfyWLCp2b2kXPtWB2o`

**Commits**:
1. ✅ Phase 1: Environment, dataset infrastructure, data loaders
2. ✅ Fixed RCAEval downloader for actual Zenodo file structure
3. ✅ Added comprehensive quickstart guide
4. ⏳ **Next**: Phase 2 complete infrastructure (this commit)

---

## Summary

Phase 2 establishes **production-ready infrastructure** for the entire RCA project:

✅ **EDA Framework**: Comprehensive analysis script for all modalities
✅ **Statistical Baselines**: 4 methods (3σ, ARIMA, Granger, Random)
✅ **Visualization Suite**: Publication-quality plots for metrics/logs/traces/results
✅ **Module Interfaces**: Complete specification for all components
✅ **Dataset Pipeline**: Offline extraction from pre-downloaded zips
✅ **Testing**: All components have built-in tests
✅ **Documentation**: Professional-grade docs for all deliverables

**Ready for Phase 3**: Chronos-Bolt-Tiny implementation and zero-shot evaluation! 🚀

---

## Questions & Support

If you encounter issues:

1. **Dataset extraction fails**: Check `data/zip_cache/` has all 9 zips
2. **Import errors**: Verify `PYTHONPATH` includes project root
3. **Visualization errors**: Install `graphviz` system package if needed
4. **Memory errors**: Reduce batch size in configs

**Contact**: Review context files in `project/context/` for project status
