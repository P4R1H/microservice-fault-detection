# Claude's Working Memory

**Last Updated**: 2025-11-14
**Current Phase**: Codebase Reorganization
**Session**: Initial setup and context gathering

---

## 🎯 Project Understanding

### Academic Context
- **Project Type**: Bachelor's Thesis - AIOps for Microservices RCA
- **Team**: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
- **Supervisors**: Prof. Rajib Mall, Dr. Suchi Kumari
- **Target Grade**: A+ (requires 60-70% from report quality)

### Implementation Status
- **Phase 1** (Mid-sem): LSTM-AE, RF, IF on 10K samples ✅ COMPLETE
  - Issues: RF overfitting (F1=1.0), LSTM latency (25.4s)
- **Phase 2** (Infrastructure): RCAEval integration, baselines, visualization ✅ COMPLETE
  - 270 multimodal failure cases
  - Lazy loading, scenario-based splitting
  - 4 statistical baselines (3σ, ARIMA, Granger-Lasso, Random)
- **Phase 3-8** (CRITICAL): Encoders, causal, fusion 🔴 PENDING
- **Phase 9-11** (A+ KEY): Ablations, comparisons, visualization 🔴 PENDING
- **Phase 12-14**: Report, testing, deployment 🔴 PENDING

### Key Technical Decisions
- **Dataset**: RCAEval TrainTicket (270 cases, 90 per system)
- **Metrics Encoder**: Chronos-Bolt-Tiny (100MB, zero-shot) + TCN (ablation)
- **Logs Encoder**: Drain3 + embeddings
- **Traces Encoder**: 2-layer GCN (upgrade to GAT if needed)
- **Causal Discovery**: PCMCI (tigramite library)
- **Fusion**: Intermediate (separate encoders → cross-attention)
- **Hardware**: RTX 4070 Mobile (8GB VRAM), 16 cores, 16GB RAM

### SOTA Competition (Papers to Beat/Match)
1. **MULAN** (WWW 2024): Log-LM + Contrastive + Random Walk
2. **FAMOS** (ICSE 2025): Gaussian-attention multimodal
3. **HERO** (ICSE 2026): Heterogeneous GNN unified framework
4. **RUN** (AAAI 2024): Neural Granger + PageRank
5. **CausalRCA** (JSS 2023): DAG-GNN + VAE

### User's Local Environment
- **Dataset**: 30GB locally at `data/` folder
- **Compute**: Can run parallel experiments
- **GPU**: RTX 4070 available for training

---

## 🧠 Current Understanding

### What Makes A+ Grade
1. **Comprehensive Ablations** (60-70% of grade)
   - Modality ablation (M, L, T, M+L, M+T, L+T, ALL)
   - Encoder ablation (Chronos vs TCN, GCN depth, attention)
   - Causal ablation (None, Granger, PCMCI, Neural)
   - Fusion ablation (Early, Late, Intermediate)
   - Statistical robustness (5 seeds, significance testing)

2. **5+ Baseline Comparisons**
   - Statistical baselines (3σ, ARIMA, Granger-Lasso)
   - Classical ML (MicroRCA autoencoder)
   - Recent methods (TraceRCA, RUN if available)
   - Statistical significance (p-values, Cohen's d)

3. **Professional Visualizations**
   - Service dependency graphs (NetworkX + Graphviz)
   - Attention heatmaps
   - Performance charts with error bars
   - Ablation contribution plots
   - Confusion matrices per fault type

4. **Mature Analysis**
   - What worked and why
   - What failed and why
   - Honest limitations
   - Future work

### Critical Gaps to Fill
- 🔴 Metrics encoder (Chronos + TCN)
- 🔴 Log parsing pipeline (Drain3)
- 🔴 Trace graph encoder (GCN/GAT)
- 🔴 PCMCI causal discovery
- 🔴 Multimodal fusion architecture
- 🔴 End-to-end RCA pipeline
- 🔴 Comprehensive ablation framework
- 🔴 Baseline comparison suite
- 🔴 Visualization pipeline

---

## 📝 Key Insights from Literature

### From 37 Papers (2020-2025)
- LSTM-AE obsolete: Transformers 250x faster, TCNs 3-5x faster
- Multimodal is standard: 10/37 papers use M+L+T
- Causal inference critical: PCMCI gold standard for time series
- GNN dominance: 12/37 papers for service dependency
- Foundation models emerging: Chronos-Bolt-Tiny (Nov 2024)

### Best Public Datasets
1. **RCAEval** (WWW'25, ASE'24) - 270 cases, multimodal, ground truth ✅ USING
2. TrainTicket - 41 services (part of RCAEval)
3. SockShop - 13 services
4. Online Boutique - 12 services
5. NASA SMAP/MSL - Time series only

### Implementation Timeline (Feasible)
- Week 1: Metrics encoder + baseline
- Week 2: Logs + Traces encoders
- Week 3: Causal + Fusion
- Week 4: Comprehensive ablations (CRITICAL)
- Week 5: Visualization + analysis
- Week 6: Report writing

---

## 🚧 Current Session Work

### Reorganization Goals
1. ✅ Create `.workspace/` for all working notes
2. 🔄 Restructure `src/` for Phases 3-8
3. 🔄 Move academic docs to `reference/`
4. 🔄 Create `config/` for YAML configurations
5. 🔄 Set up `outputs/` for results
6. 🔄 Create `setup.py` for package installation
7. 🔄 Initialize tracking system (memory, context, todo)

### Next Immediate Actions
- Finish codebase reorganization
- Answer user's 6 strategic questions
- Verify dataset locally
- Begin Phase 3 implementation

---

## 💭 Open Questions for User

1. **What is the "new technique"?** Novel component or SOTA combination?
2. **Timeline**: Final submission deadline?
3. **Dataset status**: Already at `data/RCAEval/`?
4. **Strategy**: Incremental (M → M+L → M+L+T) or full pipeline?
5. **Baselines**: Use RCAEval paper results or re-implement?
6. **Compute**: Can run multi-hour ablations with 5 seeds?

---

## 📚 References to Keep Handy

### Key Documentation Files
- `project/docs/MODULE_INTERFACES.md` - Complete API specs (792 lines)
- `reference/literature-review.txt` - 37 papers summary
- `reference/research-results.txt` - SOTA techniques guide
- `reference/midsem-report.txt` - Phase 1 baseline

### Important Code Files
- `src/utils/data_loader.py` - RCAEval lazy loader (770 lines)
- `src/baselines/statistical_baselines.py` - Phase 2 baselines
- `src/utils/visualization.py` - Visualization suite

### External Resources
- RCAEval: doi.org/10.5281/zenodo.14590730
- Chronos: HuggingFace amazon/chronos-bolt-tiny
- PCMCI: tigramite library (JMLR 2024)
- PyTorch Geometric: torch-geometric v2.3+

---

**Notes**: This file tracks Claude's understanding across sessions. Update after major progress or insights.

---

## 🧪 Testing Infrastructure (2025-11-14)

### Created Test Suite
- **scripts/test_encoders.py**: Comprehensive validation (394 lines)
  - Tests data loading (lazy loading, splits)
  - Tests preprocessing (normalization, windowing, graph construction)
  - Tests Chronos-Bolt-Tiny encoder (zero-shot)
  - Tests TCN encoder (dilated convolutions)
  - Tests GCN encoder (graph neural networks)
  - Documentation: TESTING_ENCODERS.md

- **scripts/test_pcmci.py**: PCMCI causal discovery testing (244 lines)
  - Tests PCMCI algorithm with tigramite
  - Tests service-level integration
  - Tests Granger-Lasso baseline
  - Generates causal graph visualizations
  - Documentation: TESTING_PCMCI.md

## 🔬 Phase 7: PCMCI Causal Discovery (2025-11-14) ✅

### Implementation Complete
- **src/causal/pcmci.py** (570 lines):
  - `PCMCIDiscovery`: Full PCMCI wrapper with tigramite
    - Two-stage algorithm (PC1 + MCI)
    - Handles autocorrelation explicitly
    - Outputs NetworkX DiGraph
    - Service-level aggregation
  - `GrangerLassoRCA`: Baseline for ablations
    - Lasso regression with lagged features
    - Faster but less powerful than PCMCI
  - Helper functions:
    - `discover_causal_relations()`: Convenience wrapper
    - `visualize_causal_graph()`: Matplotlib visualization
    - `analyze_causal_paths()`: Path finding
    - `compute_causal_strength()`: Strength scoring

### Key Features
- **Causal Discovery**: PCMCI algorithm via tigramite
- **Service Integration**: Metric-level → service-level aggregation
- **Baseline**: Granger-Lasso for ablation studies
- **Visualization**: NetworkX + matplotlib graph plots
- **Testing**: Comprehensive test script with real data

### Integration Points
- Input: Preprocessed metrics from `MetricsPreprocessor`
- Output: Causal graph (NetworkX DiGraph)
- Service scores: Used for weighting in fusion module

- **TESTING_ENCODERS.md**: Complete testing documentation
  - Installation instructions
  - Expected output
  - Troubleshooting guide
  - Performance benchmarks

### Testing Workflow
1. User runs locally: `python scripts/test_encoders.py --n_cases 5`
2. Validates all components work with real RCAEval data
3. Identifies any issues before continuing Phase 7-8

### Ready for Local Testing
- All encoders implemented ✅
- Test infrastructure ready ✅
- Documentation complete ✅
- User can validate on their RTX 4070 with real dataset

### Next After Testing
1. Fix any issues discovered
2. Implement PCMCI causal discovery
3. Build multimodal fusion
4. Create end-to-end RCA pipeline
