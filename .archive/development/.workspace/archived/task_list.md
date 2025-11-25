# Task List

**Last Updated**: 2025-11-14 (Comprehensive Update after Materials Analysis)

## Status: PHASE 0 - Planning Complete, Ready for Implementation

### Completed Tasks
- [x] Create folder structure
- [x] Initialize context files
- [x] Receive and read mid-semester report
- [x] Receive and read literature review (37 papers)
- [x] Receive and read research results/blueprint
- [x] Comprehensive analysis of all materials
- [x] Update memory.md with full requirements
- [x] Update decisions.md with 8 major architectural decisions
- [x] Update notes.md with actionable insights
- [x] Define evaluation metrics (AC@k, Avg@k, MRR, NAB)
- [x] Identify 7+ baselines for comparison
- [x] Plan 12+ ablation configurations

---

## PHASE 1: Environment Setup & Dataset Acquisition (Week 1, Days 1-3)

### Critical Path Items
- [ ] Verify Python environment (Python 3.8+, CUDA availability)
- [ ] Create requirements.txt with all dependencies
- [ ] Install core libraries:
  - [ ] PyTorch + CUDA support
  - [ ] PyTorch Geometric (v2.3+)
  - [ ] RCAEval: `pip install RCAEval[default]`
  - [ ] tigramite (PCMCI)
  - [ ] causal-learn (Granger baselines)
  - [ ] Drain3 (log parsing)
  - [ ] Standard stack (numpy, pandas, sklearn, networkx)
- [ ] Download RCAEval RE2-TrainTicket dataset (4.21GB)
- [ ] Verify dataset integrity and structure
- [ ] Document compute resources (GPU model, VRAM, CPU cores)

### Deliverables
- [ ] Working Python environment with all dependencies
- [ ] RCAEval dataset downloaded and validated
- [ ] Environment documentation in README

**Blockers**: Need user confirmation on GPU specs and environment constraints

---

## PHASE 2: Exploratory Data Analysis (Week 1, Days 4-7)

### Metrics Analysis
- [ ] Load metrics data from RCAEval
- [ ] Understand schema (77-376 metrics per case)
- [ ] Compute statistics (mean, std, min, max, missing values)
- [ ] Visualize time series patterns
- [ ] Identify metric types (CPU, memory, network, disk)
- [ ] Check temporal alignment (5-min granularity)

### Logs Analysis
- [ ] Load log data (8.6-26.9M lines per system)
- [ ] Inspect log format and structure
- [ ] Run initial Drain3 parsing experiment
- [ ] Analyze template frequency distribution
- [ ] Identify anomalous log patterns

### Traces Analysis
- [ ] Load trace data (39.6-76.7M traces)
- [ ] Understand trace format (spans, parent-child relationships)
- [ ] Extract service call graph topology
- [ ] Compute basic graph metrics (nodes, edges, degree distribution)
- [ ] Visualize service dependency graph for 1-2 cases

### Ground Truth Analysis
- [ ] Understand ground truth format
- [ ] Validate root cause service labels
- [ ] Validate root cause indicator labels
- [ ] Analyze fault type distribution (6 types)
- [ ] Create train/val/test splits (60/20/20) - 162/54/54 cases

### Deliverables
- [ ] EDA notebook with visualizations
- [ ] Data statistics documented
- [ ] Train/val/test split defined
- [ ] Understanding of all three modalities

---

## PHASE 3: Baseline Implementation (Week 1-2)

### Statistical Baselines (Simple, Days 1-2)
- [ ] Implement 3-sigma thresholding for metrics
- [ ] Implement ARIMA forecasting baseline
- [ ] Implement simple frequency-based log anomaly detection
- [ ] Evaluate on validation set
- [ ] Document results in experiments/

### Phase 1 Baselines (Already Exist, Day 3)
- [ ] Locate Phase 1 code (ask user)
- [ ] Extract Random Forest implementation (F1=1.00)
- [ ] Extract Isolation Forest implementation (F1=0.367)
- [ ] Adapt to RCAEval dataset format
- [ ] Re-evaluate on new dataset

### Fast Causal Baseline (Day 4)
- [ ] Implement Granger-Lasso using causal-learn
- [ ] Configure for metrics data (tau_max=3-5)
- [ ] Run on validation set
- [ ] Measure AC@1, AC@3, AC@5
- [ ] Document as causal baseline

### RCAEval Built-in Methods (Day 5)
- [ ] Explore RCAEval's 15 built-in baseline methods
- [ ] Run 3-5 most relevant methods
- [ ] Compare performance on validation set
- [ ] Document for baseline comparison table

### Deliverables
- [ ] 5+ baseline implementations working
- [ ] Initial results table (methods × metrics)
- [ ] Code in project/src/baselines/

**Success Criteria**: Have working baselines before implementing main system

---

## PHASE 4: Metrics Module - Foundation Models (Week 2)

### Chronos-Bolt-Tiny Integration (Days 1-3)
- [ ] Install chronos from Hugging Face
- [ ] Understand input format and API
- [ ] Implement zero-shot inference pipeline
- [ ] Forecast expected values for anomaly detection
- [ ] Threshold on reconstruction error
- [ ] Evaluate on validation set (F1, AUC-ROC)
- [ ] Measure inference time (should be <100ms)

### TCN Implementation (Days 4-7)
- [ ] Implement TCN architecture:
  - [ ] 7 layers, kernel_size=3
  - [ ] Dilations: [1,2,4,8,16,32,64]
  - [ ] Channels: [64,128,256]
- [ ] Create training pipeline
- [ ] Hyperparameter tuning (lr, dropout, batch_size)
- [ ] Train on training set
- [ ] Evaluate on validation set
- [ ] Compare with Chronos performance
- [ ] Measure training time (target: <10 min)

### Metrics-Only Experiments
- [ ] Run ablation: Chronos vs TCN vs LSTM-AE (if available)
- [ ] Document performance improvements
- [ ] Analyze failure cases
- [ ] Create visualization comparing methods

### Deliverables
- [ ] Chronos-Bolt-Tiny working (zero-shot)
- [ ] TCN trained and evaluated
- [ ] Metrics-only RCA results
- [ ] Code in project/src/metrics_module/

---

## PHASE 5: Logs Module (Week 2-3)

### Drain3 Parser Integration (Days 1-2)
- [ ] Configure Drain3 (similarity_threshold=0.5, depth=4)
- [ ] Parse logs from training data
- [ ] Extract log templates
- [ ] Analyze template distribution
- [ ] Handle online streaming if needed

### Log Embedding & Anomaly Scoring (Days 3-4)
- [ ] Implement template embedding (TF-IDF or sentence transformers)
- [ ] Aggregate logs into 1-min windows
- [ ] Count event frequencies per template
- [ ] Implement anomaly scoring mechanism
- [ ] Evaluate logs-only RCA

### Logs-Only Experiments
- [ ] Run logs-only anomaly detection
- [ ] Evaluate on validation set
- [ ] Compare with metrics-only performance
- [ ] Document template patterns for anomalies

### Deliverables
- [ ] Drain3 parser working
- [ ] Log embeddings generated
- [ ] Logs-only RCA results
- [ ] Code in project/src/logs_module/

---

## PHASE 6: Traces Module - Graph Neural Networks (Week 3)

### Service Dependency Graph Construction (Days 1-3)
- [ ] Parse distributed traces
- [ ] Extract parent-child span relationships
- [ ] Build directed graph: nodes=services, edges=calls
- [ ] Implement using NetworkX
- [ ] Extract node features:
  - [ ] Response time (mean, p50, p90, p99)
  - [ ] CPU/memory usage
  - [ ] Request rate
  - [ ] Error rate
- [ ] Extract edge features:
  - [ ] Call frequency
  - [ ] Latency
  - [ ] Error rate
- [ ] Visualize sample graphs

### GCN Implementation (Days 4-6)
- [ ] Implement 2-layer GCN using PyTorch Geometric
- [ ] Configuration:
  - [ ] hidden_dim=64
  - [ ] dropout=0.3-0.5
  - [ ] lr=0.01, weight_decay=5e-4
- [ ] Create training loop
- [ ] Train on graph data
- [ ] Implement RCA scoring mechanism
- [ ] Evaluate on validation set

### Traces-Only Experiments (Day 7)
- [ ] Run traces-only RCA
- [ ] Evaluate graph-based localization
- [ ] Compare with metrics and logs performance
- [ ] Visualize graph with anomalies highlighted

### GAT Implementation (Optional, if time permits)
- [ ] Implement 2-layer GAT
- [ ] 4-8 attention heads (first layer)
- [ ] Compare with GCN
- [ ] Visualize attention weights

### Deliverables
- [ ] Service dependency graphs built
- [ ] GCN trained and working
- [ ] Traces-only RCA results
- [ ] Code in project/src/traces_module/
- [ ] (Optional) GAT implementation

---

## PHASE 7: Causal Inference Module (Week 3-4)

### PCMCI Implementation (Days 1-4)
- [ ] Install tigramite library
- [ ] Prepare time series data for PCMCI
- [ ] Configure hyperparameters:
  - [ ] tau_max=3-5
  - [ ] pc_alpha=0.1-0.2
  - [ ] alpha_level=0.01-0.05
- [ ] Start with ParCorr test (linear dependencies)
- [ ] Run PCMCI on metrics data
- [ ] Discover causal graph
- [ ] Visualize causal relationships
- [ ] Compare with ground truth labels

### Causal Graph Analysis (Days 5-6)
- [ ] Evaluate causal discovery quality
- [ ] Measure precision/recall vs ground truth
- [ ] Identify causal paths to root causes
- [ ] Integrate causal graph with RCA scoring
- [ ] Test on validation set

### Alternative Causal Methods (Day 7, optional)
- [ ] Try GPDC (nonlinear) if ParCorr insufficient
- [ ] Compare with Granger-Lasso baseline
- [ ] Document trade-offs

### Deliverables
- [ ] PCMCI causal discovery working
- [ ] Causal graphs visualized
- [ ] Causal-based RCA results
- [ ] Code in project/src/causal_module/

---

## PHASE 8: Multimodal Fusion (Week 4)

### Intermediate Fusion Architecture (Days 1-4)
- [ ] Design fusion architecture:
  ```
  Metrics → TCN/Chronos Encoder →
  Logs → Drain + Embedding →      Cross-Modal Attention → Unified Rep → RCA
  Traces → GCN Encoder →
  ```
- [ ] Implement separate encoders for each modality
- [ ] Time-align all modalities to 1-min windows:
  - [ ] Aggregate metrics (mean/max/p99)
  - [ ] Count/embed log events
  - [ ] Sample representative trace spans
- [ ] Implement cross-modal attention mechanism
- [ ] Fuse representations
- [ ] Train end-to-end on training set

### Full System Integration (Days 5-7)
- [ ] Integrate causal module with fusion output
- [ ] Implement unified RCA scoring
- [ ] Train full multimodal system
- [ ] Hyperparameter tuning
- [ ] Evaluate on validation set
- [ ] Compare with single-modality and pairwise results

### Deliverables
- [ ] Multimodal fusion working
- [ ] Full system RCA results
- [ ] Code in project/src/fusion_module/

---

## PHASE 9: Comprehensive Ablation Studies (Week 4-5)

### Data Modality Ablations (Days 1-3)
- [ ] 1. Metrics-only (already have)
- [ ] 2. Logs-only (already have)
- [ ] 3. Traces-only (already have)
- [ ] 4. Metrics + Logs
- [ ] 5. Metrics + Traces
- [ ] 6. Logs + Traces
- [ ] 7. All modalities (full system)
- [ ] Create comparison table with all metrics
- [ ] Quantify contribution of each modality

### Architecture Ablations (Days 4-5)
- [ ] 8. Without GNN (replace with simple graph features)
- [ ] 9. Without PCMCI (remove causal module)
- [ ] 10. Without pretrained model (no Chronos, only TCN)
- [ ] 11. Chronos vs TCN comparison
- [ ] 12. (Optional) GCN vs GAT
- [ ] Document component contributions

### Training Strategy Ablations (Day 6, optional)
- [ ] With vs without data augmentation (if used)
- [ ] Different loss functions
- [ ] Different fusion strategies

### Statistical Analysis (Day 7)
- [ ] Run each configuration 3-5 times with different seeds
- [ ] Compute mean ± std for all metrics
- [ ] Paired t-tests between methods (p<0.05)
- [ ] Wilcoxon signed-rank for non-normal distributions

### Deliverables
- [ ] 12+ ablation configurations completed
- [ ] Comprehensive ablation table
- [ ] Statistical significance documented
- [ ] Results in experiments/ablations/

---

## PHASE 10: Baseline Comparisons (Week 5)

### Run All Baselines on Test Set (Days 1-2)
- [ ] BARO (FSE'24) - if not already run
- [ ] Random Forest (Phase 1)
- [ ] Isolation Forest (Phase 1)
- [ ] Statistical methods (3-sigma, ARIMA)
- [ ] Granger-Lasso
- [ ] MicroRCA (if implemented)
- [ ] RCAEval built-in methods
- [ ] Our full system

### Comprehensive Comparison (Days 3-4)
- [ ] Create master comparison table:
  - [ ] Methods (rows) × Metrics (columns)
  - [ ] AC@1, AC@3, AC@5, Avg@5, MRR
  - [ ] F1, Precision, Recall, AUC-ROC
  - [ ] NAB scores
- [ ] Bold best values
- [ ] Highlight statistically significant improvements
- [ ] Compute relative improvements (%)

### Failure Case Analysis (Day 5)
- [ ] Identify cases where our method fails
- [ ] Analyze failure patterns
- [ ] Compare with baseline failures
- [ ] Document insights

### Per-Fault-Type Analysis (Days 6-7)
- [ ] Break down performance by fault type:
  - [ ] CPU exhaustion
  - [ ] Memory exhaustion
  - [ ] Disk issues
  - [ ] Network delay
  - [ ] Network packet loss
  - [ ] Socket exhaustion
- [ ] Identify which modalities help most for each fault type
- [ ] Document patterns

### Deliverables
- [ ] 7+ baseline comparison table
- [ ] Statistical significance tests
- [ ] Failure case analysis
- [ ] Per-fault-type breakdown
- [ ] Results in experiments/comparisons/

---

## PHASE 11: Visualizations & Analysis (Week 5-6)

### Architecture Diagrams (Days 1-2)
- [ ] Overall system architecture (draw.io or similar)
- [ ] Data flow pipeline diagram
- [ ] Intermediate fusion architecture detail
- [ ] GNN architecture for service graphs
- [ ] Export as high-res PNG/PDF

### Performance Visualizations (Days 3-4)
- [ ] Bar charts: Method comparisons with error bars
- [ ] Line plots: Metrics over different configurations
- [ ] Heatmaps: Attention weights (if using attention)
- [ ] Confusion matrices: Classification results
- [ ] Service dependency graphs with annotations

### Ablation Result Tables (Day 5)
- [ ] Create publication-quality tables
- [ ] Bold best values
- [ ] Include standard deviations
- [ ] LaTeX format (for docs)

### Causal Graph Visualizations (Day 6)
- [ ] Visualize discovered causal graphs
- [ ] Compare with ground truth
- [ ] Highlight causal paths to root causes
- [ ] Use NetworkX + Graphviz

### Additional Visualizations (Day 7, optional)
- [ ] Which modality contributes most? (attention analysis)
- [ ] Interactive dashboard (Streamlit)
- [ ] SHAP values for interpretability

### Deliverables
- [ ] All diagrams in high-res format
- [ ] All charts and plots
- [ ] Tables in LaTeX/Markdown
- [ ] Files in project/docs/figures/

---

## PHASE 12: Documentation (Week 6)

### Code Documentation (Days 1-2)
- [ ] Add docstrings to all functions/classes
- [ ] Create comprehensive README.md:
  - [ ] Project overview
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Directory structure
  - [ ] Dependencies
- [ ] Create requirements.txt
- [ ] Document hyperparameters
- [ ] Add comments to complex code sections

### Academic Report Writing (Days 3-7)

#### final_report.md
- [ ] Abstract (200 words)
- [ ] Introduction & Motivation
- [ ] Related Work (10-15 papers, emphasize 2024-2025)
- [ ] Methodology:
  - [ ] Dataset description (RCAEval)
  - [ ] System architecture
  - [ ] Each module detailed
  - [ ] Multimodal fusion approach
- [ ] Experimental Setup:
  - [ ] Evaluation metrics
  - [ ] Baseline methods
  - [ ] Implementation details
  - [ ] Hyperparameters
- [ ] Results:
  - [ ] Baseline comparisons
  - [ ] Ablation studies
  - [ ] Statistical significance
  - [ ] Per-fault-type analysis
- [ ] Discussion:
  - [ ] Key insights
  - [ ] Failure case analysis
  - [ ] Which modalities matter most
- [ ] Limitations (honest assessment)
- [ ] Conclusion & Future Work
- [ ] References (citations)

#### architecture.md
- [ ] System overview
- [ ] Module descriptions with diagrams
- [ ] Data flow
- [ ] Design decisions

#### methodology.md
- [ ] Dataset details
- [ ] Preprocessing steps
- [ ] Model architectures
- [ ] Training procedures

#### results.md
- [ ] All performance tables
- [ ] All charts
- [ ] Statistical tests

#### ablations.md
- [ ] Ablation study results
- [ ] Component contribution analysis

#### experiments.md
- [ ] Experimental setup
- [ ] Hyperparameters
- [ ] Reproducibility details

#### limitations.md
- [ ] Known constraints
- [ ] Failure modes
- [ ] Future improvements

### Presentation Preparation (Day 7)
- [ ] Create slides (15-20 slides)
- [ ] Include key visualizations
- [ ] Practice presentation

### Deliverables
- [ ] Comprehensive README
- [ ] Publication-grade report in /docs
- [ ] All supporting documentation
- [ ] Presentation slides

---

## PHASE 13: Final Testing & Reproducibility (Week 6)

### Reproducibility Checks (Days 1-2)
- [ ] Test installation on clean environment
- [ ] Verify all dependencies install correctly
- [ ] Run all experiments from scratch
- [ ] Confirm results match documented values
- [ ] Check all random seeds are set

### Code Quality Review (Day 3)
- [ ] Remove debug code
- [ ] Fix any TODOs
- [ ] Ensure consistent code style
- [ ] Run linters (pylint, black)
- [ ] Check for security issues

### Documentation Review (Day 4)
- [ ] Proofread all markdown docs
- [ ] Check all citations
- [ ] Verify all figures are referenced
- [ ] Ensure consistent terminology
- [ ] Check for typos

### Final Experiments (Days 5-6)
- [ ] Run final test set evaluation
- [ ] Confirm no data leakage
- [ ] Generate final result tables
- [ ] Create final visualizations

### Submission Preparation (Day 7)
- [ ] Package all deliverables
- [ ] Create submission archive
- [ ] Final README check
- [ ] Verify all files included

### Deliverables
- [ ] Fully reproducible codebase
- [ ] Complete documentation
- [ ] Final results on test set
- [ ] Submission-ready package

---

## PHASE 14: Optional Enhancements (If Time Permits)

### Advanced Features
- [ ] GAT implementation for comparison
- [ ] Anomaly Transformer baseline
- [ ] AWS CloudWatch integration (fake adapter)
- [ ] Interactive dashboard with Streamlit
- [ ] SHAP-based interpretability
- [ ] Real-time demo with simulated faults

### Additional Baselines
- [ ] CIRCA (Causal Bayesian Networks)
- [ ] RCD (Hierarchical causal discovery)
- [ ] MicroRCA implementation
- [ ] Additional RCAEval methods

### Extended Analysis
- [ ] Cross-system generalization (TrainTicket → SockShop)
- [ ] Hyperparameter sensitivity analysis
- [ ] Scalability analysis
- [ ] Attention weight analysis

---

## Risk Mitigation Strategies

### If Behind Schedule
**Priority 1 (Must Have)**:
- Metrics + Traces (skip logs if needed)
- Chronos OR TCN (not both)
- GCN (skip GAT)
- PCMCI with ParCorr (skip GPDC)
- 5 baselines minimum
- 7 ablations minimum

**Priority 2 (Should Have)**:
- All three modalities
- Both Chronos AND TCN
- 7+ baselines
- 10+ ablations

**Priority 3 (Nice to Have)**:
- GAT
- Advanced visualizations
- Interactive dashboard

### If Causal Discovery Too Slow
- [ ] Pivot to Granger-Lasso (5-min implementation)
- [ ] Use correlation-based "learned dependencies"
- [ ] Honestly document limitations

### If GNN Training Stagnates
- [ ] Fall back to NetworkX + graph features + Random Forest
- [ ] Document as "graph-based baseline"

### If Dataset Too Small for Deep Learning
- [ ] Emphasize transfer learning with Chronos zero-shot
- [ ] Use feature extraction instead of end-to-end training
- [ ] Focus on interpretability

### If Running Out of Time
- [ ] Cut multimodal complexity but maintain ablation rigor
- [ ] Professors value thoroughness over scope
- [ ] Comprehensive ablations on simpler system > complex system with weak evaluation

---

## Current Blockers

1. **User Input Needed**:
   - GPU specifications and VRAM available (assuming 8GB)
   - Exact submission deadline
   - Location of Phase 1 code (if reusing features)
   - Preferences: Chronos vs TCN priority, CatBoost usage

2. **Environment Setup**:
   - Need to verify Python environment
   - Need to test RCAEval installation
   - Need to download 4.21GB dataset

---

## Success Metrics

**Technical**:
- ✅ AC@1 > 0.50 (competitive with SOTA)
- ✅ Statistical significance over baselines (p<0.05)
- ✅ Multimodal outperforms single-modality
- ✅ Sub-second inference time

**Academic**:
- ✅ 10+ ablation configurations
- ✅ 5+ baseline comparisons
- ✅ Professional visualizations
- ✅ Publication-quality report

**Project Management**:
- ✅ Code runs on first try
- ✅ All experiments reproducible
- ✅ Documentation complete
- ✅ On-time submission

---

**Next Immediate Steps**:
1. Get user confirmation on compute resources and timeline
2. Set up Python environment
3. Install RCAEval and download dataset
4. Begin Phase 2 (EDA)
