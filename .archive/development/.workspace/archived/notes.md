# Quick Notes & Reminders

**Last Updated**: 2025-11-14 (Major Update after Materials Analysis)

## Today's Progress
- ✅ Created complete project folder structure
- ✅ Initialized all context files
- ✅ Read and analyzed all three provided documents:
  - midsem-report.txt (38K) - Phase 1 complete, identified overfitting + bottleneck issues
  - literature-review.txt (28K) - 37 papers (2020-2025), clear SOTA trends
  - research-results.txt (28K) - Strategic blueprint for A+ project
- ✅ Updated memory.md with comprehensive requirements and findings
- ✅ Updated decisions.md with 8 major architectural decisions
- ✅ Comprehensive analysis complete - ready for implementation

## Critical Insights (ACTION ITEMS)

### MUST DO 🎯
1. **Install RCAEval FIRST**: `pip install RCAEval[default]` then download 4.21GB dataset
2. **Implement BOTH Chronos AND TCN**: Not either/or - both provide excellent ablation study
3. **Start with simple baselines**: Granger-Lasso (5-min), Statistical methods BEFORE complex PCMCI
4. **Focus on Report Quality**: 60-70% of grade - ablations, baselines, visualizations matter MORE than perfect code
5. **Leverage RCAEval baselines**: 15 methods already implemented - use them!

### MUST NOT DO ❌
- Don't use LSTM-AE (obsolete in 2025, 25.4s bottleneck)
- Don't skip statistical baselines (shows lack of thoroughness)
- Don't implement GAT first (GCN simpler, similar performance on <30 nodes)
- Don't augment test data (data leakage = academic misconduct)
- Don't ignore 2024-2025 papers (literature review must be current)
- Don't cherry-pick results (report all metrics from 3-5 runs)

### Phase 1 Learnings (From Midsem Report)
- ✅ Random Forest: F1=1.00 BUT suspected overfitting (sample/feature ratio 113:1)
- ✅ LSTM-AE: F1=0.632 BUT 25.4s training time (bottleneck)
- ✅ 88-dimensional features work well (65-70% importance from rolling stats)
- ⚠️ Need to address overfitting → CatBoost with regularization
- ⚠️ Need to address latency → Replace LSTM with TCN/Chronos

## Quick Reminders
- **NEVER hallucinate** - always ask when info is missing
- **Update context files** after each major step
- **Keep /docs polished** - academic grade only
- **Code goes in /src** - organized by module
- **Experiments in /experiments** - with timestamps
- User will add bibliography entries later (just cite in text for now)
- **Report quality > Code perfection** (60-70% of grade)

## Technical Notes

### RCAEval RE2-TrainTicket Dataset (PRIMARY)
- WWW'25 and ASE 2024 benchmark (DOI: 10.5281/zenodo.14590730)
- 270 multimodal failure cases (90 per system: TrainTicket, SockShop, Online Boutique)
- **Metrics**: 77-376 metrics/case, 5-min granularity
- **Logs**: 8.6-26.9M lines, structured format (perfect for Drain3)
- **Traces**: 39.6-76.7M distributed traces with call graphs
- **Fault types**: CPU, MEM, DISK, SOCKET, DELAY, LOSS (80% of production failures)
- **Ground truth**: Root cause service AND root cause indicator
- **Size**: 4.21GB compressed
- **Installation**: `pip install RCAEval[default]`

### Key Libraries & Versions
- **PyTorch** - Primary deep learning framework
- **PyTorch Geometric** (v2.3+) - `pip install torch-geometric` (simplified in 2024)
- **tigramite** - PCMCI causal discovery (JMLR 2024)
- **causal-learn** - Granger-Lasso baseline
- **Drain3** - Log parsing (94%+ accuracy)
- **chronos** - Chronos-Bolt-Tiny foundation model (Hugging Face)
- Standard: numpy, pandas, scikit-learn, networkx

### Model Specifications

**Chronos-Bolt-Tiny**:
- 20M parameters, 100MB VRAM
- Zero-shot (no training needed)
- 250x faster inference than original Chronos
- Amazon, Nov 2024

**TCN Configuration**:
- 7 layers, kernel_size=3
- Dilation factors: [1,2,4,8,16,32,64]
- Channels: [64,128,256]
- Receptive field: 381 timesteps
- <10M parameters
- Dropout: 0.3, LR: 1e-3, Batch: 32

**GCN Configuration**:
- 2-3 layers (more = over-smoothing)
- hidden_dim=64
- dropout=0.3-0.5
- lr=0.01, weight_decay=5e-4
- Training: <1 min on CPU for small graphs

**PCMCI Hyperparameters**:
- tau_max=3-5 (fault propagation lag)
- pc_alpha=0.1-0.2 (liberal parent discovery)
- alpha_level=0.01-0.05 (conservative final graph)
- Start with ParCorr test (linear), upgrade to GPDC (nonlinear) if needed

## Evaluation Metrics (RCAEval Standard)

**RCA Metrics**:
- **AC@k** (k=1,3,5): Accuracy at k - is ground truth in top-k?
- **Avg@k**: Weighted by rank (1/rank if in top-k, else 0)
- **MRR**: Mean Reciprocal Rank

**Anomaly Detection**:
- **NAB scoring**: Standard, Reward Low FP, Reward Low FN
- **Standard ML**: F1, Precision, Recall, AUC-ROC

**Statistical Significance**:
- Mean ± std from 3-5 runs (different seeds)
- Paired t-tests (p<0.05)
- Wilcoxon signed-rank (non-normal)

## Baseline Methods (7+ Required)

1. **BARO** (FSE'24) - Bayesian online change point
2. **Random Forest** (Phase 1) - Already have
3. **Isolation Forest** (Phase 1) - Already have
4. **Statistical** (3-sigma, ARIMA) - Simple to add
5. **Granger-Lasso** - 5-min causal baseline
6. **MicroRCA** (2020) - Graph PageRank
7. **RCAEval built-in** - 15 methods available

## Ablation Studies (10+ Required)

**Data Modalities**:
1. Metrics-only
2. Logs-only
3. Traces-only
4. Metrics + Logs
5. Metrics + Traces
6. Logs + Traces
7. All modalities

**Architecture**:
8. Without GNN
9. Without PCMCI
10. Without pretrained foundation model (Chronos)
11. TCN vs Chronos comparison
12. GCN vs GAT (if implemented)

## Key Literature (Must Cite in Report)

**Foundation Models**:
- Chronos-Bolt-Tiny (Amazon, Nov 2024) - Zero-shot time series
- TimesNet (ICLR 2023) - 2D variation modeling
- Anomaly Transformer (ICLR 2022) - Association discrepancy

**Causal Inference**:
- PCMCI (Science Advances 2019, JMLR 2024) - Gold standard
- RUN (AAAI 2024) - Neural Granger + contrastive learning
- CIRCA (KDD 2022) - Causal Bayesian Networks
- RCD (NeurIPS 2022) - Hierarchical causal discovery

**Microservice RCA**:
- RCAEval (WWW'25, ASE 2024) - THE benchmark
- BARO (FSE 2024) - Bayesian RCA
- HERO (ICSE 2026) - Heterogeneous GNN
- MicroRCA (NOMS 2020) - PageRank baseline
- TraceRCA (IWQoS 2021) - Pattern mining

**Multimodal Fusion**:
- FAMOS (ICSE 2025) - Gaussian attention multimodal
- MULAN (WWW 2024) - Log language model + contrastive
- DeepTraLog (ICSE 2022) - GGNN + Deep SVDD

**GNN for Microservices**:
- Sleuth (ASPLOS 2023) - Transfer learning GNN
- Eadro (ICSE 2023) - GAT multi-task
- CausalRCA (JSS 2023) - DAG-GNN

**Surveys** (cite for background):
- GNN for Microservices (Sensors 2022)
- AIOps Solutions (arXiv 2024)

## Ideas & Explorations

**Visualizations for Report**:
- Service dependency graphs (NetworkX → Graphviz)
- Attention weight heatmaps (cross-modal attention)
- Causal graph from PCMCI
- Performance comparison charts (bar plots with error bars)
- Ablation result tables (bold best values)
- Confusion matrices for classification

**Architecture Diagrams**:
- Overall system architecture (draw.io)
- Data flow pipeline (ingestion → processing → fusion → RCA)
- Intermediate fusion architecture detail
- GNN architecture for service graphs

**Advanced (if time permits)**:
- Interactive dashboard with Streamlit
- Which modality contributes most? (attention analysis)
- Interpretability through SHAP values
- Real-time demo with simulated faults

## Common Pitfalls to Avoid

1. **Insufficient baselines** - Need 5+, not 1-2
2. **Cherry-picking results** - Report ALL runs (3-5 with different seeds)
3. **Missing ablations** - Must show component contributions
4. **Poor dataset description** - Document preprocessing thoroughly
5. **Ignoring recent papers** - Must include 2024-2025 work
6. **Unreproducible results** - Document all hyperparameters and seeds
7. **Data leakage** - NEVER augment test data
8. **Overclaiming** - Be honest about limitations

## Questions for User (Next Session)

1. **Compute Resources**: GPU model, VRAM available? (assuming 8GB)
2. **Timeline**: Exact submission deadline?
3. **Phase 1 Code**: Where is existing implementation? Reuse features?
4. **Preferences**:
   - Start with Chronos or TCN first?
   - CatBoost for metrics or different approach?
5. **AWS CloudWatch**: Required or optional?
6. **Documentation style**: Any specific format requirements?

## Implementation Timeline (4-6 Weeks)

**Week 1** - Foundation:
- Install RCAEval + download dataset
- EDA on all modalities
- Simple baselines (statistical, Granger)
- Understand ground truth format

**Weeks 2-3** - Core Development:
- Chronos-Bolt-Tiny integration
- TCN implementation
- PCMCI causal discovery
- GCN for service graphs
- Intermediate fusion architecture

**Week 4** - Experiments:
- Full ablation study (10+ configs)
- 5+ baseline comparisons
- Statistical significance tests
- Failure case analysis

**Weeks 5-6** - Documentation:
- Write methodology + results sections
- Create all visualizations
- Related work (10-15 papers)
- Code cleanup + README
- Presentation slides

## Success Criteria

✅ **Technical**:
- Working multimodal RCA system
- 10+ ablation configurations
- 5+ baseline comparisons
- Reproducible code

✅ **Academic**:
- Publication-quality report
- Professional visualizations
- Honest limitations discussion
- Current literature review (2024-2025)

✅ **Evaluation**:
- Statistical significance (p<0.05)
- Multiple metrics (AC@k, Avg@k, MRR, NAB)
- Error bars from 3-5 runs
- Comparison with RCAEval standards

---

*This file is for quick, informal notes. Structured information goes in memory.md, task_list.md, or decisions.md.*
