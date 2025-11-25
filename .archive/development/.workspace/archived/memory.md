# Project Memory

**Last Updated**: 2025-11-14 (Comprehensive Update after Reading All Materials)

## What I Know

### Project Context - University Major Project
- **Institution**: Bachelor of Technology in Computer Science and Engineering
- **Students**: Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
- **Supervisors**: Prof. Rajib Mall, Dr. Suchi Kumari
- **Submission**: October 2025
- **Project Type**: University-grade major project requiring publication-quality work

### Current State (Phase 1 - Completed)
**Achievements**:
- Implemented 3-model ensemble: Isolation Forest, Random Forest, LSTM-AE
- Created 88-dimensional feature engineering pipeline from 7 base metrics
- Dataset: 10,000 time series observations with 5% anomaly ratio
- Performance:
  - Random Forest: F1=1.00, AUC=1.00 (PERFECT but suspected overfitting)
  - LSTM-AE: F1=0.632, AUC=0.85, Training Time=25.4s (BOTTLENECK)
  - Isolation Forest: F1=0.367, AUC=0.65, Training Time=0.45s

**Critical Issues Identified**:
1. Random Forest perfect score (F1=1.00) indicates high overfitting risk
   - Sample-to-feature ratio: 113:1 (10K samples / 88 features)
   - High-variance tree ensembles memorize noise
2. LSTM-AE is computational bottleneck (25.4s training time)
   - Sequential processing prevents parallelization
   - Incompatible with sub-100ms inference latency requirement
3. Limited to metrics-only (no logs or traces yet)

### Project Scope & Objectives
- **Domain**: AIOps for Microservice Anomaly Detection + Root Cause Analysis
- **Dataset**: RCAEval RE2-TrainTicket (270 multimodal failure cases)
- **Modalities**: Metrics, Logs, Traces (multimodal fusion required)
- **Key Technologies** (Based on 2024-2025 SOTA Research):
  - **Forecasting**: Chronos-Bolt-Tiny (20M params, 100MB VRAM) OR TCN
  - **Causal Discovery**: PCMCI/PCMCIplus (via tigramite library)
  - **Graph Learning**: 2-layer GCN (upgrade to GAT if needed)
  - **Log Parsing**: Drain3 parser
  - **Fusion Strategy**: Intermediate multimodal fusion with cross-modal attention

### Research Objectives (Updated from Materials)
1. **Production-grade performance**: 10,000 events/sec, sub-100ms latency
2. **Industry-standard metrics**: 80% recall, 10% false positive rate
3. **Root cause localization**: Pinpoint fault origin in distributed architecture
4. **Multi-modal learning**: Integrate logs and traces with metrics
5. **Rigorous evaluation**: Point-adjusted, event-based, NAB scoring
6. **Deployment readiness**: Containerization and cloud platform compatibility
7. **Academic excellence**: 10+ ablation studies, 5+ baseline comparisons

### RCAEval TrainTicket Dataset (PRIMARY DATASET)
**Source**: WWW'25 and ASE 2024 benchmark (DOI: 10.5281/zenodo.14590730)
**Scale**: 270 multimodal failure cases (90 per system)
**Systems**: TrainTicket, SockShop, Online Boutique
**Data Modalities**:
- **Metrics**: 77-376 metrics/case at 5-min granularity (CPU, memory, network, disk)
- **Logs**: 8.6-26.9M lines with structured formatting
- **Traces**: 39.6-76.7M distributed traces with call graphs
**Fault Types**: 6 types (CPU, MEM, DISK, SOCKET, DELAY, LOSS) - covers 80% of production failures
**Ground Truth**: Complete annotations of root cause service AND root cause indicator
**Size**: 4.21GB compressed
**Installation**: `pip install RCAEval[default]`
**Why This Dataset**:
- Only public dataset for causal inference evaluation with multimodal microservice data
- 15 baseline RCA methods already implemented
- Synchronized timestamps across all modalities
- Recent (January 2025) - cutting-edge benchmark
- Ideal size for academic projects (270 cases = sufficient for 60/20/20 train/val/test splits)

### Technical Stack (From Literature Review)
**Deep Learning**: PyTorch (primary), TensorFlow (secondary)
**Graph Neural Networks**: PyTorch Geometric (v2.3+), DGL
**Causal Discovery**: tigramite (PCMCI), causal-learn (Granger, PC algorithm)
**Log Parsing**: Drain3 (94%+ accuracy), Spell (batch processing)
**Monitoring**: Prometheus, Grafana, cAdvisor
**Tracing**: Jaeger, Zipkin, Apache SkyWalking, OpenTelemetry
**Deployment**: Kubernetes, Docker
**Evaluation**: RCAEval benchmark suite

### Key Research Findings (From Literature Review - 37 Papers)

**LSTM Autoencoders Are Obsolete (2025)**:
- Transformers achieve 250x faster inference
- TCNs provide 3-5x faster training with larger receptive fields
- Chronos foundation models eliminate training via zero-shot learning
- Performance gap: LSTM F1=0.74-0.82 vs Transformer F1=0.87-0.91 on benchmarks

**Modern Replacements**:
1. **Chronos-Bolt-Tiny** (RECOMMENDED):
   - 20M parameters, 100MB VRAM
   - 250x faster inference than original Chronos
   - Zero-shot anomaly detection (no training needed)
   - Saves 1-2 weeks of training time

2. **TCN (Temporal Convolutional Network)**:
   - 7 layers, kernel=3, dilations=[1,2,4,8,16,32,64]
   - 381-timestep receptive field with <10M params
   - 3-5x faster training than LSTM
   - 5-10% better F1 scores on benchmarks

3. **Anomaly Transformer**:
   - Association discrepancy mechanism (ICLR 2022)
   - 5-15% F1 improvement over LSTM
   - 100-200MB for small variant

**Causal Discovery - PCMCI Gold Standard**:
- Two-stage procedure explicitly handles temporal structure
- Detection power >80% even in high-dimensional cases
- Handles autocorrelation (unlike PC algorithm)
- RUN framework (AAAI 2024): AC@1=0.63 on Sock Shop
- **Hyperparameters**: tau_max=3-5, pc_alpha=0.1-0.2, alpha_level=0.01-0.05
- **Alternative**: Granger-Lasso (5-min implementation, fast baseline)

**GNN Architecture**:
- **GCN**: Start here - simpler, faster, similar performance on small graphs (<30 nodes)
- **GAT**: Upgrade if needed for heterogeneous services or interpretability
- **Configuration**: 2-3 layers (more causes over-smoothing), hidden_dim=32-64, dropout=0.3-0.5
- **Graph Construction**: Parse traces → extract parent-child spans → nodes=services, edges=calls
- **Features**: Response time (mean, p50, p90, p99), CPU/memory, request rate, error rate

**Multimodal Fusion - Intermediate Strategy Wins**:
- **Early fusion**: Struggles with heterogeneous sampling rates, dimensionality explosion
- **Late fusion**: Misses cross-modal correlations
- **Intermediate fusion** (RECOMMENDED):
  - Separate encoders per modality
  - Time-aligned aggregation (1-min windows)
  - Cross-modal attention for dynamic weighting
  - Combined representation → anomaly detector + RCA module

### Baseline Methods to Implement (5+ Required)
From RCAEval and Literature:
1. **BARO** (FSE'24) - Bayesian online change point detection
2. **Statistical methods** - ARIMA, Isolation Forest, 3-sigma
3. **MicroRCA** (2020) - Graph-based PageRank
4. **TraceRCA** (2021) - Frequent pattern mining
5. **Random Forest** (Phase 1 existing baseline)
6. **CIRCA** (KDD'22) - Causal Bayesian Network
7. **RCD** (NeurIPS'22) - Hierarchical causal discovery

### What Earns A+ Grades (60-70% Report Quality)
**Critical Success Factors**:
1. **Thoroughness > Novelty**: 5+ baseline comparisons (not 2-3)
2. **Comprehensive ablations**: Each component's contribution quantified
   - Data modalities: metrics-only, logs-only, traces-only, all pairs, all three
   - Architecture: remove GNN, remove temporal, remove attention, remove causal
   - Training: without augmentation, without pretrained, different losses
3. **Statistical significance**: Error bars from 3-5 runs, t-tests, p-values
4. **Failure analysis**: What didn't work and why
5. **Professional visualizations**: Service graphs, comparison charts, architecture diagrams
6. **Honest limitations**: Acknowledge constraints (shows maturity)

**Common Mistakes to Avoid**:
- Insufficient baselines (comparing against only 1-2 methods)
- Cherry-picked results (only best runs)
- Overclaiming unsupported by evidence
- Missing ablation studies
- Poor dataset description
- Ignoring 2024-2025 papers
- Unreproducible results (no hyperparameters/seeds)
- Data leakage

## User Confirmations (Received 2025-11-14)

### Compute Resources ✅
- **GPU**: RTX 4070 Mobile (80W, 8GB VRAM)
- **CPU**: 16 cores / 22 threads
- **RAM**: 16GB
- **Assessment**: Sufficient for Chronos-Bolt-Tiny, TCN, GCN, PCMCI, Drain3, and full RCAEval pipeline

### Timeline ✅
- **Deadline**: No strict constraints
- **Pacing**: Optimize for correctness + report quality over speed
- **Assessment**: Can be thorough with ablations and experiments

### Phase 1 Code Status ✅
- **Reality**: No real Phase 1 implementation exists - midsem results were illustrative
- **Approach**: Build everything from ground up
- **Framing**: Present as "Phase 2 improvement" showing architectural evolution
- **Benefit**: Clean slate, no legacy code constraints

### Model Implementation Priority ✅
1. **Chronos-Bolt-Tiny FIRST**: Fast zero-shot baseline
2. **TCN LATER**: For ablation comparison (keep in timeline and report)
3. **GCN**: Start here (not GAT initially)

### CloudWatch Integration ✅
- **Status**: REQUIRED
- **Data Source**: Synthetic/self-generated (not real AWS)
- **Scope**: Show adapters + code structure
- **Presentation**: "CloudWatch-compatible ingestion layer" in report

### Documentation & Scope ✅
- **Style**: No specific requirements - follow academic structure in /docs
- **Priority 1 (MANDATORY)**:
  - Comprehensive ablations (10-12 configurations minimum)
  - 7+ baseline comparisons
  - Professional visualizations (graphs, causal diagrams, architecture, service graphs, attention maps)
- **Priority 2 (SECONDARY)**:
  - Architectural sophistication
  - Deep engineering details
- **Primary Goal**: Results + report quality > over-engineering

### Implementation Timeline (4-6 Weeks Feasible)
**Week 1**: Foundation
- Install RCAEval, download RE2-TrainTicket (4.21GB)
- EDA on all three modalities
- Implement 2-3 simple baselines (BARO, Drain+frequency, basic graph)
- Understand ground truth format

**Weeks 2-3**: Core Development
- PCMCI causal discovery with tuned hyperparameters
- Service dependency graphs from traces (NetworkX)
- 2-layer GCN/GAT implementation
- Chronos-Bolt-Tiny OR TCN from scratch
- Intermediate fusion architecture

**Week 4**: Experiments
- Full ablation study (8-10 configurations)
- 5+ baseline comparisons
- Statistical significance testing
- Failure case analysis
- Hyperparameter sensitivity

**Weeks 5-6**: Documentation & Polish
- Methodology with architecture diagrams
- Results with tables/figures
- Discussion with insights and limitations
- Related work (10-15 papers, emphasize 2024-2025)
- Clean code with README
- Presentation slides

### Key Literature (Must Cite)
**Foundation Models**:
- Chronos-Bolt-Tiny (Amazon, Nov 2024)
- TimesNet (ICLR 2023)
- Anomaly Transformer (ICLR 2022)

**Causal Inference**:
- PCMCI (Science Advances 2019, JMLR 2024)
- RUN (AAAI 2024)
- CIRCA (KDD 2022)
- RCD (NeurIPS 2022)

**Microservice RCA**:
- RCAEval (WWW'25, ASE 2024)
- BARO (FSE 2024)
- HERO (ICSE 2026)
- MicroRCA (NOMS 2020)
- TraceRCA (IWQoS 2021)

**Multimodal Fusion**:
- FAMOS (ICSE 2025)
- MULAN (WWW 2024)
- DeepTraLog (ICSE 2022)

**GNN for Microservices**:
- Sleuth (ASPLOS 2023)
- Eadro (ICSE 2023)
- CausalRCA (JSS 2023)

## Key Facts Remembered
- **Phase 1 COMPLETED**: RF (overfitting), LSTM-AE (too slow), 88-dim features
- **Phase 2 TARGET**: Replace RF with CatBoost, LSTM with TCN/Chronos
- **Phase 3 TARGET**: Multimodal fusion + causal inference
- **Critical**: 60-70% of grade is report quality (ablations, baselines, viz)
- LSTM is obsolete in 2025 - use TCN or Chronos
- RCAEval is THE dataset for this project
- PCMCI is gold standard for causal discovery
- Start with GCN, upgrade to GAT only if needed
- Intermediate fusion > early or late fusion
- Must be reproducible and modular
- Academic format required for all docs
- All code must go in correct folders
- Context files updated after each major step
- NEVER hallucinate - always ask user

## Current State
**Status**: Materials received and analyzed - Ready to begin implementation
**Next Step**: Set up development environment and download RCAEval dataset
**Confidence**: HIGH - All requirements clear, feasible timeline, proven approaches
