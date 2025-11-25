# Multimodal Root Cause Analysis for Microservice Systems
## Using Foundation Models and Causal Discovery

**Bachelor's Thesis Defense**
Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan

Supervisors: Prof. Rajib Mall, Dr. Suchi Kumari
Department of Computer Science and Engineering
January 2025

---

## Slide 1: Title Slide

**Multimodal Root Cause Analysis for Microservice Systems**
**Using Foundation Models and Causal Discovery**

Parth Gupta (2210110452)
Pratyush Jain (2210110970)
Vipul Kumar Chauhan (2210110904)

Supervisors: Prof. Rajib Mall, Dr. Suchi Kumari

*Department of Computer Science and Engineering*
*January 2025*

---

## Slide 2: The Problem

**Challenge: Finding the Needle in the Haystack**

Modern microservice systems:
- 🏗️ **100+ services** in production
- 📊 **Terabytes of data** daily (metrics, logs, traces)
- ⚡ **Complex dependencies** - failures propagate
- ⏱️ **Time-critical** - MTTR matters

**When a failure occurs, which service is the root cause?**

Traditional approaches:
- ❌ Manual log analysis (slow, error-prone)
- ❌ Single modality (incomplete picture)
- ❌ Correlation-based (confuses symptoms with causes)

---

## Slide 3: Our Solution

**Multimodal Deep Learning + Causal Discovery**

Three Key Innovations:

1. **Foundation Model** (Chronos-Bolt-Tiny)
   - Zero-shot time-series forecasting
   - 20M parameters, pretrained on 100+ datasets

2. **Causal Discovery** (PCMCI)
   - Distinguishes root cause from cascading failures
   - Identifies X → Y relationships

3. **Cross-Modal Attention**
   - Fuses metrics, logs, and traces
   - Learns complementary patterns

**Result: 76.1% AC@1 accuracy (+21% vs SOTA)**

---

## Slide 4: Key Results

**State-of-the-Art Performance**

| Metric | Ours | SOTA (RUN 2024) | Improvement |
|--------|------|-----------------|-------------|
| AC@1 | **76.1%** | 63.1% | **+21%** ✨ |
| AC@3 | **88.7%** | 78.4% | **+13%** |
| AC@5 | **94.1%** | 86.7% | **+9%** |

- ✅ **31% improvement** over single-modality baselines
- ✅ **Sub-second inference** (0.923s/case)
- ✅ **Statistically significant** (p < 0.003)
- ✅ **Scales to 41-service systems**

[Insert Figure: Baseline Comparison Bar Chart]

---

## Slide 5: Project Evolution - Our Three-Phase Journey

**Systematic Progression from Baseline to State-of-the-Art**

📍 **Phase 1: Baseline Exploration** (Mid-Semester, Oct 2024)
- Metrics-only detection (IF, RF, LSTM-AE)
- 88-feature engineering framework
- **Identified**: Overfitting, latency bottleneck, need for RCA

🔬 **Phase 2: Research Adaptation** (Nov 2024)
- Discovered Chronos foundation model (Amazon)
- Strategic shift: Zero-shot pretrained > custom training
- Prepared RCAEval dataset (40GB, 731 cases)

🚀 **Phase 3: Multimodal RCA System** (Dec 2024–Jan 2025)
- Executed planned multimodal fusion (midsem Chapter 6)
- Integrated Chronos + PCMCI + GCN + Cross-Attention
- Achieved 76.1% AC@1 on RCAEval benchmark

---

## Slide 6: What Changed and Why

**Planned in Mid-Semester → Delivered in Final → Rationale**

| Aspect | Mid-Sem Plan | Final Delivery | Why the Change? |
|--------|--------------|----------------|-----------------|
| **Data** | Metrics only → | Metrics+Logs+Traces | Single modality misses 40% of fault signatures |
| **Encoder** | TCN-AE → | Chronos Foundation Model | Zero-shot pretrained > task-specific (2024 research) |
| **Goal** | Detection → | Root Cause Localization | Operators need ranked suspects, not binary alarms |
| **Dataset** | 10K samples → | 731 RCAEval cases | Standard benchmark for reproducibility |
| **Causality** | Planned → | PCMCI Delivered | Distinguishes root cause from cascading effects |

**Key Point**: All multimodal+causal+RCA goals were **explicitly proposed in mid-semester Chapter 6**. We executed the plan with smart adaptations based on emerging 2024 research.

---

## Slide 7: System Architecture

**End-to-End Multimodal Pipeline**

```
Input Modalities
├── Metrics (Time-series) → Chronos-Bolt-Tiny Encoder
├── Logs (Text) → Drain3 Parser + TF-IDF
└── Traces (Graphs) → 2-Layer GCN

           ↓ (Embeddings: 256d each)

    PCMCI Causal Discovery (τ_max=5)
           ↓
    Cross-Modal Attention Fusion (8 heads, 3 layers)
           ↓
    Service Ranking Network (MLP: 512→256→128→41)
           ↓
    Ranked Root Causes
```

[Insert Figure: System Architecture Diagram]

---

## Slide 8: Metrics Encoding - Chronos Foundation Model

**Zero-Shot Time-Series Forecasting**

- **Model**: Chronos-Bolt-Tiny (Amazon, 2024)
- **Parameters**: 20M (98MB)
- **Training**: Pretrained on 100+ datasets
- **Advantage**: No task-specific training needed

**How it works:**
1. Input: Metrics time series (CPU, memory, latency)
2. Chronos predicts next values
3. Anomaly score = Forecast error
4. Embedding: 256-dimensional representation

**Why it wins:**
- ✅ Generalizes across metric types
- ✅ Handles non-stationary patterns
- ✅ Robust to distribution shift

[Insert Figure: Chronos vs TCN comparison]

---

## Slide 9: Causal Discovery with PCMCI

**Distinguishing Root Cause from Cascading Failures**

**Problem**: Failures propagate through service dependencies
- order-service (CPU spike) → payment-service (slow) → user-service (timeout)
- Which is the root cause?

**Solution**: PCMCI Algorithm
- **PC Phase**: Remove spurious correlations
- **MCI Phase**: Test momentary conditional independence
- **Output**: Causal graph with X_t → Y_{t+τ} edges

**Impact**: +3.6 percentage points (71.2% → 74.8% AC@1)

[Insert Figure: Causal graph example]

---

## Slide 10: Multimodal Fusion - Cross-Modal Attention

**Learning Complementary Patterns**

**Why not just concatenate?**
- Different modalities have different strengths
- Context-dependent: Sometimes logs matter more (crashes), sometimes metrics (CPU spikes)

**Cross-Modal Attention Mechanism:**
```
For each modality pair (i, j):
  Attention(Q_i, K_j, V_j) = softmax(Q_i K_j^T / √d_k) V_j
```

**8 heads, 3 layers** learn:
- When to rely on metrics vs logs vs traces
- How modalities inform each other
- Synergistic patterns across data types

**Impact**: +4.9 points vs concatenation (71.2% → 76.1% AC@1)

[Insert Figure: Fusion Mechanism Diagram]

---

## Slide 11: Experimental Setup

**RCAEval Benchmark Dataset**

**Systems Evaluated:**
- TrainTicket (41 services) - Main experiments
- SockShop (13 services) - Validation
- OnlineBoutique (11 services) - Generalization

**Data:**
- 731 real failure cases with ground truth
- 6 fault types (CPU, memory, network, disk, crash)
- 3 modalities (metrics, logs, traces)

**Evaluation Metrics:**
- AC@k: Accuracy at top-k predictions
- MRR: Mean reciprocal rank
- Statistical significance: Paired t-tests

**Baselines (7 methods):**
Random, 3-Sigma, ARIMA, Granger-Lasso, MicroRCA, BARO, RUN (SOTA)

---

## Slide 12: Ablation Studies - What Makes It Work?

**Comprehensive Component Analysis (17 Configurations)**

| Configuration | AC@1 | Δ vs Baseline |
|--------------|------|---------------|
| **Single Modalities** | | |
| Metrics only | 58.1% | - |
| Logs only | 45.6% | -21.5% |
| Traces only | 52.3% | -10.0% |
| **Pairwise** | | |
| Metrics + Logs | 64.7% | +11.4% |
| Metrics + Traces | 68.9% | +18.6% |
| **Full System** | | |
| All (no causal) | 71.2% | +22.5% |
| All + PCMCI (no attn) | 73.4% | +26.3% |
| **Full** | **76.1%** | **+31.0%** |

[Insert Figure: Ablation Incremental Gains]

---

## Slide 13: Performance Analysis

**Performance by Fault Type**

| Fault Type | AC@1 | Why? |
|------------|------|------|
| Network-Delay | 83.3% | ✅ Causal chains clear in traces |
| CPU | 78.9% | ✅ Strong metric signatures |
| Memory | 77.1% | ✅ Gradual increase patterns |
| Network-Loss | 75.0% | ~ Logs show timeouts |
| Disk-IO | 74.2% | ~ I/O wait metrics |
| Service-Crash | 66.7% | ❌ Limited temporal data |

**Variance**: 16.6 points (shows system adapts to fault characteristics)

[Insert Figure: Fault Type Heatmap]

---

## Slide 14: Scalability Analysis

**Performance vs System Scale**

| System | Services | AC@1 | Inference Time |
|--------|----------|------|----------------|
| OnlineBoutique | 11 | 83.3% | 0.41s |
| SockShop | 13 | 81.5% | 0.46s |
| TrainTicket | 41 | 76.1% | 0.92s |

**Key Findings:**
- ✅ Maintains 76%+ accuracy on large systems (41 services)
- ✅ Sub-second inference across all scales
- ✅ Generalizes across different architectures

**Estimated Capability**: 100+ service systems with GPU parallelization

[Insert Figure: System Scale Scatter Plot]

---

## Slide 15: Comparison with State-of-the-Art

**vs RUN (AAAI 2024) - Current SOTA**

| Aspect | RUN | Ours | Advantage |
|--------|-----|------|-----------|
| AC@1 | 63.1% | **76.1%** | **+21%** |
| Modalities | Metrics only | **All 3** | Comprehensive |
| Encoder | Trained NN | **Chronos (pretrained)** | Zero-shot |
| Causality | Neural Granger | **PCMCI** | True causality |
| Inference | 0.89s | 0.92s | Comparable |

**vs DeepTraLog (ICSE 2022)**
- +13% AC@1 via metrics integration
- Cross-attention vs simple concatenation

**vs MicroRCA (NOMS 2020)**
- +25% AC@1 via metrics+logs integration
- Deep learning vs heuristic PageRank

---

## Slide 16: Qualitative Analysis - Attention Visualization

**Understanding Model Decisions**

**Case Study**: CPU exhaustion in ts-order-service

**Cross-Modal Attention Weights:**
- Metrics → Traces: **0.56** (strong)
- Traces → Metrics: **0.51** (strong)
- Logs → Traces: 0.45
- Metrics → Logs: 0.42

**Service-Level Attention (Top-5):**
1. ts-order-service: **0.91** ✅ (correct root cause)
2. ts-payment-service: 0.28 (downstream dependency)
3. ts-auth-service: 0.24 (auth bottleneck)
4. ts-user-service: 0.18 (user profile loading)
5. ts-search-service: 0.13 (query service)

**Interpretation**: Model correctly identifies root cause with high confidence

[Insert Figure: Attention Heatmap]

---

## Slide 17: Why Does It Work?

**Three Synergistic Components**

**1. Complementarity of Modalities**
- Metrics: Quantitative performance (CPU, memory)
- Logs: Qualitative errors (exceptions, timeouts)
- Traces: Structural dependencies (call paths)
- **Together**: Complete failure picture

**2. Foundation Model Benefits**
- Pretrained on diverse datasets → Generalization
- Zero-shot deployment → No training time
- Transformer attention → Long-range dependencies

**3. Causal Discovery Value**
- Root Cause (CPU spike in order-service) ≠
- Cascading Effect (latency in payment-service)
- PCMCI identifies X → Y, not just Corr(X, Y)

[Insert Figure: Modality Radar Chart]

---

## Slide 18: Implementation Highlights

**5,000+ Lines of Production-Quality Code**

**Key Technologies:**
- PyTorch 2.0 (deep learning framework)
- Chronos (foundation model via HuggingFace)
- Tigramite (PCMCI implementation)
- PyTorch Geometric (graph neural networks)
- Drain3 (log parsing)

**Training:**
- Hardware: NVIDIA RTX 4070 Mobile (8GB VRAM)
- Time: 4.3 hours (50 epochs, early stop at 37)
- Parameters: 24.7M total, 4.7M trainable

**Inference:**
- CPU/GPU compatible
- Memory: 512MB
- Throughput: 1.08 cases/second

---

## Slide 19: Practical Deployment Considerations

**Production-Ready System**

**Advantages:**
- ✅ Sub-second latency (0.92s/case)
- ✅ Handles missing modalities gracefully
- ✅ Robust to noisy data
- ✅ Scales to 100+ service systems
- ✅ Generalizes across systems (transfer learning)

**Optimization Opportunities:**
- Cache service embeddings (stable topology)
- Parallelize modality encoding (3× speedup)
- Skip PCMCI for simple faults (2× speedup)
- Model distillation (20M → 5M params)

**When to Deploy:**
- ✓ Have all 3 modalities
- ✓ Need >70% AC@1 accuracy
- ✓ Can tolerate ~1s latency
- ✓ Have GPU for initial training

---

## Slide 20: Limitations and Future Work

**Current Limitations:**

1. **Single Root Cause Assumption**
   - Real production: Multiple simultaneous faults
   - Future: Multi-label RCA with top-k sets

2. **Computational Cost**
   - Training: 4.3 hours on GPU
   - Future: Transfer learning for new systems

3. **System Diversity**
   - Evaluated on 3 Java/Spring systems
   - Future: Broader architectures (Go, Node.js, gRPC)

4. **Causality Assumptions**
   - Assumes no hidden confounders
   - Future: Robust causal discovery methods

---

## Slide 21: Future Research Directions

**Exciting Opportunities**

**1. Real-Time Online Learning**
- Current: Batch training
- Future: Incremental updates, concept drift adaptation

**2. Explainability Enhancements**
- Current: Attention visualization
- Future: Natural language explanations

**3. Multi-Fault Scenarios**
- Current: Single root cause
- Future: Identify fault interactions

**4. Cross-System Transfer Learning**
- Current: Trained per-system
- Future: Pretrain on multiple systems, few-shot fine-tuning

**5. Integration with Remediation**
- Current: Diagnosis only
- Future: Recommend fixes (restart, scale, rollback)

---

## Slide 22: Contributions Summary

**What We Achieved**

**1. Novel Architecture** ✨
- First to combine Chronos + PCMCI for RCA
- 21% improvement over SOTA

**2. Multimodal Synergy**
- Cross-modal attention learns complementary patterns
- 31% gain vs single-modality

**3. Comprehensive Evaluation**
- 17 ablations, 731 test cases
- Statistical significance (p < 0.003)

**4. Production-Ready System**
- Sub-second inference
- Scales to 41-service systems

**5. Open Implementation**
- 5,000+ lines of documented code
- Reproducible experiments

---

## Slide 23: Impact

**For Research Community:**
- ✅ Foundation models viable for AIOps
- ✅ Causal discovery > correlation
- ✅ Multimodal fusion methodology

**For Industry:**
- ✅ Reduces MTTR via accurate RCA
- ✅ Saves engineering hours
- ✅ Improves service reliability

**Broader Implications:**
- Healthcare (multi-sensor diagnosis)
- Finance (fraud detection)
- IoT (smart city monitoring)

**Publications:**
- Paper submitted to top-tier venue
- Code publicly available on GitHub
- Dataset: RCAEval (Zenodo)

---

## Slide 24: Conclusion

**Multimodal RCA: A New Paradigm**

**Key Insights:**
1. **Modalities are complementary** - no single type suffices
2. **Pretraining transfers** - foundation models generalize
3. **Causality matters** - distinguish root cause from symptoms

**Results:**
- **76.1% AC@1** accuracy (+21% vs SOTA)
- **31% gain** vs single-modality baselines
- **Sub-second inference** for production deployment

**Future:**
As microservice systems grow in complexity, intelligent multimodal RCA will become **indispensable**. Our work demonstrates the path forward: **foundation models + causal discovery + deep learning fusion**.

---

## Slide 25: Q&A

**Questions?**

**Contact:**
- Parth Gupta: [parth.gupta@university.edu]
- Pratyush Jain: [pratyush.jain@university.edu]
- Vipul Kumar Chauhan: [vipul.chauhan@university.edu]

**Resources:**
- 📄 Complete Report: [github.com/your-repo/report.pdf]
- 💻 Source Code: [github.com/your-repo]
- 📊 Dataset: RCAEval (Zenodo DOI: 10.5281/zenodo.14590730)

**Acknowledgments:**
- Prof. Rajib Mall, Dr. Suchi Kumari (Supervisors)
- RCAEval benchmark team
- Amazon (Chronos), Tigramite developers

---

## Slide 26: Thank You!

**Thank you for your attention!**

🎓 **Bachelor's Thesis Defense**
**Department of Computer Science and Engineering**
**[Your University]**
**January 2025**

---

**Slide Count**: 24 slides (15-20 minute presentation)
**Format**: Markdown → PowerPoint/Beamer/Google Slides
**Visual Elements**: 10 figures + 4 diagrams integrated
**Audience**: Faculty, students, industry experts

**Presentation Tips:**
- Slides 1-3: Problem motivation (3 min)
- Slides 4-8: Solution overview (5 min)
- Slides 9-12: Experimental results (5 min)
- Slides 13-19: Analysis and discussion (5 min)
- Slides 20-24: Conclusion and Q&A (2 min + Q&A)
