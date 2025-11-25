# Decision Log

All architectural and strategic decisions made during the project.

---

## Reorganization Phase (2025-11-14)

### D001: Workspace Location
- **Date**: 2025-11-14
- **Status**: ✅ Implemented
- **Decision**: Use `.workspace/` (hidden folder) for all working notes
- **Alternatives Considered**:
  - `workspace/` (visible folder)
  - Keep in `project/context/`
- **Rationale**:
  - Hidden folder keeps project root clean
  - Follows Unix convention for metadata
  - Clear separation from production code
  - Won't be accidentally committed if gitignore updates
- **Impact**: Low risk, high organization benefit

### D002: Docs Folder Policy
- **Date**: 2025-11-14
- **Status**: ✅ Implemented
- **Decision**: `docs/` contains ONLY formal, polished, publication-ready documentation
- **Alternatives Considered**:
  - Mix working notes with formal docs
  - Separate into `docs/formal/` and `docs/notes/`
- **Rationale**:
  - User explicitly requested separation
  - Easier to generate final deliverables
  - Professional appearance for reviewers
  - Working notes belong in `.workspace/`
- **Examples of Formal Docs**:
  - ARCHITECTURE.md (system design)
  - API_REFERENCE.md (interface documentation)
  - EVALUATION.md (methodology)
- **Examples of Working Notes** (go to .workspace/):
  - Experiment logs, scratch notes, brainstorming
- **Impact**: Medium - requires discipline to maintain

### D003: Academic Reference Storage
- **Date**: 2025-11-14
- **Status**: 🔄 In Progress
- **Decision**: Move `/context` academic docs to `reference/` directory
- **Alternatives Considered**:
  - Keep at root level
  - Move to `.workspace/`
  - Delete (bad idea)
- **Rationale**:
  - These are permanent reference materials (literature review, midsem report)
  - Not working notes (shouldn't be in .workspace)
  - Not active code (shouldn't be in project/)
  - Need to preserve for citations and context
- **Files to Move**:
  - `context/literature-review.txt` → `reference/`
  - `context/midsem-report.txt` → `reference/`
  - `context/research-results.txt` → `reference/`
- **Impact**: Low - just organization

### D004: Source Code Organization
- **Date**: 2025-11-14
- **Status**: 🔄 In Progress
- **Decision**: Create domain-specific folders in `src/` for each major component
- **Alternatives Considered**:
  - Flat structure (everything in `src/`)
  - Feature-based (e.g., `src/phase3/`, `src/phase4/`)
- **Rationale**:
  - Scalability for Phases 3-8
  - Clear separation of concerns
  - Easier imports and testing
  - Industry best practice
- **New Structure**:
  ```
  src/
  ├── data/          # Data loading, preprocessing
  ├── encoders/      # Metrics, logs, traces encoders
  ├── causal/        # PCMCI, causal discovery
  ├── fusion/        # Multimodal fusion
  ├── models/        # RCA models
  ├── evaluation/    # Metrics, ablations
  ├── baselines/     # Existing baselines
  └── utils/         # General utilities
  ```
- **Impact**: Medium - affects imports, but clean structure worth it

### D005: Configuration Management
- **Date**: 2025-11-14
- **Status**: 🔄 In Progress
- **Decision**: Use YAML files in `config/` for all hyperparameters
- **Alternatives Considered**:
  - Hardcoded in Python
  - Command-line arguments only
  - JSON files
- **Rationale**:
  - YAML more readable than JSON
  - Easier to version control configs
  - Reproducibility critical for research
  - Industry standard in ML projects
- **Config Files**:
  - `model_config.yaml` - Architecture hyperparameters
  - `experiment_config.yaml` - Training settings
  - `data_config.yaml` - Dataset paths and splits
- **Impact**: Low risk, high reproducibility benefit

### D006: Output Directory Structure
- **Date**: 2025-11-14
- **Status**: 🔄 In Progress
- **Decision**: Gitignore `outputs/` with structured subdirectories
- **Alternatives Considered**:
  - Save outputs in `data/`
  - Save outputs in project root
- **Rationale**:
  - Clear separation of code and results
  - Easy to clean up experiments
  - Won't bloat git repository
  - Organized by type (models, results, figures)
- **Structure**:
  ```
  outputs/
  ├── models/        # Saved checkpoints
  ├── results/       # CSV/JSON metrics
  ├── figures/       # Visualizations
  └── logs/          # Training logs
  ```
- **Impact**: Low - standard practice

---

## Technical Decisions (From Previous Context)

### D007: Metrics Encoder Choice
- **Date**: 2025-11-14 (from memory.md)
- **Status**: 📋 Planned
- **Decision**: Implement BOTH Chronos-Bolt-Tiny (primary) AND TCN (comparison)
- **Alternatives Considered**:
  - LSTM-AE (rejected - obsolete, 25.4s latency)
  - Transformer (too heavy for 8GB VRAM)
  - Only Chronos (no ablation)
- **Rationale**:
  - Chronos: Zero-shot, 100MB, 250x faster than LSTM
  - TCN: Trained model, parallelizable, 80% faster than LSTM
  - Having both enables encoder ablation study
  - Literature shows TCN competitive with transformers
- **Research Evidence**:
  - Chronos-Bolt-Tiny: Amazon Nov 2024, 20M params
  - TCN: Proven 3-5x faster training, comparable F1 to LSTM
- **Impact**: High - critical for meeting latency requirements

### D008: Causal Discovery Method
- **Date**: 2025-11-14 (from memory.md)
- **Status**: 📋 Planned
- **Decision**: Use PCMCI/PCMCIplus from tigramite library
- **Alternatives Considered**:
  - PC algorithm (order-dependent, no temporal modeling)
  - Granger causality (linear only, less powerful)
  - NOTEARS (static DAGs, no time)
  - Neural methods (harder to implement, less interpretable)
- **Rationale**:
  - Gold standard for time series causal discovery
  - Explicitly handles autocorrelation
  - Detection power >80% in high-dimensional cases
  - ASE 2024 evaluation confirms effectiveness
  - Excellent documentation (JMLR 2024)
- **Hyperparameters**:
  - `tau_max=5` (5-minute fault propagation)
  - `pc_alpha=0.15` (liberal parent discovery)
  - `alpha_level=0.05` (conservative edges)
- **Impact**: High - critical for causal RCA component

### D009: GNN Architecture
- **Date**: 2025-11-14 (from memory.md)
- **Status**: 📋 Planned
- **Decision**: Start with 2-layer GCN, upgrade to GAT only if needed
- **Alternatives Considered**:
  - GAT from start (more complex, minimal benefit for small graphs)
  - Heterogeneous GNN (overkill for current scope)
  - Skip GNN entirely (misses service topology)
- **Rationale**:
  - GCN simpler to implement and debug
  - 2022 survey shows GCN sufficient for 10-30 node microservices
  - GCN memory: 5-10MB vs GAT: 10-20MB
  - Can upgrade to GAT in ablation if needed
  - PyG 2.3+ makes both easy to implement
- **Architecture**:
  - 2-3 layers (more causes over-smoothing)
  - Hidden dim: 64-128
  - Dropout: 0.3-0.5
- **Impact**: Medium - affects trace encoding quality

### D010: Fusion Strategy
- **Date**: 2025-11-14 (from memory.md)
- **Status**: 📋 Planned
- **Decision**: Intermediate fusion with cross-modal attention
- **Alternatives Considered**:
  - Early fusion (concatenate raw features)
  - Late fusion (separate predictions, ensemble)
- **Rationale**:
  - Literature shows intermediate fusion best for M+L+T
  - Handles heterogeneous sampling rates
  - Learns modality importance dynamically
  - FAMOS (ICSE 2025) and MULAN (WWW 2024) use this
- **Architecture**:
  ```
  Separate encoders → Cross-attention → Combined representation → RCA
  ```
- **Ablation**: Compare all three fusion types
- **Impact**: High - core of multimodal approach

---

## Process Decisions

### D011: Development Strategy
- **Date**: 2025-11-14
- **Status**: 📋 Planned
- **Decision**: Incremental implementation (Metrics → +Logs → +Traces)
- **Alternatives Considered**:
  - Build full pipeline then test
  - Random order
- **Rationale**:
  - Creates natural ablation story
  - Easier to debug incrementally
  - Each step provides publishable results
  - Matches Phase 3-8 structure
- **Timeline**:
  - Week 1: Metrics-only working
  - Week 2: +Logs working
  - Week 2: +Traces working
  - Week 3: Full fusion
- **Impact**: Medium - affects development flow

### D012: Evaluation Framework
- **Date**: 2025-11-14
- **Status**: 📋 Planned
- **Decision**: Match RCAEval standard metrics (AC@k, Avg@k, MRR)
- **Alternatives Considered**:
  - Only accuracy
  - Custom metrics
- **Rationale**:
  - RCAEval benchmark standard
  - Direct comparison with paper results
  - Industry-accepted metrics for RCA
  - Comprehensive ranking evaluation
- **Metrics**:
  - AC@1, AC@3, AC@5 (accuracy at top-k)
  - Avg@k (position-weighted accuracy)
  - MRR (mean reciprocal rank)
  - Statistical tests (paired t-test, p<0.05)
- **Impact**: Low - standard practice

---

## Open Decisions (Need User Input)

### D013: Novel Contribution
- **Date**: 2025-11-14
- **Status**: ❓ Awaiting User Input
- **Question**: What is the "new technique" to beat/match SOTA?
- **Options**:
  1. Novel fusion architecture
  2. Novel causal integration
  3. Comprehensive SOTA combination (Chronos+PCMCI+GCN)
  4. Something else?
- **Why It Matters**: Affects how we frame contributions in report
- **Impact**: HIGH - determines novelty claim

### D014: Baseline Strategy
- **Date**: 2025-11-14
- **Status**: ❓ Awaiting User Input
- **Question**: Use RCAEval paper baseline results or re-implement?
- **Options**:
  1. Use published results directly (faster)
  2. Re-implement all baselines (more thorough)
  3. Hybrid (use some, implement some)
- **Why It Matters**: Affects Week 4 timeline
- **Impact**: Medium - time vs thoroughness tradeoff

### D015: Implementation Scope
- **Date**: 2025-11-14
- **Status**: ❓ Awaiting User Input
- **Question**: Incremental evaluation or full pipeline first?
- **Options**:
  1. Incremental: Metrics → +Logs → +Traces (3 eval cycles)
  2. Full: Build everything → Evaluate once
- **Why It Matters**: Affects ablation story quality
- **Impact**: Medium - affects development approach

---

## Decision Template

```markdown
### DXXX: Decision Title
- **Date**: YYYY-MM-DD
- **Status**: ✅ Implemented | 🔄 In Progress | 📋 Planned | ❓ Awaiting Input
- **Decision**: What was decided
- **Alternatives Considered**: What else was considered
- **Rationale**: Why this decision was made
- **Impact**: High | Medium | Low
- **Evidence**: Research papers, benchmarks, etc.
```

---

**Keep this log updated for every significant decision!**

## Strategic Decisions (User Confirmed - 2025-11-14)

### D013: Novel Contribution ✅ DECIDED
- **Date**: 2025-11-14
- **Status**: ✅ Confirmed by user
- **Decision**: Comprehensive SOTA Combination (Option C)
- **Specifics**: Chronos + PCMCI + GCN with intermediate fusion
- **Framing**: "First comprehensive integration of foundation models, causal discovery, and graph learning for multimodal microservice RCA"
- **Impact**: HIGH - shapes report narrative as integration novelty, not algorithmic novelty

### D016: Dataset Confirmed ✅ VERIFIED  
- **Date**: 2025-11-14
- **Status**: ✅ Exists locally
- **Details**: 3 systems, 3 RE versions, 731 cases loaded

### D017: Implementation Strategy ✅ DECIDED
- **Date**: 2025-11-14  
- **Status**: ✅ Confirmed (Option A - Incremental)
- **Plan**: Metrics → +Logs → +Traces → Full fusion (4 evaluation cycles)

### D018: Baselines ✅ DECIDED
- **Date**: 2025-11-14
- **Status**: ✅ Hybrid approach (Option C)
- **Details**: Phase 2 baselines + 1-2 re-implementations + cite published

### D019: Compute ✅ CONFIRMED
- **Date**: 2025-11-14
- **Status**: ✅ Can run 12-hour experiments, 3-5 seeds
