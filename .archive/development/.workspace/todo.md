# Task Tracking

**Last Updated**: 2025-11-14

---

## 🔥 Current Sprint: Codebase Reorganization

### In Progress
- [ ] Create complete `.workspace/` structure
  - [x] memory.md
  - [x] context.md
  - [x] todo.md (this file)
  - [ ] decisions.md
  - [ ] notes.md
  - [ ] experiments/ directory

### Pending - Reorganization
- [ ] Create new `src/` module structure
  - [ ] `src/data/` (move from utils)
  - [ ] `src/encoders/` with placeholders
  - [ ] `src/causal/` with placeholders
  - [ ] `src/fusion/` with placeholders
  - [ ] `src/models/` with placeholders
  - [ ] `src/evaluation/` with placeholders
  - [ ] Update `src/__init__.py`

- [ ] Move academic references
  - [ ] Create `reference/` directory
  - [ ] Move `/context/*.txt` → `reference/`
  - [ ] Move `ENVIRONMENT.md` → `reference/`
  - [ ] Move `TESTING.md` → `reference/`
  - [ ] Archive `project/context/*.md` → `.workspace/archived/`

- [ ] Create configuration system
  - [ ] `project/config/` directory
  - [ ] `config/model_config.yaml` template
  - [ ] `config/experiment_config.yaml` template
  - [ ] `config/data_config.yaml` template

- [ ] Set up outputs structure
  - [ ] `project/outputs/` directory
  - [ ] `outputs/models/` for checkpoints
  - [ ] `outputs/results/` for metrics
  - [ ] `outputs/figures/` for visualizations
  - [ ] `outputs/logs/` for training logs

- [ ] Create project setup
  - [ ] `project/setup.py` for pip installation
  - [ ] Update `project/.gitignore`
  - [ ] Update `project/requirements.txt` if needed

- [ ] Update documentation
  - [ ] Update main README with new structure
  - [ ] Create `docs/ARCHITECTURE.md` (formal)
  - [ ] Update `docs/MODULE_INTERFACES.md` if needed
  - [ ] Create migration guide for new structure

### Completed
- [x] Analyze current codebase structure
- [x] Read all documentation files
- [x] Understand project scope and phases
- [x] Design new directory structure
- [x] Create `.workspace/` directory
- [x] Write `memory.md`
- [x] Write `context.md`
- [x] Write `todo.md`

---

## 🎯 Next Sprint: Dataset Verification & User Questions

### Pending
- [ ] Ask user 6 strategic questions
  1. [ ] What is the "new technique" to beat SOTA?
  2. [ ] Final submission deadline?
  3. [ ] Dataset already at `data/RCAEval/`?
  4. [ ] Strategy: Incremental or full pipeline?
  5. [ ] Baselines: Use paper results or re-implement?
  6. [ ] Can run multi-hour ablations?

- [ ] Verify dataset locally
  - [ ] Create parallel verification script
  - [ ] Check 270 cases discovered
  - [ ] Verify all modalities present
  - [ ] Check ground truth annotations
  - [ ] Generate summary statistics

- [ ] Run initial EDA
  - [ ] Fault type distribution
  - [ ] System distribution
  - [ ] Modality completeness check
  - [ ] Data quality analysis

---

## 🚀 Future Sprints (Phases 3-8)

### Phase 3-4: Metrics Encoder (Week 1)
- [ ] Implement Chronos-Bolt-Tiny encoder
- [ ] Implement TCN encoder (alternative)
- [ ] Create metrics preprocessing pipeline
- [ ] Run metrics-only RCA baseline
- [ ] First ablation: Chronos vs TCN
- [ ] Compare against Phase 2 statistical baselines

### Phase 5: Logs Encoder (Week 2)
- [ ] Integrate Drain3 log parser
- [ ] Create log embedding pipeline
- [ ] Temporal alignment with metrics
- [ ] Run logs-only RCA
- [ ] Ablation: Metrics vs Logs

### Phase 6: Traces Encoder (Week 2)
- [ ] Parse trace data to service graphs
- [ ] Implement 2-layer GCN
- [ ] Extract node/edge features
- [ ] Run traces-only RCA
- [ ] Ablation: M vs L vs T

### Phase 7: Causal Discovery (Week 3)
- [ ] Integrate tigramite PCMCI
- [ ] Tune hyperparameters (tau_max, alpha)
- [ ] Generate causal graphs per case
- [ ] Integrate with RCA ranking
- [ ] Ablation: With/without causal

### Phase 8: Multimodal Fusion (Week 3)
- [ ] Implement cross-modal attention
- [ ] Create fusion architecture
- [ ] End-to-end RCA pipeline
- [ ] Ablation: Early vs Late vs Intermediate fusion

### Phase 9: Comprehensive Ablations (Week 4) ⭐ CRITICAL
- [ ] Modality ablations (7 configs)
- [ ] Encoder ablations (3-4 configs)
- [ ] Causal ablations (4 configs)
- [ ] Fusion ablations (3 configs)
- [ ] Statistical robustness (5 seeds each)
- [ ] Create ablation master script
- [ ] Generate all results tables

### Phase 10: Baseline Comparisons (Week 4)
- [ ] Compare vs 5+ baselines
- [ ] Statistical significance testing
- [ ] Effect size calculations
- [ ] Create comparison tables
- [ ] Generate performance charts

### Phase 11: Visualization (Week 5)
- [ ] Service dependency graphs
- [ ] Attention heatmaps
- [ ] Performance bar charts
- [ ] Ablation contribution plots
- [ ] Confusion matrices
- [ ] Causal graph visualizations
- [ ] Timeline plots

### Phase 12: Report Writing (Week 6)
- [ ] Write methodology chapter
- [ ] Write results chapter
- [ ] Write discussion chapter
- [ ] Create architecture diagrams
- [ ] Professional LaTeX formatting
- [ ] Proofread and polish

---

## 📝 Notes

### Priority System
- 🔥 **CRITICAL**: Blocks other work, must complete ASAP
- ⭐ **HIGH**: Important for A+ grade
- 🔹 **MEDIUM**: Necessary but not urgent
- ⚪ **LOW**: Nice to have

### Completion Tracking
- Total tasks: ~80 across all phases
- Completed: ~10 (12%)
- Current sprint: Reorganization (20% done)
- On track for 4-6 week timeline

---

**Update this file after completing each task!**
