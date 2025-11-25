# Codebase Reorganization Guide

**Date**: 2025-11-14
**Phase**: 2 → 3 Transition

This document explains the major codebase reorganization completed to support Phases 3-8 implementation.

---

## 🎯 Why Reorganize?

### Previous Issues
1. **Two context folders**: Root `/context` + `project/context/` caused confusion
2. **Mixed documentation**: Working notes mixed with formal docs
3. **No workspace**: Hard to resume work across sessions
4. **Flat src structure**: Not scalable for 6 new modules (Phases 3-8)
5. **No config management**: Hyperparameters hardcoded

### New Structure Benefits
- ✅ Clean separation: Working notes vs formal docs
- ✅ Resumable: `.workspace/` tracks all context
- ✅ Scalable: Organized `src/` for all phases
- ✅ Reproducible: YAML configs for all experiments
- ✅ Professional: Publication-ready `docs/` folder

---

## 📁 Major Changes

### 1. Workspace System (NEW)

**Created**: `.workspace/` directory for all working notes

```
.workspace/
├── memory.md        # Long-term project understanding (AI memory)
├── context.md       # Current session tracking
├── todo.md          # Task management
├── decisions.md     # All architectural decisions
├── notes.md         # Scratch space
├── experiments/     # Experiment logs
└── archived/        # Old context files moved here
```

**Purpose**: Enables seamless work resumption across sessions

**What Goes Here**:
- Session notes, brainstorming, temporary ideas
- Experiment tracking and results logs
- Decision rationale and alternatives
- Task lists and progress tracking

**What Doesn't**:
- Formal documentation (goes to `docs/`)
- Code (goes to `src/`)
- Academic references (goes to `reference/`)

### 2. Reference Directory (NEW)

**Created**: `reference/` for academic materials

**Moved Files**:
- `/context/literature-review.txt` → `reference/`
- `/context/midsem-report.txt` → `reference/`
- `/context/research-results.txt` → `reference/`
- `project/ENVIRONMENT.md` → `reference/`
- `project/TESTING.md` → `reference/`

**Archived**:
- `project/context/*.md` → `.workspace/archived/`

**Purpose**: Separate permanent references from working notes

### 3. Source Code Restructuring

**Old Structure**:
```
src/
├── baselines/
└── utils/
    ├── data_loader.py
    └── visualization.py
```

**New Structure**:
```
src/
├── data/                # Data handling (MOVED)
│   ├── loader.py       # Formerly utils/data_loader.py
│   └── preprocessing.py
├── encoders/            # NEW - Phase 3-6
│   ├── metrics_encoder.py   # Chronos + TCN
│   ├── logs_encoder.py      # Drain3 + embeddings
│   └── traces_encoder.py    # GCN + GAT
├── causal/              # NEW - Phase 7
│   └── pcmci.py
├── fusion/              # NEW - Phase 8
│   └── multimodal_fusion.py
├── models/              # NEW - Phase 8
│   └── rca_model.py
├── evaluation/          # NEW - Phase 9-11
│   └── metrics.py      # AC@k, MRR, significance tests
├── baselines/           # EXISTING
└── utils/               # EXISTING
    └── visualization.py
```

**Key Changes**:
1. `data_loader.py` moved to `data/loader.py` (more logical)
2. Created 5 new modules for Phases 3-8
3. Each module has placeholder files with docstrings

**Import Changes**:
```python
# OLD
from src.utils.data_loader import RCAEvalDataLoader

# NEW
from src.data.loader import RCAEvalDataLoader
# OR
from src.data import RCAEvalDataLoader
```

### 4. Configuration System (NEW)

**Created**: `project/config/` with YAML templates

**Files**:
1. `model_config.yaml` - All model hyperparameters
   - Encoder configs (Chronos, TCN, GCN, GAT)
   - Fusion settings
   - Causal discovery params
   - Device and precision

2. `experiment_config.yaml` - Training & evaluation
   - Batch size, learning rate, epochs
   - Ablation configurations
   - Baseline comparisons
   - Logging settings

3. `data_config.yaml` - Dataset settings
   - Paths and splits
   - Preprocessing parameters
   - Data augmentation

**Benefits**:
- No more hardcoded hyperparameters
- Easy experiment reproduction
- Version control for configs
- Override mechanism for quick tests

**Usage**:
```python
import yaml

with open('config/model_config.yaml') as f:
    config = yaml.safe_load(f)

embedding_dim = config['model']['metrics_encoder']['chronos']['embedding_dim']
```

### 5. Output Directory Structure (NEW)

**Created**: `project/outputs/` (gitignored)

```
outputs/
├── models/      # Saved model checkpoints
├── results/     # CSV/JSON metrics files
├── figures/     # Publication-quality plots
└── logs/        # Training logs, tensorboard
```

**Purpose**: Organized storage for all experimental outputs

**Gitignored**: Yes, to avoid bloating repository

### 6. Package Installation (NEW)

**Created**: `project/setup.py`

**Benefits**:
- Proper Python package structure
- Editable installation: `pip install -e .`
- Automatic dependency management
- Entry points for CLI commands

**Usage**:
```bash
# Install in development mode
cd project
pip install -e .

# Now you can import from anywhere
python -c "from src.data import RCAEvalDataLoader"

# CLI commands available
fd-download --all
fd-verify
fd-eda --all
```

### 7. Documentation Policy (CLARIFIED)

**`project/docs/` = FORMAL ONLY**

**Belongs Here**:
- Architecture diagrams (polished)
- API reference (complete)
- Evaluation methodology (final)
- Publication-ready content

**Does NOT Belong**:
- Working notes (→ `.workspace/notes.md`)
- Experiment logs (→ `.workspace/experiments/`)
- Task lists (→ `.workspace/todo.md`)
- Decisions (→ `.workspace/decisions.md`)

**Current Formal Docs**:
- `MODULE_INTERFACES.md` - Complete API specification
- `PHASE2_SETUP.md` - Phase 2 deliverables

**To Be Added** (Phases 12-14):
- `ARCHITECTURE.md` - System design
- `API_REFERENCE.md` - Usage guide
- `EVALUATION.md` - Comprehensive results

---

## 🔄 Migration Checklist

If you had local changes, update them:

### Code Imports
- [ ] Update imports: `src.utils.data_loader` → `src.data.loader`
- [ ] Update tests to match new structure
- [ ] Update scripts if they import from src

### File Locations
- [ ] Academic references now in `reference/`
- [ ] Working notes now in `.workspace/`
- [ ] Formal docs only in `project/docs/`

### Configuration
- [ ] Use YAML configs instead of hardcoded values
- [ ] Check `config/*.yaml` for all settings
- [ ] Override with `config/*_override.yaml` for local changes

### Outputs
- [ ] All results go to `outputs/`
- [ ] All figures go to `outputs/figures/`
- [ ] All logs go to `outputs/logs/`

---

## 🚀 Next Steps After Reorganization

### 1. Verify Dataset
```bash
python scripts/verify_dataset.py
```

### 2. Install Package
```bash
cd project
pip install -e .
```

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Begin Phase 3
Start implementing metrics encoders:
- `src/encoders/metrics_encoder.py`
- Follow template docstrings

---

## 📝 Key Takeaways

1. **`.workspace/`** = All working memory (AI can resume here)
2. **`reference/`** = Academic materials (permanent)
3. **`project/docs/`** = Formal documentation ONLY (publication-ready)
4. **`project/config/`** = All hyperparameters (no hardcoding)
5. **`project/outputs/`** = All experimental results (gitignored)
6. **`src/`** = Domain-organized modules (scalable for Phases 3-8)

---

## ❓ Questions?

Check:
- `.workspace/memory.md` - Full project understanding
- `.workspace/decisions.md` - Why each choice was made
- `docs/MODULE_INTERFACES.md` - Complete API specification

---

**Reorganization Complete**: 2025-11-14
**Ready for**: Phase 3-8 Implementation
