# Testing Instructions

**Phase 2 Complete** - Ready for validation with extracted dataset

---

## Prerequisites

✅ **Dataset extracted** to `data/RCAEval/`
✅ **Python environment** with all dependencies from `requirements.txt`
✅ **Working directory**: `project/`

---

## Test Suite Overview

We have 3 test scripts to validate Phase 2 infrastructure:

| Test | Purpose | Runtime | Output |
|------|---------|---------|--------|
| `test_data_loading.py` | Verify dataset extraction and loader | ~5-10 sec | Console + pass/fail |
| `test_baselines.py` | Test statistical baselines on real data | ~2-5 min | Console + CSV results |
| `eda_analysis.py` | Comprehensive EDA on all modalities | ~5-10 min | Console + plots + stats |

---

## Test 1: Data Loading (REQUIRED - Run First)

### Purpose
Verify dataset extraction was successful and data loader works correctly.

### Command
```bash
cd project
python tests/test_data_loading.py
```

### Expected Output
```
================================================================================
TEST 1: Data Loader Initialization
================================================================================
✅ Data loader initialized successfully

================================================================================
TEST 2: Load All Cases
================================================================================
✅ Loaded 270 total failure cases
   Systems found: {'TrainTicket', 'SockShop', 'OnlineBoutique'}

================================================================================
TEST 3: Load TrainTicket System
================================================================================
✅ Loaded 90 TrainTicket cases

================================================================================
TEST 4: Inspect Sample Case
================================================================================

📦 Sample Case: case_001
   System: TrainTicket
   Fault Type: CPU
   Root Cause Service: ts-order-service
   Root Cause Indicator: cpu_usage_percent

📊 Data Modalities:
   ✅ Metrics: (60, 123) (timesteps × features)
      Columns (first 5): ['cpu_usage_percent', 'memory_usage_mb', ...]
   ✅ Logs: 45678 entries
      Columns: ['timestamp', 'service', 'level', 'message', ...]
   ✅ Traces: 123456 spans
      Columns: ['span_id', 'service', 'latency', 'parent_span', ...]

================================================================================
TEST 5: Load Train/Val/Test Splits
================================================================================
✅ Dataset splits created:
   Train: 162 cases (60.0%)
   Val:   54 cases (20.0%)
   Test:  54 cases (20.0%)
✅ No data leakage: splits are disjoint

================================================================================
TEST 6: Fault Type Distribution
================================================================================

🔥 Fault Types:
   CPU: 45 cases (16.7%)
   MEM: 45 cases (16.7%)
   DISK: 45 cases (16.7%)
   DELAY: 45 cases (16.7%)
   LOSS: 45 cases (16.7%)
   SOCKET: 45 cases (16.7%)

================================================================================
TEST 7: System Distribution
================================================================================

📦 Systems:
   OnlineBoutique: 90 cases (33.3%)
   SockShop: 90 cases (33.3%)
   TrainTicket: 90 cases (33.3%)

================================================================================
✅ ALL TESTS PASSED!
================================================================================

Dataset Statistics Summary:
  Total cases: 270
  Systems: 3
  Fault types: 6
  Data completeness:
    Metrics: 270/270 (100.0%)
    Logs: 270/270 (100.0%)
    Traces: 270/270 (100.0%)

🎉 Dataset is ready for experiments!
```

### What If It Fails?

**Error: `FileNotFoundError: RCAEval dataset not found`**
- **Cause**: Dataset not extracted yet
- **Fix**: Run `python scripts/download_dataset.py --all` first

**Error: `ModuleNotFoundError: No module named 'pandas'`**
- **Cause**: Dependencies not installed
- **Fix**: Run `pip install -r requirements.txt`

**Error: Cases loaded but metrics/logs/traces are `None`**
- **Cause**: Dataset directory structure mismatch
- **Fix**: Check that extracted structure matches:
  ```
  data/RCAEval/
  ├── TrainTicket/
  │   ├── RE1/
  │   │   ├── cases/
  │   │   │   ├── case_001/
  │   │   │   │   ├── metrics.csv
  │   │   │   │   ├── logs.csv
  │   │   │   │   └── traces.csv
  │   │   └── ground_truth.csv
  │   ├── RE2/
  │   └── RE3/
  ```

---

## Test 2: Statistical Baselines (Run After Test 1 Passes)

### Purpose
Test all statistical baseline methods on real RCAEval data to establish performance lower bounds.

### Command
```bash
cd project
python tests/test_baselines.py
```

### Expected Output
```
================================================================================
BASELINE TESTING ON RCAEVAL DATASET
================================================================================

📂 Loading dataset...
✅ Loaded: 162 train, 54 val, 54 test cases

🧪 Testing on 10 cases from validation set

🔧 Initializing baseline methods...

================================================================================
RUNNING BASELINES
================================================================================

📦 Case 1/10: case_042
   System: TrainTicket
   Fault: CPU
   Ground Truth: ts-order-service
   3-Sigma: AC@1=0.00, MRR=0.167
   ARIMA: AC@1=0.00, MRR=0.200
   Granger: AC@1=1.00, MRR=1.000
   Random: AC@1=0.00, MRR=0.083

[... more cases ...]

================================================================================
BASELINE PERFORMANCE SUMMARY
================================================================================

Method               AC@1     AC@3     AC@5     MRR      N
--------------------------------------------------------------------------------
3-Sigma              0.100    0.300    0.500    0.250    10
ARIMA                0.200    0.400    0.600    0.350    10
Granger-Lasso        0.300    0.500    0.700    0.450    10
Random Walk          0.050    0.150    0.250    0.100    10

================================================================================
✅ BASELINE TESTING COMPLETE!
================================================================================

💾 Results saved to: project/outputs/baseline_tests/baseline_results.csv

📊 Next Steps:
   1. Run EDA: python scripts/eda_analysis.py --all
   2. Review baseline performance
   3. Begin implementing advanced methods (Chronos, GCN, PCMCI)
```

### Expected Performance Range

Based on literature review, statistical baselines should achieve:

| Method | Expected AC@1 | Expected MRR |
|--------|--------------|--------------|
| Random Walk | 0.03-0.08 | 0.08-0.15 |
| 3-Sigma | 0.10-0.20 | 0.20-0.30 |
| ARIMA | 0.15-0.25 | 0.25-0.35 |
| Granger-Lasso | 0.20-0.30 | 0.30-0.40 |

**Note**: These are just baselines! Advanced methods (Chronos + PCMCI + GNN + Fusion) should achieve **AC@1 > 0.70**.

### Output Files
- `project/outputs/baseline_tests/baseline_results.csv` - Performance metrics

---

## Test 3: Exploratory Data Analysis (Run After Test 1 Passes)

### Purpose
Comprehensive EDA on all three modalities to understand data characteristics.

### Command
```bash
cd project

# Quick analysis (TrainTicket only, fastest)
python scripts/eda_analysis.py

# Full analysis (all systems, ~10 minutes)
python scripts/eda_analysis.py --all

# Specific system
python scripts/eda_analysis.py --systems TrainTicket SockShop
```

### Expected Output
```
================================================================================
RCAEval Dataset - Exploratory Data Analysis
================================================================================

📊 Loaded 270 failure cases

================================================================================
1. Dataset-Level Statistics
================================================================================

📦 System Distribution:
   OnlineBoutique: 90 cases (33.3%)
   SockShop: 90 cases (33.3%)
   TrainTicket: 90 cases (33.3%)

🔥 Fault Type Distribution:
   CPU: 45 cases (16.7%)
   MEM: 45 cases (16.7%)
   DISK: 45 cases (16.7%)
   DELAY: 45 cases (16.7%)
   LOSS: 45 cases (16.7%)
   SOCKET: 45 cases (16.7%)

🎯 Root Cause Service Distribution:
   Unique services: 47
   Top 10 services:
      ts-order-service: 12 cases
      frontend: 11 cases
      ...

📊 Data Modality Availability:
   Metrics: 270/270 cases (100.0%)
   Logs: 270/270 cases (100.0%)
   Traces: 270/270 cases (100.0%)

💾 Saved distribution plots to: project/outputs/eda/dataset_distributions.png

================================================================================
2. Metrics Modality Analysis
================================================================================

📈 Analyzing 270 cases with metrics

🔢 Dimensionality:
   Features per case:
      Min: 77, Max: 376, Median: 189
   Timesteps per case:
      Min: 60, Max: 120, Median: 60

📊 Sample Case (case_001):
   Shape: (60, 123)
   Columns: ['cpu_usage_percent', 'memory_usage_mb', ...]

📉 Statistical Properties:
   Missing values: 342 (0.46%)
   Mean: 0.4721
   Std: 0.3214

💾 Saved metrics statistics to: project/outputs/eda/metrics_statistics.txt

================================================================================
3. Logs Modality Analysis
================================================================================

📝 Analyzing 270 cases with logs

📊 Log Volume:
   Total logs: 4,567,890
   Logs per case:
      Min: 8,654
      Max: 26,987
      Median: 15,432
      Mean: 16,918

📄 Sample Case (case_001):
   Log entries: 15,432
   Columns: ['timestamp', 'service', 'level', 'message']

🔍 Log Level Distribution:
      INFO: 12,345 (80.0%)
      ERROR: 2,456 (15.9%)
      WARNING: 523 (3.4%)
      DEBUG: 108 (0.7%)

================================================================================
4. Traces Modality Analysis
================================================================================

🔗 Analyzing 270 cases with traces

📊 Trace Volume:
   Total spans: 15,678,234
   Spans per case:
      Min: 39,654
      Max: 76,789
      Median: 56,432
      Mean: 58,068

🔍 Sample Case (case_001):
   Trace spans: 56,432
   Columns: ['span_id', 'service', 'latency', 'parent_span']

🏢 Services in trace:
      Unique services: 41
      Top services: ['frontend', 'ts-order-service', 'ts-user-service', ...]

================================================================================
5. Cross-Modality Patterns
================================================================================

📊 Modality Completeness:
   all_three: 270 cases (100.0%)
   metrics_logs: 0 cases (0.0%)
   metrics_traces: 0 cases (0.0%)
   logs_traces: 0 cases (0.0%)
   metrics_only: 0 cases (0.0%)
   logs_only: 0 cases (0.0%)
   traces_only: 0 cases (0.0%)
   none: 0 cases (0.0%)

================================================================================
6. Root Cause Patterns
================================================================================

🔥 Fault Types by System:

   TrainTicket:
      CPU: 15 cases
      MEM: 15 cases
      DISK: 15 cases
      DELAY: 15 cases
      LOSS: 15 cases
      SOCKET: 15 cases

   [... similar for other systems ...]

🎯 Root Cause Indicators:
   Unique indicators: 23
   Top 10 indicators:
      cpu_usage_percent: 45 cases
      memory_usage_mb: 45 cases
      ...

================================================================================
✅ EDA complete! Results saved to: project/outputs/eda
================================================================================
```

### Output Files
- `project/outputs/eda/dataset_distributions.png` - System/fault/service distributions
- `project/outputs/eda/metrics_statistics.txt` - Detailed metrics statistics

---

## Validation Checklist

After running all tests, verify:

- [ ] `test_data_loading.py` passes with 270 cases loaded
- [ ] All cases have metrics, logs, and traces (100% completeness)
- [ ] Train/val/test splits are 162/54/54 with no overlap
- [ ] Statistical baselines produce AC@1 scores in expected ranges
- [ ] EDA analysis completes and generates output files
- [ ] Baseline results saved to `outputs/baseline_tests/`
- [ ] EDA plots saved to `outputs/eda/`

---

## What's Next?

Once all tests pass:

### Immediate Next Steps (Phase 3)
1. ✅ Review baseline performance (should be low - that's good!)
2. ✅ Analyze EDA outputs to understand data characteristics
3. ✅ Begin implementing Chronos-Bolt-Tiny metrics encoder
4. ✅ Implement preprocessing modules

### Phase 3 Goals
- Chronos-Bolt-Tiny zero-shot inference
- Preprocessing pipeline (metrics, logs, traces)
- GCN encoder for service graphs
- Drain3 log parsing

---

## Troubleshooting

### Test runs but gets low performance

**This is EXPECTED!** Statistical baselines are intentionally simple and should have low performance (AC@1 ~ 0.10-0.30). This establishes the lower bound that advanced methods should beat.

### Test crashes with memory error

- Reduce test set size in `test_baselines.py`: Change `test_cases = val[:10]` to `test_cases = val[:5]`
- For EDA, use `--systems TrainTicket` to analyze one system at a time

### Tests are slow

- **Data loading**: Should be <10 seconds
- **Baselines**: 2-5 minutes is normal (Granger-Lasso is slow)
- **EDA**: 5-10 minutes for full dataset is normal

Speed up by:
- Using smaller test set
- Analyzing one system at a time
- Reducing `max_vars` in Granger-Lasso

---

## Support

If tests fail unexpectedly:
1. Check dataset extraction: `ls -la data/RCAEval/`
2. Check Python environment: `python --version` (should be 3.8+)
3. Check dependencies: `pip list | grep pandas`
4. Review error traceback for specific issues

---

**Ready to test?** Start with Test 1 (data loading) and proceed sequentially! 🚀
