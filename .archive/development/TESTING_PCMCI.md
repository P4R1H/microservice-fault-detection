# Testing PCMCI Causal Discovery

This guide explains how to test PCMCI causal discovery with your local RCAEval dataset.

## What is PCMCI?

PCMCI (Peter-Clark Momentary Conditional Independence) is a causal discovery algorithm for time series data:

- **Two-stage procedure**: PC1 (parent discovery) + MCI (momentary conditional independence)
- **Handles autocorrelation**: Explicitly models temporal dependencies
- **High detection power**: >80% accuracy in high-dimensional cases
- **Fast**: Minutes for 10-50K datapoints

**Use Cases**:
- Discover causal relationships between service metrics
- Identify fault propagation paths
- Enhance RCA by weighting services based on causal influence

## Prerequisites

### 1. Install Dependencies

```bash
cd project

# Core dependencies (if not already installed)
pip install torch numpy pandas scikit-learn networkx matplotlib

# PCMCI causal discovery
pip install tigramite

# All other requirements
pip install -r requirements.txt
```

### 2. Verify Dataset Exists

```bash
ls -la data/RCAEval/

# Should show:
# - TrainTicket/
# - SockShop/
# - OnlineBoutique/
```

## Running the Test Suite

### Quick Test (3 cases)

```bash
python scripts/test_pcmci.py --n_cases 3
```

### Custom Dataset Path

```bash
python scripts/test_pcmci.py --data_path /path/to/RCAEval --n_cases 5
```

### With Visualization

```bash
python scripts/test_pcmci.py --n_cases 3 --output_dir outputs/causal_graphs
```

This will save causal graph visualizations to `outputs/causal_graphs/`.

## What Gets Tested

### ✅ Step 1: Data Loading
- Load failure cases from RCAEval
- Extract metrics time series
- Preprocess with z-score normalization

### ✅ Step 2: PCMCI Causal Discovery
- Initialize PCMCI with tigramite
- Run two-stage causal discovery:
  - **PC1**: Parent discovery with conditional independence
  - **MCI**: Momentary conditional independence testing
- Build NetworkX DiGraph from results
- Generate human-readable summary

### ✅ Step 3: Service-Level Integration
- Aggregate metric-level causality to services
- Compute causal scores: `out_degree - 0.5 * in_degree`
- Rank services by root cause likelihood
- Validate against ground truth

### ✅ Step 4: Granger-Lasso Baseline
- Run Granger-Lasso for comparison
- Faster but less powerful than PCMCI
- Good for ablation studies

### ✅ Step 5: Visualization
- Generate causal graph plots
- Show edge labels (lag, p-value)
- Save to PNG files

## Expected Output

```
======================================================================
PCMCI CAUSAL DISCOVERY TEST SUITE
======================================================================
Dataset: data/RCAEval
Test cases: 3

======================================================================
STEP 1: Loading Data for PCMCI
======================================================================
✓ Found 731 total failure cases
✓ Testing with 3 cases

======================================================================
STEP 2: Testing PCMCI Causal Discovery
======================================================================
  Case ID: TrainTicket_RE2_CPU_001
  System: TrainTicket
  Root cause: ts-order-service
  Metrics shape: (72, 7)
✓ Preprocessed metrics: (72, 7)
  Data shape: (72, 7)
  Variable names: ['cpu_usage', 'memory_usage', 'response_time', ...]

  Running PCMCI algorithm...

✓ PCMCI completed successfully

PCMCI Causal Discovery Summary
==================================================
Variables: 7
Max lag: 5
PC alpha: 0.15
Alpha level: 0.05
Total edges: 12

Discovered Causal Edges:
--------------------------------------------------
cpu_usage(t-1) --> memory_usage(t) [p=0.0023]
cpu_usage(t-2) --> response_time(t) [p=0.0145]
memory_usage(t) --> response_time(t) [p=0.0008]
response_time(t-1) --> error_rate(t) [p=0.0356]
...

==================================================

  Testing service-level integration...
✓ Service-level scores computed:
    ts-order-service: 0.8542
    other-service: 0.3127
✓ Correctly identified root cause: ts-order-service

======================================================================
STEP 3: Testing Granger-Lasso Baseline
======================================================================
  Running Granger-Lasso...
✓ Granger-Lasso completed
  Discovered 18 causal edges
  Nodes: 7

======================================================================
SUMMARY
======================================================================
✓ Data loading: PASSED
✓ PCMCI discovery: PASSED
✓ Granger-Lasso baseline: PASSED

🎉 PCMCI causal discovery tested successfully!

Next steps:
  1. Implement multimodal fusion (Phase 8)
  2. Integrate causal graph with trace encoder
  3. Build end-to-end RCA model
```

## API Usage

### Basic PCMCI Discovery

```python
from src.causal import PCMCIDiscovery
import numpy as np

# Prepare time series data
data = np.random.randn(100, 5)  # (timesteps, variables)
var_names = ['cpu', 'memory', 'latency', 'throughput', 'errors']

# Initialize PCMCI
pcmci = PCMCIDiscovery(
    tau_max=5,           # Max time lag
    pc_alpha=0.15,       # Parent discovery threshold
    alpha_level=0.05,    # Final edge threshold
    independence_test='parcorr'  # Partial correlation
)

# Discover causal graph
results = pcmci.discover_graph(data, var_names)

# Access results
causal_graph = results['causal_graph']  # NetworkX DiGraph
print(results['summary'])
```

### Service-Level Integration

```python
# Map metrics to services
service_mapping = {
    'service-a': ['cpu', 'memory'],
    'service-b': ['latency', 'throughput'],
    'service-c': ['errors']
}

# Compute service-level causal scores
service_scores = pcmci.integrate_with_services(
    causal_graph,
    service_mapping
)

# Rank services by root cause likelihood
for service, score in sorted(service_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{service}: {score:.4f}")
```

### Visualization

```python
from src.causal import visualize_causal_graph

visualize_causal_graph(
    causal_graph,
    output_path='causal_graph.png',
    figsize=(14, 10),
    show_edge_labels=True
)
```

### Convenience Function

```python
from src.causal import discover_causal_relations

# One-line causal discovery
results = discover_causal_relations(
    data,
    var_names=var_names,
    method='pcmci',  # or 'granger'
    tau_max=5,
    pc_alpha=0.15
)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'tigramite'"

```bash
pip install tigramite
```

This will install the tigramite library for PCMCI causal discovery.

### "PCMCI is taking too long"

Reduce `tau_max` or the number of variables:

```python
pcmci = PCMCIDiscovery(
    tau_max=3,  # Reduce from 5 to 3
    pc_alpha=0.2  # Increase for faster convergence
)
```

### "Too many causal edges detected"

Make the thresholds more conservative:

```python
pcmci = PCMCIDiscovery(
    pc_alpha=0.1,      # More conservative parent discovery
    alpha_level=0.01   # More conservative final edges
)
```

### "Not enough causal edges detected"

Make the thresholds more liberal:

```python
pcmci = PCMCIDiscovery(
    pc_alpha=0.2,      # More liberal parent discovery
    alpha_level=0.1    # More liberal final edges
)
```

### "ValueError: Data contains NaN values"

Ensure proper preprocessing:

```python
from src.data.preprocessing import MetricsPreprocessor

preprocessor = MetricsPreprocessor(
    fill_method='forward',  # Forward fill missing values
    clip_outliers=True
)
processed = preprocessor.fit_transform(data)
```

## Performance Benchmarks

On RTX 4070 Mobile (CPU-based):

| Component | Variables | Timesteps | Time | Memory |
|-----------|-----------|-----------|------|--------|
| PCMCI (ParCorr) | 7 | 72 | ~5s | ~50MB |
| PCMCI (ParCorr) | 20 | 500 | ~45s | ~200MB |
| PCMCI (GPDC) | 7 | 72 | ~30s | ~100MB |
| Granger-Lasso | 7 | 72 | ~1s | ~20MB |
| Granger-Lasso | 20 | 500 | ~8s | ~50MB |

**Notes**:
- PCMCI runs on CPU (tigramite doesn't use GPU)
- ParCorr (partial correlation) is faster than GPDC (Gaussian processes)
- Granger-Lasso is 5-10x faster but less accurate

## Hyperparameter Tuning

### tau_max (Max Time Lag)

- **Default**: 5 (for 5-min interval data)
- **Range**: 3-10
- **Impact**: Higher = more lagged relationships detected, but slower
- **Recommendation**:
  - 3-5 for microservices (fast propagation)
  - 5-10 for distributed systems (slower propagation)

### pc_alpha (Parent Discovery Threshold)

- **Default**: 0.15
- **Range**: 0.05-0.3
- **Impact**: Higher = more liberal parent discovery, more edges in PC1 stage
- **Recommendation**:
  - 0.1-0.15 for high-confidence edges only
  - 0.15-0.2 for more exploratory discovery

### alpha_level (Final Edge Threshold)

- **Default**: 0.05
- **Range**: 0.01-0.1
- **Impact**: Higher = more edges in final graph
- **Recommendation**:
  - 0.01-0.05 for conservative causal claims
  - 0.05-0.1 for exploratory analysis

### independence_test

- **Options**: 'parcorr', 'gpdc'
- **ParCorr**: Linear relationships, fast
- **GPDC**: Non-linear relationships, slow
- **Recommendation**: Start with 'parcorr', use 'gpdc' if needed

## Next Steps After Testing

Once all tests pass:

1. ✅ **PCMCI validated** - Ready for integration
2. 🔄 **Implement Multimodal Fusion** - Cross-attention between metrics/logs/traces
3. 🔄 **Integrate with Trace Encoder** - Use causal graph to weight service embeddings
4. 🔄 **End-to-end RCA** - Service ranking model with causal features
5. 🔄 **Evaluation** - Ablation studies (with/without PCMCI)
6. 🔄 **Experiments** - Compare PCMCI vs Granger-Lasso vs no causality

---

**Questions?** Check `.workspace/memory.md` for full project context or open an issue.
