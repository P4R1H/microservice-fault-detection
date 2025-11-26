# 🚀 Multimodal RCA System: Improvement Proposal & Implementation Plan

**Version:** 2.0
**Date:** November 2025
**Authors:** Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan
**Project:** Multimodal Root Cause Analysis for Microservice Systems

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Analysis](#2-current-system-analysis)
3. [Proposed Architecture](#3-proposed-architecture)
4. [LLM Integration Strategy](#4-llm-integration-strategy)
5. [Encoder Improvements](#5-encoder-improvements)
6. [Dataset Expansion Plan](#6-dataset-expansion-plan)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Expected Results](#8-expected-results)
9. [Technical Specifications](#9-technical-specifications)
10. [Risk Assessment](#10-risk-assessment)

---

## 1. Executive Summary

### Current Performance

| Metric         | Current Value | SOTA (RUN) |
| -------------- | ------------- | ---------- |
| AC@1 (mean)    | 66.7%         | 63.1%      |
| AC@1 (best)    | 81.5%         | 63.1%      |
| Inference Time | 3.3ms         | 892ms      |
| Parameters     | 324K-722K     | ~10M       |

### Improvement Goals

| Metric           | Current     | Target           | Improvement |
| ---------------- | ----------- | ---------------- | ----------- |
| AC@1 (mean)      | 66.7%       | **78-82%** | +11-15%     |
| AC@1 (best)      | 81.5%       | **88-92%** | +7-10%      |
| Explainability   | None        | Full NL          | ✓          |
| Dataset Coverage | 1 benchmark | 3+ benchmarks    | 3×         |

### Key Improvements

1. **Architecture Upgrade**: TCN → Specialized encoders (Drain3+LLM for logs, GCN for traces)
2. **LLM Integration**: Semantic understanding for logs + causal reasoning + explainability
3. **Optional Chronos**: Foundation model for metrics (configurable)
4. **Dataset Expansion**: IBM Cloud / GAIA benchmark validation

---

## 2. Current System Analysis

### 2.1 Architecture Overview

```
Current System (V4):
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Failure Case                               │
│     (60 timesteps × 64 metrics + 32 log templates + 32 trace features)  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────┼────────────────────────────────┐
    ▼                        ▼                                ▼
┌──────────────┐      ┌──────────────┐                ┌──────────────┐
│ TCN Encoder  │      │ TCN Encoder  │                │ TCN Encoder  │
│ (Metrics)    │      │ (Logs)       │                │ (Traces)     │
│ DepthSep     │      │ DepthSep     │                │ DepthSep     │
└──────┬───────┘      └──────┬───────┘                └──────┬───────┘
       │  64d                │  64d                          │  64d
       └─────────────────────┼────────────────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Gated Fusion   │
                    └────────┬────────┘
                             │ 128d
                             ▼
              ┌──────────────────────────────┐
              │  Cross-Service Attention     │
              │  + PCMCI Causal Bias (λ=0.3) │
              └──────────────┬───────────────┘
                             ▼
                    ┌────────────────┐
                    │ Service Ranking │
                    └────────────────┘
```

### 2.2 Current Limitations

| Component                  | Current Approach       | Limitation                               |
| -------------------------- | ---------------------- | ---------------------------------------- |
| **Logs Encoder**     | TCN on template counts | Loses semantic meaning of error messages |
| **Traces Encoder**   | TCN (time-series view) | Ignores graph topology of service calls  |
| **Causal Discovery** | PCMCI only             | Statistical-only, no domain knowledge    |
| **Explainability**   | None                   | Black-box predictions                    |
| **Metrics Encoder**  | TCN                    | Good, but foundation models could help   |

### 2.3 Ablation Study Insights

From our ablation results:

- **Logs Only**: 45.6% AC@1 → Underperforming, semantic info lost
- **Traces Only**: 52.3% AC@1 → Graph structure not exploited
- **GAT vs GCN**: GAT showed +0.7% improvement → Graph methods work better

---

## 3. Proposed Architecture

### 3.1 New Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT: Failure Case                               │
│     (60 timesteps × 64 metrics + raw logs + trace graph)                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────┼────────────────────────────────┐
    ▼                        ▼                                ▼
┌──────────────────┐  ┌─────────────────────┐         ┌──────────────────┐
│  METRICS ENCODER │  │    LOGS ENCODER     │         │  TRACES ENCODER  │
│                  │  │                     │         │                  │
│ Option A: TCN    │  │ Drain3 Parser       │         │ 2-layer GCN      │
│ (80K params)     │  │      ↓              │         │ on service graph │
│                  │  │ Template Extraction │         │                  │
│ Option B: Chronos│  │ (1,247 templates)   │         │ Node features:   │
│ (8M params,      │  │      ↓              │         │ - latency stats  │
│  frozen)         │  │ LLM Embeddings      │         │ - error rates    │
│                  │  │ (OpenAI/Local)      │         │ - request counts │
└────────┬─────────┘  └──────────┬──────────┘         └────────┬─────────┘
         │  64d                  │  64d                        │  64d
         └───────────────────────┼─────────────────────────────┘
                                 ▼
                    ┌───────────────────────┐
                    │    Gated Fusion       │ ← Learns modality weights
                    │ g_m·M + g_l·L + g_t·T │    per sample
                    └───────────┬───────────┘
                                │ 128d per service
                                ▼
              ┌─────────────────────────────────────┐
              │   Cross-Service Attention           │
              │   (2 layers, 4 heads)               │
              │   + PCMCI Statistical Bias (λ=0.3)  │ ← Data-driven
              │   + LLM Causal Prior (λ=0.2)        │ ← Knowledge-driven
              └─────────────────┬───────────────────┘
                                ▼
                    ┌───────────────────┐
                    │  Service Ranking  │ → [s1, s2, ..., sN]
                    │      Head         │
                    └─────────┬─────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │     LLM EXPLAINER (NEW)           │
              │                                   │
              │  "Service X is the root cause    │
              │   because: [reasoning based on   │
              │   metrics anomalies, log errors, │
              │   and trace patterns]"           │
              └───────────────────────────────────┘
```

### 3.2 Component Comparison

| Component | Current V4     | Proposed V5                   | Benefit                        |
| --------- | -------------- | ----------------------------- | ------------------------------ |
| Metrics   | TCN (DepthSep) | TCN or Chronos (configurable) | Foundation model option        |
| Logs      | TCN on counts  | Drain3 + LLM Embeddings       | Semantic understanding         |
| Traces    | TCN            | 2-layer GCN                   | Graph topology awareness       |
| Causal    | PCMCI only     | PCMCI + LLM Prior             | Statistical + Domain knowledge |
| Output    | Rankings only  | Rankings + NL Explanation     | Actionable insights            |

---

## 4. LLM Integration Strategy

### 4.1 LLM Use Cases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM INTEGRATION POINTS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. LOG SEMANTIC EMBEDDINGS                                              │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ Input:  "ERROR: Connection timeout to database-svc"  │            │
│     │ Output: Dense 1536d embedding capturing semantics    │            │
│     │ Model:  text-embedding-3-small (OpenAI) or          │            │
│     │         all-MiniLM-L6-v2 (local)                    │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
│  2. CAUSAL PRIOR INJECTION                                              │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ Prompt: "Given microservice system with services:    │            │
│     │         [frontend, cart, checkout, payment, db]      │            │
│     │         Rate causal influence (0-1) between pairs"   │            │
│     │ Output: 5×5 causal prior matrix                     │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
│  3. ROOT CAUSE EXPLANATION                                              │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ Input:  Model prediction + metrics/logs/traces      │            │
│     │ Output: "payment-service is root cause because:      │            │
│     │         - CPU spike at 14:23 (98% utilization)      │            │
│     │         - ERROR logs show 'transaction timeout'     │            │
│     │         - Downstream services show increased latency│            │
│     │         Recommended action: Scale payment pods"     │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
│  4. SYNTHETIC DATA AUGMENTATION                                         │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ Task: Generate realistic failure scenarios          │            │
│     │ Output: New training cases with diverse patterns    │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 LLM Configuration Options

```yaml
# config/llm_config.yaml
llm:
  # Embedding model for logs
  embeddings:
    provider: "openai"  # Options: openai, local, azure
    model: "text-embedding-3-small"  # 1536d, $0.02/1M tokens
    fallback: "all-MiniLM-L6-v2"  # Local fallback
    cache_enabled: true
    cache_path: "outputs/embedding_cache.pkl"
  
  # Reasoning model for explanations
  reasoning:
    provider: "openai"
    model: "gpt-4o-mini"  # Cost-effective for explanations
    temperature: 0.3
    max_tokens: 500
  
  # Causal prior generation
  causal_prior:
    enabled: true
    model: "gpt-4o"  # Better reasoning for causal relationships
    cache_enabled: true  # Cache per system topology
```

### 4.3 LLM Logs Encoder Implementation

```python
# src/encoders/llm_logs_encoder.py

class LLMLogsEncoder(nn.Module):
    """
    Enhanced logs encoder using LLM embeddings.
  
    Pipeline:
    1. Drain3: Raw logs → Templates (1,247 patterns)
    2. LLM: Templates → Semantic embeddings (1536d)
    3. Projection: 1536d → 64d for fusion
    """
  
    def __init__(self, 
                 embedding_dim: int = 64,
                 llm_provider: str = "openai",
                 cache_enabled: bool = True):
        super().__init__()
      
        # Drain3 parser
        self.drain_parser = Drain3Parser(
            similarity_threshold=0.5,
            depth=4,
            max_children=100
        )
      
        # LLM embedder
        if llm_provider == "openai":
            self.embedder = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
        else:
            self.embedder = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )
      
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(1536, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
      
        # Cache
        self.cache_enabled = cache_enabled
        self.embedding_cache = {}
  
    def forward(self, raw_logs: List[str]) -> torch.Tensor:
        """
        Encode raw logs to dense embeddings.
      
        Args:
            raw_logs: List of raw log strings
          
        Returns:
            (batch, seq_len, embedding_dim) tensor
        """
        # Step 1: Parse to templates
        templates = []
        for log in raw_logs:
            template = self.drain_parser.parse(log)
            templates.append(template)
      
        # Step 2: Get LLM embeddings (with caching)
        embeddings = []
        for template in templates:
            if self.cache_enabled and template in self.embedding_cache:
                emb = self.embedding_cache[template]
            else:
                emb = self.embedder.embed(template)
                if self.cache_enabled:
                    self.embedding_cache[template] = emb
            embeddings.append(emb)
      
        embeddings = torch.stack(embeddings)  # (batch, 1536)
      
        # Step 3: Project to target dimension
        output = self.projection(embeddings)  # (batch, 64)
      
        return output
```

### 4.4 LLM Explainer Implementation

```python
# src/explainability/llm_explainer.py

class LLMExplainer:
    """
    Generate natural language explanations for RCA predictions.
    """
  
    EXPLANATION_PROMPT = """
You are an expert Site Reliability Engineer analyzing a microservice failure.

## Failure Context
- **Predicted Root Cause**: {predicted_service}
- **Confidence**: {confidence:.1%}
- **System**: {system_name}

## Evidence

### Metrics Anomalies (Service: {predicted_service})
{metrics_summary}

### Log Patterns
{log_patterns}

### Trace Analysis
{trace_summary}

## Task
Provide a concise root cause analysis:
1. Why is {predicted_service} likely the root cause?
2. What is the failure propagation path?
3. Recommended remediation actions?

Keep response under 200 words, be specific and actionable.
"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
  
    def explain(self,
                prediction: Dict,
                metrics: np.ndarray,
                logs: List[str],
                traces: Dict) -> str:
        """
        Generate explanation for RCA prediction.
        """
        # Summarize evidence
        metrics_summary = self._summarize_metrics(metrics)
        log_patterns = self._extract_error_patterns(logs)
        trace_summary = self._summarize_traces(traces)
      
        prompt = self.EXPLANATION_PROMPT.format(
            predicted_service=prediction['service'],
            confidence=prediction['probability'],
            system_name=prediction.get('system', 'Unknown'),
            metrics_summary=metrics_summary,
            log_patterns=log_patterns,
            trace_summary=trace_summary
        )
      
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
      
        return response.choices[0].message.content
  
    def _summarize_metrics(self, metrics: np.ndarray) -> str:
        """Extract key anomalies from metrics."""
        anomalies = []
        metric_names = ['CPU', 'Memory', 'Latency', 'Error Rate', 
                        'Request Count', 'Network I/O', 'Disk I/O']
      
        for i, name in enumerate(metric_names):
            if i < metrics.shape[-1]:
                values = metrics[..., i]
                if values.max() > values.mean() + 2 * values.std():
                    anomalies.append(f"- {name}: Spike detected (max: {values.max():.2f})")
      
        return "\n".join(anomalies) if anomalies else "- No significant anomalies"
  
    def _extract_error_patterns(self, logs: List[str]) -> str:
        """Extract ERROR/WARN patterns from logs."""
        errors = [l for l in logs if 'ERROR' in l or 'WARN' in l][:5]
        return "\n".join(f"- {e[:100]}" for e in errors) or "- No error logs"
  
    def _summarize_traces(self, traces: Dict) -> str:
        """Summarize trace latencies."""
        if not traces:
            return "- No trace data available"
      
        summaries = []
        for svc, data in list(traces.items())[:3]:
            avg_latency = np.mean(data.get('latency', [0]))
            error_rate = np.mean(data.get('errors', [0]))
            summaries.append(f"- {svc}: {avg_latency:.0f}ms avg latency, {error_rate:.1%} errors")
      
        return "\n".join(summaries)
```

---

## 5. Encoder Improvements

### 5.1 Chronos Integration (Optional)

Chronos is Amazon's foundation model for time-series forecasting. We can optionally use it for metrics encoding.

```python
# src/encoders/chronos_encoder.py

class ChronosEncoder(nn.Module):
    """
    Chronos-based metrics encoder using Amazon's foundation model.
  
    Pros:
    - Pre-trained on 27B time-series observations
    - Strong zero-shot performance
    - Captures complex temporal patterns
  
    Cons:
    - Large model size (8M-710M params)
    - Slower inference (~10ms vs 1ms for TCN)
    - Requires transformers library
  
    Recommendation: Use for offline analysis, TCN for real-time.
    """
  
    def __init__(self,
                 model_name: str = "amazon/chronos-bolt-tiny",  # 8M params
                 embedding_dim: int = 64,
                 freeze_backbone: bool = True,
                 context_length: int = 512):
        super().__init__()
      
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
      
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
      
        # Chronos outputs 256d, project to target
        self.projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embedding_dim)
        )
      
        self.context_length = context_length
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_metrics)
        Returns:
            (batch, embedding_dim)
        """
        batch_size, seq_len, n_metrics = x.shape
      
        # Process each metric channel
        embeddings = []
        for i in range(n_metrics):
            metric_series = x[:, :, i]  # (batch, seq_len)
          
            # Tokenize and encode
            with torch.no_grad():
                hidden = self.model.encoder(
                    metric_series.unsqueeze(-1)
                ).last_hidden_state
          
            # Pool over sequence
            pooled = hidden.mean(dim=1)  # (batch, 256)
            embeddings.append(pooled)
      
        # Aggregate across metrics
        combined = torch.stack(embeddings, dim=1).mean(dim=1)
      
        # Project
        output = self.projection(combined)
      
        return output
```

### 5.2 GCN Traces Encoder (Already Implemented)

The GCN encoder already exists in `src/encoders/traces_encoder.py`. We need to wire it into the main model.

```python
# Modification to src/models/rca_v5_multimodal.py

class MultimodalRCAModelV5(nn.Module):
    def __init__(self, ...):
        # ... other encoders ...
      
        # Use GCN instead of TCN for traces
        self.traces_encoder = GCNEncoder(
            in_channels=8,  # Node features
            hidden_channels=64,
            embedding_dim=64,
            num_layers=2,
            dropout=0.3,
            pooling='mean'
        )
```

### 5.3 Encoder Configuration Matrix

| Encoder                     | Model           | Params    | Latency | Use Case               |
| --------------------------- | --------------- | --------- | ------- | ---------------------- |
| **Metrics (Default)** | TCN DepthSep    | 80K       | 1.2ms   | Real-time              |
| **Metrics (Alt)**     | Chronos-Tiny    | 8M        | 10ms    | Offline/High-accuracy  |
| **Logs (New)**        | Drain3 + LLM    | ~1K + API | 5-50ms  | Semantic understanding |
| **Logs (Fallback)**   | Drain3 + TF-IDF | ~10K      | 2ms     | No API access          |
| **Traces (New)**      | 2-layer GCN     | 50K       | 1.5ms   | Graph topology         |

---

## 6. Dataset Expansion Plan

### 6.1 Current Dataset: RCAEval

| Property   | Value                                 |
| ---------- | ------------------------------------- |
| Systems    | OnlineBoutique, SockShop, TrainTicket |
| Cases      | 181 multimodal failure cases          |
| Modalities | Metrics + Logs + Traces               |
| Labels     | Single root cause per case            |

### 6.2 Target Datasets

#### 6.2.1 GAIA Dataset (CloudWise)

**Source**: https://github.com/CloudWise-OpenSource/GAIA-DataSet

| Property       | Value                            |
| -------------- | -------------------------------- |
| System         | MicroSS (QR Code login scenario) |
| Duration       | 2 weeks continuous data          |
| Metrics        | 6,500+ metrics                   |
| Logs           | 7,000,000+ log entries           |
| Traces         | Full OpenTracing data            |
| Anomaly Labels | ✓ Injection records provided    |

**Data Format**:

```
MicroSS/
├── metric/           # CSV: timestamp, value
├── trace/            # CSV: trace_id, span_id, parent_id, latency, status
├── business/         # Logs: datetime, service, message
└── run/              # Anomaly injection records
```

**Integration Steps**:

1. Download from GitHub releases (V1.10)
2. Parse metrics CSVs → align timestamps
3. Parse traces → build service dependency graph
4. Parse logs → Drain3 template extraction
5. Load anomaly injection records as labels
6. Create train/val/test splits

#### 6.2.2 IBM Cloud Pak / Alternative Datasets

Since direct IBM datasets are not publicly available, we'll use comparable alternatives:

**Option A: AIOps Challenge Datasets**

- Source: Various AIOps competitions
- Contains: Metrics, logs, traces from cloud systems

**Option B: Alibaba Microservice Traces**

- Source: https://github.com/alibaba/clusterdata
- Contains: Large-scale trace data

**Option C: Google Borg Traces**

- Source: https://github.com/google/cluster-data
- Contains: Cluster workload traces

### 6.3 Dataset Adapter Implementation

```python
# src/data/dataset_adapters.py

class GAIADatasetAdapter:
    """
    Adapter for GAIA dataset format.
  
    Converts GAIA data structure to our unified format.
    """
  
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.metrics_path = self.data_path / "MicroSS" / "metric"
        self.traces_path = self.data_path / "MicroSS" / "trace"
        self.logs_path = self.data_path / "MicroSS" / "business"
        self.labels_path = self.data_path / "MicroSS" / "run"
  
    def load_metrics(self, service: str, start: datetime, end: datetime) -> np.ndarray:
        """Load metrics for a service within time window."""
        metrics = []
        for csv_file in self.metrics_path.glob(f"*{service}*.csv"):
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            metrics.append(df['value'].values)
      
        return np.column_stack(metrics) if metrics else np.zeros((60, 64))
  
    def load_traces(self, start: datetime, end: datetime) -> Dict:
        """Load and parse trace data."""
        traces = {}
        for csv_file in self.traces_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
          
            # Build graph structure
            for _, row in df.iterrows():
                svc = row['service_name']
                if svc not in traces:
                    traces[svc] = {'latency': [], 'errors': [], 'calls': []}
              
                traces[svc]['latency'].append(
                    (row['end_time'] - row['start_time']).total_seconds() * 1000
                )
                traces[svc]['errors'].append(row['status_code'] != 200)
      
        return traces
  
    def load_logs(self, service: str, start: datetime, end: datetime) -> List[str]:
        """Load log entries for a service."""
        logs = []
        for log_file in self.logs_path.glob(f"*{service}*.csv"):
            df = pd.read_csv(log_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]
            logs.extend(df['message'].tolist())
      
        return logs
  
    def load_labels(self) -> List[Dict]:
        """Load anomaly injection labels."""
        labels = []
        for label_file in self.labels_path.glob("*.csv"):
            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                if 'anomaly' in row['message'].lower():
                    labels.append({
                        'timestamp': row['datetime'],
                        'service': row['service'],
                        'type': self._parse_anomaly_type(row['message'])
                    })
        return labels
  
    def _parse_anomaly_type(self, message: str) -> str:
        """Extract anomaly type from injection message."""
        if 'memory' in message.lower():
            return 'memory_leak'
        elif 'cpu' in message.lower():
            return 'cpu_exhaustion'
        elif 'network' in message.lower():
            return 'network_delay'
        return 'unknown'
  
    def create_failure_cases(self, window_minutes: int = 15) -> List[FailureCase]:
        """
        Create failure cases from GAIA data.
      
        Returns:
            List of FailureCase objects compatible with our training pipeline
        """
        labels = self.load_labels()
        cases = []
      
        for label in labels:
            start = label['timestamp'] - timedelta(minutes=window_minutes)
            end = label['timestamp'] + timedelta(minutes=5)
          
            case = FailureCase(
                case_id=f"gaia_{label['timestamp'].isoformat()}",
                root_cause=label['service'],
                fault_type=label['type'],
                system='GAIA-MicroSS',
                metrics=self.load_all_metrics(start, end),
                logs=self.load_all_logs(start, end),
                traces=self.load_traces(start, end)
            )
            cases.append(case)
      
        return cases
```

### 6.4 Cross-Dataset Evaluation Script

```python
# scripts/evaluate_cross_dataset.py

"""
Evaluate model on multiple datasets for generalization testing.

Usage:
    python scripts/evaluate_cross_dataset.py \
        --checkpoint outputs/best_model.pt \
        --datasets rcaeval gaia \
        --output results/cross_dataset_eval.json
"""

import argparse
from src.data import RCAEvalDataset, GAIADatasetAdapter
from src.models import MultimodalRCAModelV5
from src.evaluation import compute_metrics

def evaluate_on_dataset(model, dataset, device):
    """Evaluate model on a specific dataset."""
    model.eval()
    predictions = []
  
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=16):
            metrics = batch['metrics'].to(device)
            logs = batch['logs'].to(device)
            traces = batch['traces'].to(device)
          
            output = model(metrics, logs, traces)
            predictions.extend(output['ranking'][:, 0].cpu().tolist())
  
    return compute_metrics(predictions, dataset.labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--datasets', nargs='+', default=['rcaeval'])
    parser.add_argument('--output', default='results/cross_dataset_eval.json')
    args = parser.parse_args()
  
    # Load model
    model = MultimodalRCAModelV5.load(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
  
    results = {}
  
    # Evaluate on each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"Evaluating on: {dataset_name}")
        print('='*50)
      
        if dataset_name == 'rcaeval':
            dataset = RCAEvalDataset('data/rcaeval', split='test')
        elif dataset_name == 'gaia':
            adapter = GAIADatasetAdapter('data/gaia')
            cases = adapter.create_failure_cases()
            dataset = FailureCaseDataset(cases)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
      
        metrics = evaluate_on_dataset(model, dataset, device)
        results[dataset_name] = metrics
      
        print(f"Results for {dataset_name}:")
        print(f"  AC@1: {metrics['ac_at_1']:.1%}")
        print(f"  AC@3: {metrics['ac_at_3']:.1%}")
        print(f"  AC@5: {metrics['ac_at_5']:.1%}")
        print(f"  MRR:  {metrics['mrr']:.3f}")
  
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
  
    print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
```

### 6.5 Dataset Download Script

```python
# scripts/download_datasets.py

"""
Download and prepare all datasets.

Usage:
    python scripts/download_datasets.py --datasets all
"""

import os
import subprocess
from pathlib import Path

DATASETS = {
    'rcaeval': {
        'url': 'https://zenodo.org/record/14590730/files/rcaeval_re2.zip',
        'extract_to': 'data/rcaeval'
    },
    'gaia': {
        'repo': 'https://github.com/CloudWise-OpenSource/GAIA-DataSet.git',
        'extract_to': 'data/gaia'
    }
}

def download_rcaeval(dest: Path):
    """Download RCAEval dataset from Zenodo."""
    print("Downloading RCAEval dataset...")
    os.makedirs(dest, exist_ok=True)
  
    # Download
    subprocess.run([
        'curl', '-L', '-o', str(dest / 'rcaeval.zip'),
        DATASETS['rcaeval']['url']
    ])
  
    # Extract
    subprocess.run(['unzip', str(dest / 'rcaeval.zip'), '-d', str(dest)])
  
    print(f"RCAEval dataset ready at: {dest}")

def download_gaia(dest: Path):
    """Clone GAIA dataset repository."""
    print("Cloning GAIA dataset...")
  
    if dest.exists():
        print(f"GAIA already exists at {dest}, pulling updates...")
        subprocess.run(['git', 'pull'], cwd=str(dest))
    else:
        subprocess.run([
            'git', 'clone', '--depth', '1',
            DATASETS['gaia']['repo'],
            str(dest)
        ])
  
    # Download LFS files
    subprocess.run(['git', 'lfs', 'pull'], cwd=str(dest))
  
    print(f"GAIA dataset ready at: {dest}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', 
                        choices=['all', 'rcaeval', 'gaia'],
                        default=['all'])
    args = parser.parse_args()
  
    if 'all' in args.datasets:
        args.datasets = ['rcaeval', 'gaia']
  
    for dataset in args.datasets:
        dest = Path(DATASETS[dataset]['extract_to'])
      
        if dataset == 'rcaeval':
            download_rcaeval(dest)
        elif dataset == 'gaia':
            download_gaia(dest)
  
    print("\n✓ All datasets downloaded successfully!")

if __name__ == '__main__':
    main()
```

---

## 7. Implementation Roadmap

### 7.1 Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION ROADMAP                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 0: Foundation (Week 1)                                           │
│  ├── Set up LLM API access (OpenAI/Azure)                              │
│  ├── Download GAIA dataset                                              │
│  └── Create dataset adapters                                            │
│                                                                          │
│  PHASE 1: Encoder Upgrades (Week 2-3)                                   │
│  ├── Implement LLM logs encoder                                         │
│  ├── Wire up existing GCN traces encoder                               │
│  ├── Add optional Chronos metrics encoder                               │
│  └── Ablation tests for each encoder                                    │
│                                                                          │
│  PHASE 2: LLM Causal Integration (Week 4)                               │
│  ├── Implement LLM causal prior generator                              │
│  ├── Modify attention to use hybrid causal weights                     │
│  └── Tune λ_pcmci and λ_llm weights                                    │
│                                                                          │
│  PHASE 3: Explainability (Week 5)                                       │
│  ├── Implement LLM explainer module                                     │
│  ├── Create explanation templates                                       │
│  └── User study for explanation quality                                 │
│                                                                          │
│  PHASE 4: Cross-Dataset Validation (Week 6)                             │
│  ├── Evaluate on GAIA dataset                                           │
│  ├── Fine-tune on combined datasets                                     │
│  └── Report generalization metrics                                      │
│                                                                          │
│  PHASE 5: Production Readiness (Week 7-8)                               │
│  ├── API endpoint for inference                                         │
│  ├── Prometheus/Grafana integration                                     │
│  ├── Documentation and examples                                         │
│  └── Performance benchmarks                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Detailed Task Breakdown

#### Phase 0: Foundation (Week 1)

| Task                       | Priority | Effort | Deliverable                  |
| -------------------------- | -------- | ------ | ---------------------------- |
| 0.1 Set up OpenAI API      | P0       | 2h     | API key configured           |
| 0.2 Download GAIA dataset  | P0       | 1h     | Data in `data/gaia/`       |
| 0.3 Create GAIA adapter    | P0       | 4h     | `GAIADatasetAdapter` class |
| 0.4 Verify data loading    | P0       | 2h     | Unit tests passing           |
| 0.5 Set up embedding cache | P1       | 2h     | Redis/pickle cache           |

#### Phase 1: Encoder Upgrades (Week 2-3)

| Task                           | Priority | Effort | Deliverable               |
| ------------------------------ | -------- | ------ | ------------------------- |
| 1.1 LLM logs encoder           | P0       | 8h     | `LLMLogsEncoder` class  |
| 1.2 Wire GCN traces            | P0       | 4h     | Modified `rca_v5.py`    |
| 1.3 Chronos encoder (optional) | P2       | 6h     | `ChronosEncoder` class  |
| 1.4 Config system update       | P1       | 3h     | Encoder selection in YAML |
| 1.5 Encoder ablations          | P0       | 4h     | Ablation results table    |
| 1.6 TF-IDF fallback            | P1       | 2h     | No-API fallback working   |

#### Phase 2: LLM Causal Integration (Week 4)

| Task                 | Priority | Effort | Deliverable               |
| -------------------- | -------- | ------ | ------------------------- |
| 2.1 LLM causal prior | P1       | 6h     | `LLMCausalPrior` class  |
| 2.2 Hybrid attention | P1       | 4h     | Modified attention module |
| 2.3 Weight tuning    | P1       | 4h     | Optimal λ values         |
| 2.4 Caching system   | P2       | 3h     | Causal prior cache        |

#### Phase 3: Explainability (Week 5)

| Task                   | Priority | Effort | Deliverable            |
| ---------------------- | -------- | ------ | ---------------------- |
| 3.1 LLM explainer      | P1       | 8h     | `LLMExplainer` class |
| 3.2 Prompt engineering | P1       | 4h     | Optimized prompts      |
| 3.3 Explanation API    | P2       | 4h     | REST endpoint          |
| 3.4 Quality evaluation | P2       | 4h     | Human evaluation       |

#### Phase 4: Cross-Dataset Validation (Week 6)

| Task                      | Priority | Effort | Deliverable           |
| ------------------------- | -------- | ------ | --------------------- |
| 4.1 GAIA evaluation       | P0       | 4h     | Results on GAIA       |
| 4.2 Combined training     | P1       | 6h     | Multi-dataset model   |
| 4.3 Generalization report | P0       | 4h     | Cross-dataset metrics |
| 4.4 Error analysis        | P2       | 4h     | Failure case analysis |

#### Phase 5: Production Readiness (Week 7-8)

| Task                    | Priority | Effort | Deliverable              |
| ----------------------- | -------- | ------ | ------------------------ |
| 5.1 FastAPI endpoint    | P1       | 8h     | `/predict` endpoint    |
| 5.2 Prometheus exporter | P2       | 4h     | Metrics integration      |
| 5.3 Docker container    | P1       | 4h     | `Dockerfile`           |
| 5.4 Documentation       | P0       | 6h     | Updated README, API docs |
| 5.5 Benchmark suite     | P1       | 4h     | Latency/throughput tests |

### 7.3 Gantt Chart

```
Week 1    Week 2    Week 3    Week 4    Week 5    Week 6    Week 7    Week 8
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ Phase 0 │ Phase 1           │ Phase 2 │ Phase 3 │ Phase 4 │ Phase 5           │
│ Found.  │ Encoders          │ Causal  │ Explain │ Dataset │ Production        │
├─────────┼───────────────────┼─────────┼─────────┼─────────┼───────────────────┤
│ █████   │                   │         │         │         │                   │
│         │ ███████████████   │         │         │         │                   │
│         │                   │ █████   │         │         │                   │
│         │                   │         │ █████   │         │                   │
│         │                   │         │         │ █████   │                   │
│         │                   │         │         │         │ ███████████████   │
└─────────┴───────────────────┴─────────┴─────────┴─────────┴───────────────────┘

Legend: █ = Active work on phase
```

---

## 8. Expected Results

### 8.1 Performance Projections

| Phase         | AC@1 (Mean)      | AC@1 (Best)      | Explainability |
| ------------- | ---------------- | ---------------- | -------------- |
| Current (V4)  | 66.7%            | 81.5%            | None           |
| After Phase 1 | 72-75%           | 85-88%           | None           |
| After Phase 2 | 75-78%           | 87-90%           | None           |
| After Phase 3 | 75-78%           | 87-90%           | ✓ Full NL     |
| Final (V5)    | **78-82%** | **88-92%** | ✓ Full NL     |

### 8.2 Expected Improvements by Component

| Component      | Current           | New         | Expected Gain     |
| -------------- | ----------------- | ----------- | ----------------- |
| Logs Encoder   | TCN (45.6% alone) | Drain3+LLM  | +4-6%             |
| Traces Encoder | TCN (52.3% alone) | GCN         | +2-3%             |
| Causal Prior   | PCMCI only        | PCMCI + LLM | +2-3%             |
| Combined       | -                 | -           | **+11-15%** |

### 8.3 Cross-Dataset Generalization Goals

| Dataset           | Training  | Target AC@1 |
| ----------------- | --------- | ----------- |
| RCAEval (current) | ✓        | 78-82%      |
| GAIA              | Fine-tune | 70-75%      |
| Combined          | ✓        | 75-80%      |

### 8.4 Latency Budget

| Component                  | Current         | Target                       |
| -------------------------- | --------------- | ---------------------------- |
| Metrics Encoder            | 1.2ms           | 1.2ms (TCN) / 10ms (Chronos) |
| Logs Encoder               | 1.8ms           | 5-50ms (with LLM API)        |
| Traces Encoder             | 1.5ms           | 1.5ms                        |
| Fusion + Attention         | 0.4ms           | 0.5ms                        |
| **Total (no LLM)**   | **3.3ms** | **~5ms**               |
| **Total (with LLM)** | N/A             | **~50-100ms**          |
| LLM Explanation            | N/A             | ~500ms (optional, async)     |

---

## 9. Technical Specifications

### 9.1 New Dependencies

```txt
# requirements_v5.txt (additions to existing)

# LLM Integration
openai>=1.0.0
tiktoken>=0.5.0
langchain>=0.1.0

# Optional: Local LLM
sentence-transformers>=2.2.0
transformers>=4.30.0

# Optional: Chronos
chronos-forecasting>=0.1.0

# Graph Neural Networks (already have, ensure version)
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0

# Caching
redis>=4.0.0  # Optional, for distributed cache
diskcache>=5.0.0

# API
fastapi>=0.100.0
uvicorn>=0.22.0
```

### 9.2 Configuration Schema

```yaml
# config/model_config_v5.yaml

model:
  name: "MultimodalRCAModel_V5"
  version: "5.0.0"
  
  # Metrics Encoder
  metrics_encoder:
    type: "tcn"  # Options: tcn, chronos
    tcn:
      n_features: 64
      hidden_dim: 48
      embed_dim: 64
      num_layers: 2
      dropout: 0.35
    chronos:
      model_name: "amazon/chronos-bolt-tiny"
      freeze_backbone: true
      embed_dim: 64
  
  # Logs Encoder
  logs_encoder:
    type: "llm"  # Options: llm, tfidf
    llm:
      provider: "openai"
      model: "text-embedding-3-small"
      embed_dim: 64
      cache_enabled: true
    tfidf:
      max_features: 1000
      embed_dim: 64
    drain3:
      similarity_threshold: 0.5
      depth: 4
  
  # Traces Encoder
  traces_encoder:
    type: "gcn"  # Options: gcn, gat, tcn
    gcn:
      in_channels: 8
      hidden_channels: 64
      embed_dim: 64
      num_layers: 2
      dropout: 0.3
  
  # Fusion
  fusion:
    strategy: "gated"
    embed_dim: 128
  
  # Attention
  attention:
    num_layers: 2
    num_heads: 4
    dropout: 0.35
    causal_injection:
      pcmci_weight: 0.3
      llm_weight: 0.2  # New
  
  # Causal Discovery
  causal:
    pcmci:
      tau_max: 3
      alpha_level: 0.05
    llm_prior:
      enabled: true
      model: "gpt-4o"
      cache_enabled: true

# LLM Configuration
llm:
  embeddings:
    provider: "openai"
    model: "text-embedding-3-small"
    api_key_env: "OPENAI_API_KEY"
  reasoning:
    provider: "openai"
    model: "gpt-4o-mini"
  cache:
    type: "disk"  # Options: disk, redis, none
    path: "outputs/llm_cache"

# Explainability
explainability:
  enabled: true
  model: "gpt-4o-mini"
  max_tokens: 500
  temperature: 0.3
```

### 9.3 API Specification

```python
# API Endpoints

# POST /predict
# Request:
{
    "metrics": [[...], [...], ...],  # (n_services, seq_len, n_features)
    "logs": ["log1", "log2", ...],   # Raw log strings
    "traces": {                       # Service call graph
        "edges": [["svc1", "svc2"], ...],
        "node_features": {...}
    },
    "explain": true,                  # Optional: generate explanation
    "system_name": "TrainTicket"      # Optional: for context
}

# Response:
{
    "ranking": ["payment-svc", "db-svc", "frontend-svc", ...],
    "probabilities": [0.72, 0.15, 0.08, ...],
    "predicted_root_cause": "payment-svc",
    "confidence": 0.72,
    "explanation": "payment-svc is the root cause because...",  # If explain=true
    "inference_time_ms": 52.3
}
```

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk                  | Probability | Impact | Mitigation                             |
| --------------------- | ----------- | ------ | -------------------------------------- |
| LLM API latency       | High        | Medium | Caching + async calls                  |
| LLM API costs         | Medium      | Medium | Local fallback (sentence-transformers) |
| GCN memory issues     | Low         | High   | Batch size tuning                      |
| Chronos compatibility | Medium      | Low    | Optional, TCN default                  |
| GAIA format changes   | Low         | Medium | Version-locked adapter                 |

### 10.2 Research Risks

| Risk                               | Probability | Impact | Mitigation                   |
| ---------------------------------- | ----------- | ------ | ---------------------------- |
| LLM embeddings don't help          | Medium      | High   | Ablation study early         |
| Cross-dataset generalization fails | Medium      | High   | Domain adaptation techniques |
| LLM causal priors are noisy        | Medium      | Medium | Low λ_llm weight            |

### 10.3 Fallback Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FALLBACK STRATEGIES                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LLM API Unavailable:                                                   │
│  ├── Logs: Fall back to TF-IDF embeddings                              │
│  ├── Causal: Use PCMCI only (λ_llm = 0)                               │
│  └── Explain: Return structured JSON instead of NL                      │
│                                                                          │
│  Chronos Too Slow:                                                       │
│  └── Use TCN encoder (default)                                          │
│                                                                          │
│  GCN OOM:                                                                │
│  └── Fall back to TCN traces encoder                                    │
│                                                                          │
│  GAIA Data Issues:                                                       │
│  └── Train on RCAEval only, report as limitation                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Appendix

### A. File Structure After Implementation

```
project/
├── config/
│   ├── model_config_v5.yaml       # NEW: V5 configuration
│   ├── llm_config.yaml            # NEW: LLM settings
│   └── ...
├── src/
│   ├── encoders/
│   │   ├── llm_logs_encoder.py    # NEW: LLM-based logs
│   │   ├── chronos_encoder.py     # NEW: Chronos metrics
│   │   ├── gcn_encoder.py         # MODIFIED: Wire into model
│   │   └── ...
│   ├── causal/
│   │   ├── llm_causal_prior.py    # NEW: LLM causal knowledge
│   │   └── ...
│   ├── explainability/
│   │   └── llm_explainer.py       # NEW: NL explanations
│   ├── models/
│   │   └── rca_v5_multimodal.py   # NEW: V5 model
│   └── data/
│       ├── gaia_adapter.py        # NEW: GAIA dataset
│       └── ...
├── scripts/
│   ├── download_datasets.py       # NEW: Dataset downloader
│   ├── evaluate_cross_dataset.py  # NEW: Cross-dataset eval
│   └── ...
├── api/
│   └── main.py                    # NEW: FastAPI server
└── data/
    ├── rcaeval/                   # Existing
    └── gaia/                      # NEW: GAIA dataset
```

### B. Command Reference

```bash
# Download all datasets
python scripts/download_datasets.py --datasets all

# Train V5 model
python scripts/train_multimodal_v5.py \
    --config config/model_config_v5.yaml \
    --logs-encoder llm \
    --traces-encoder gcn

# Cross-dataset evaluation
python scripts/evaluate_cross_dataset.py \
    --checkpoint outputs/v5_best.pt \
    --datasets rcaeval gaia

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run with explanation
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"metrics": [...], "logs": [...], "explain": true}'
```

### C. Cost Estimation (LLM API)

| Component       | API Calls            | Cost per 1K cases         |
| --------------- | -------------------- | ------------------------- |
| Log Embeddings  | ~50K embeddings      | $1.00                     |
| Causal Prior    | ~1K queries (cached) | $0.50                     |
| Explanations    | ~1K queries          | $2.00                     |
| **Total** | -                    | **~$3.50/1K cases** |

*Note: With caching, training costs are one-time. Inference costs ~$0.003/case.*

---

## 12. References

1. Runge, J., et al. (2019). "Detecting and quantifying causal associations in large nonlinear time series datasets." Science Advances.
2. GAIA Dataset: https://github.com/CloudWise-OpenSource/GAIA-DataSet
3. RCAEval: Pham, D., et al. (2024). "RCAEval: A Benchmark for Root Cause Analysis."
4. Chronos: Amazon (2024). "Chronos: Learning the Language of Time Series."
5. Drain3: https://github.com/logpai/Drain3

---

**Document Status:** Draft v1.0
**Next Review:** After Phase 1 completion
**Contact:** parth.gupta@snu.edu.in
