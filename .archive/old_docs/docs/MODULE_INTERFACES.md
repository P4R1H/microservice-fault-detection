# Module Interfaces Specification

This document defines the interfaces and data flow between all components of the multimodal RCA system.

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA LAYER                          │
│  RCAEvalDataLoader → FailureCase(metrics, logs, traces, label)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Metrics    │  │     Logs     │  │    Traces    │          │
│  │ Preprocessor │  │   Parser     │  │  Graph       │          │
│  │              │  │   (Drain3)   │  │  Builder     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODALITY ENCODERS                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Chronos    │  │   Drain3     │  │     GCN      │          │
│  │  Bolt-Tiny   │  │  Template    │  │   Encoder    │          │
│  │   Encoder    │  │  Embeddings  │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CAUSAL DISCOVERY                              │
│                  PCMCI / PCMCIplus                               │
│              (Temporal causal graphs)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MULTIMODAL FUSION                               │
│          Cross-modal attention + Feature fusion                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RCA PREDICTION                                │
│          Service ranking → Root cause localization               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Data Structures

### 2.1 FailureCase
```python
@dataclass
class FailureCase:
    """Single failure case from RCAEval dataset"""
    case_id: str
    system: str  # 'TrainTicket', 'SockShop', 'OnlineBoutique'
    fault_type: str  # 'CPU', 'MEM', 'DISK', 'SOCKET', 'DELAY', 'LOSS'
    root_cause_service: str
    root_cause_indicator: str

    metrics: pd.DataFrame  # Shape: (timesteps, n_metrics)
    logs: pd.DataFrame     # Columns: ['timestamp', 'service', 'level', 'message']
    traces: pd.DataFrame   # Columns: ['span_id', 'service', 'latency', 'parent_span']

    timestamp: pd.Timestamp
    duration_minutes: int
```

### 2.2 ModalityEmbedding
```python
@dataclass
class ModalityEmbedding:
    """Embedding from a single modality encoder"""
    modality: str  # 'metrics', 'logs', 'traces'
    embedding: torch.Tensor  # Shape: (batch_size, embedding_dim)
    attention_weights: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None
```

### 2.3 RCAResult
```python
@dataclass
class RCAResult:
    """Root cause analysis result"""
    case_id: str
    predicted_ranking: List[Tuple[str, float]]  # [(service, score), ...]
    ground_truth_service: str

    # Evaluation metrics
    ac_at_1: float
    ac_at_3: float
    ac_at_5: float
    mrr: float
    rank: int

    # Interpretability
    causal_graph: Optional[np.ndarray] = None
    attention_maps: Optional[Dict[str, torch.Tensor]] = None
    anomaly_scores: Optional[Dict[str, float]] = None
```

---

## 3. Module Interfaces

### 3.1 Data Loading Module
**Location**: `src/utils/data_loader.py`

```python
class RCAEvalDataLoader:
    """Load and preprocess RCAEval dataset"""

    def __init__(self, data_dir: str):
        """Initialize data loader"""
        pass

    def load_all_cases(self, systems: List[str] = None) -> List[FailureCase]:
        """Load all failure cases"""
        pass

    def load_splits(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[List[FailureCase], List[FailureCase], List[FailureCase]]:
        """Load train/val/test splits"""
        pass
```

**Input**: Path to RCAEval dataset directory
**Output**: List of `FailureCase` objects
**Dependencies**: pandas, numpy

---

### 3.2 Preprocessing Module
**Location**: `src/preprocessing/`

#### 3.2.1 Metrics Preprocessor
```python
class MetricsPreprocessor:
    """Preprocess metrics data for Chronos/TCN"""

    def __init__(
        self,
        normalize: bool = True,
        fill_method: str = 'forward',
        window_size: int = 60
    ):
        """Initialize preprocessor"""
        pass

    def preprocess(self, metrics: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess metrics time series

        Args:
            metrics: Raw metrics DataFrame (timesteps × metrics)

        Returns:
            Preprocessed tensor: (batch=1, seq_len, n_features)
        """
        pass

    def extract_service_metrics(
        self,
        metrics: pd.DataFrame,
        service_mapping: Dict[str, List[str]]
    ) -> Dict[str, pd.DataFrame]:
        """Group metrics by service"""
        pass
```

**Input**: Raw metrics DataFrame
**Output**: Normalized tensor ready for Chronos/TCN
**Dependencies**: pandas, torch, sklearn

#### 3.2.2 Log Parser
```python
class LogParser:
    """Parse logs using Drain3 algorithm"""

    def __init__(
        self,
        depth: int = 4,
        similarity_threshold: float = 0.4,
        max_clusters: int = 1024
    ):
        """Initialize Drain3 parser"""
        pass

    def parse(self, logs: pd.DataFrame) -> LogParseResult:
        """
        Parse log messages into templates

        Args:
            logs: Raw logs DataFrame

        Returns:
            LogParseResult with template IDs and embeddings
        """
        pass

    def get_template_embeddings(self, template_ids: List[int]) -> torch.Tensor:
        """Convert template IDs to embeddings"""
        pass
```

**Input**: Raw log DataFrame
**Output**: Template IDs and embeddings
**Dependencies**: drain3, torch

#### 3.2.3 Trace Graph Builder
```python
class TraceGraphBuilder:
    """Build service dependency graph from traces"""

    def __init__(self):
        """Initialize graph builder"""
        pass

    def build_graph(self, traces: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Build adjacency matrix from traces

        Args:
            traces: Raw traces DataFrame

        Returns:
            (adjacency_matrix, service_names)
        """
        pass

    def extract_node_features(
        self,
        traces: pd.DataFrame,
        services: List[str]
    ) -> torch.Tensor:
        """Extract node features (latency, error rate, call count)"""
        pass
```

**Input**: Raw traces DataFrame
**Output**: Adjacency matrix + node features
**Dependencies**: pandas, numpy, torch

---

### 3.3 Encoder Module
**Location**: `src/models/encoders/`

#### 3.3.1 Metrics Encoder (Chronos)
```python
class ChronosMetricsEncoder(nn.Module):
    """Encode metrics using Chronos-Bolt-Tiny"""

    def __init__(
        self,
        model_name: str = "amazon/chronos-bolt-tiny",
        embedding_dim: int = 256,
        freeze_backbone: bool = True
    ):
        """Initialize Chronos encoder"""
        pass

    def forward(
        self,
        metrics: torch.Tensor,  # (batch, seq_len, n_features)
        service_mask: Optional[torch.Tensor] = None
    ) -> ModalityEmbedding:
        """
        Encode metrics time series

        Returns:
            ModalityEmbedding with shape (batch, embedding_dim)
        """
        pass
```

**Input**: Preprocessed metrics tensor
**Output**: `ModalityEmbedding` object
**Dependencies**: torch, transformers, chronos-forecasting

#### 3.3.2 Logs Encoder
```python
class LogsEncoder(nn.Module):
    """Encode log templates"""

    def __init__(
        self,
        vocab_size: int = 1024,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 2
    ):
        """Initialize logs encoder"""
        pass

    def forward(
        self,
        template_ids: torch.Tensor,  # (batch, seq_len)
        timestamps: torch.Tensor
    ) -> ModalityEmbedding:
        """
        Encode log template sequence

        Returns:
            ModalityEmbedding with shape (batch, embedding_dim)
        """
        pass
```

**Input**: Template IDs from Drain3
**Output**: `ModalityEmbedding` object
**Dependencies**: torch

#### 3.3.3 Traces Encoder (GCN)
```python
class TracesGCNEncoder(nn.Module):
    """Encode service dependency graph using GCN"""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        n_layers: int = 2
    ):
        """Initialize GCN encoder"""
        pass

    def forward(
        self,
        node_features: torch.Tensor,  # (n_nodes, input_dim)
        adjacency_matrix: torch.Tensor  # (n_nodes, n_nodes)
    ) -> ModalityEmbedding:
        """
        Encode service dependency graph

        Returns:
            ModalityEmbedding with shape (n_nodes, output_dim)
        """
        pass
```

**Input**: Node features + adjacency matrix
**Output**: `ModalityEmbedding` object
**Dependencies**: torch, torch_geometric

---

### 3.4 Causal Discovery Module
**Location**: `src/models/causal/`

```python
class PCMCICausalDiscovery:
    """PCMCI causal discovery for temporal data"""

    def __init__(
        self,
        tau_max: int = 5,
        alpha_level: float = 0.01,
        method: str = 'pcmci'  # or 'pcmciplus'
    ):
        """Initialize PCMCI"""
        pass

    def discover_causal_graph(
        self,
        metrics: pd.DataFrame,
        service_mapping: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Discover causal relationships between services

        Args:
            metrics: Time series metrics
            service_mapping: Mapping of services to metrics

        Returns:
            (causal_adjacency_matrix, service_names)
        """
        pass

    def rank_by_causality(
        self,
        causal_graph: np.ndarray,
        service_names: List[str],
        anomaly_scores: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank services by causal influence

        Returns:
            List of (service, causal_score)
        """
        pass
```

**Input**: Metrics time series
**Output**: Causal adjacency matrix
**Dependencies**: tigramite, numpy

---

### 3.5 Fusion Module
**Location**: `src/models/fusion/`

```python
class MultimodalFusionModule(nn.Module):
    """Intermediate multimodal fusion with cross-modal attention"""

    def __init__(
        self,
        embedding_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """Initialize fusion module"""
        pass

    def forward(
        self,
        metrics_emb: ModalityEmbedding,
        logs_emb: ModalityEmbedding,
        traces_emb: ModalityEmbedding
    ) -> torch.Tensor:
        """
        Fuse multimodal embeddings

        Args:
            metrics_emb: Metrics embedding (batch, embedding_dim)
            logs_emb: Logs embedding (batch, embedding_dim)
            traces_emb: Traces embedding (n_services, embedding_dim)

        Returns:
            Fused embedding (n_services, embedding_dim)
        """
        pass
```

**Input**: Three `ModalityEmbedding` objects
**Output**: Fused embedding tensor
**Dependencies**: torch

---

### 3.6 RCA Module
**Location**: `src/models/rca/`

```python
class MultimodalRCAModel(nn.Module):
    """Complete multimodal RCA system"""

    def __init__(
        self,
        metrics_encoder: ChronosMetricsEncoder,
        logs_encoder: LogsEncoder,
        traces_encoder: TracesGCNEncoder,
        fusion_module: MultimodalFusionModule,
        causal_module: Optional[PCMCICausalDiscovery] = None
    ):
        """Initialize RCA model"""
        pass

    def forward(
        self,
        metrics: torch.Tensor,
        logs: torch.Tensor,
        traces: Tuple[torch.Tensor, torch.Tensor],  # (node_features, adj_matrix)
        service_names: List[str]
    ) -> RCAResult:
        """
        Perform root cause analysis

        Returns:
            RCAResult with service ranking and metrics
        """
        pass

    def rank_services(
        self,
        fused_embedding: torch.Tensor,
        service_names: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rank services by anomaly likelihood

        Returns:
            List of (service, score) sorted by score
        """
        pass
```

**Input**: Preprocessed multimodal data
**Output**: `RCAResult` object
**Dependencies**: torch, all encoders

---

## 4. Baseline Interfaces

### 4.1 Statistical Baselines
**Location**: `src/baselines/statistical_baselines.py`

```python
class ThreeSigmaDetector:
    def rank_services(
        self,
        metrics: pd.DataFrame,
        service_mapping: Dict[str, str]
    ) -> List[Tuple[str, float]]:
        """Rank services by 3-sigma anomaly score"""
        pass

class ARIMAForecaster:
    def rank_services(
        self,
        metrics: pd.DataFrame,
        service_mapping: Dict[str, str]
    ) -> List[Tuple[str, float]]:
        """Rank services by ARIMA residuals"""
        pass

class GrangerLassoRCA:
    def rank_services(
        self,
        metrics: pd.DataFrame,
        service_mapping: Dict[str, str]
    ) -> List[Tuple[str, float]]:
        """Rank services by Granger causality"""
        pass
```

### 4.2 ML Baselines
**Location**: `src/baselines/ml_baselines.py`

```python
class RandomForestRCA:
    """Random Forest for RCA (Phase 1 baseline)"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Random Forest"""
        pass

    def predict(self, X: np.ndarray) -> str:
        """Predict root cause service"""
        pass

    def rank_services(self, X: np.ndarray) -> List[Tuple[str, float]]:
        """Rank services by feature importance"""
        pass
```

---

## 5. Evaluation Module
**Location**: `src/evaluation/metrics.py`

```python
def evaluate_ranking(
    predicted_ranking: List[Tuple[str, float]],
    ground_truth_service: str,
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Evaluate root cause localization

    Returns:
        {'AC@1': ..., 'AC@3': ..., 'AC@5': ..., 'Avg@5': ..., 'MRR': ...}
    """
    pass

def aggregate_results(
    results: List[RCAResult],
    groupby: Optional[str] = None  # 'system', 'fault_type', etc.
) -> pd.DataFrame:
    """Aggregate evaluation results"""
    pass

def statistical_significance_test(
    method_a_results: List[float],
    method_b_results: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Perform paired t-test"""
    pass
```

---

## 6. Training Pipeline
**Location**: `src/training/trainer.py`

```python
class RCATrainer:
    """Training loop for RCA model"""

    def __init__(
        self,
        model: MultimodalRCAModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
        """Initialize trainer"""
        pass

    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """Train one epoch"""
        pass

    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate model"""
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        pass
```

---

## 7. Data Flow Example

### End-to-End Inference Flow

```python
# 1. Load data
loader = RCAEvalDataLoader('data/RCAEval')
train, val, test = loader.load_splits()

# 2. Preprocess single case
case = test[0]
metrics_preprocessor = MetricsPreprocessor()
log_parser = LogParser()
graph_builder = TraceGraphBuilder()

metrics_tensor = metrics_preprocessor.preprocess(case.metrics)
log_templates = log_parser.parse(case.logs)
adj_matrix, node_features = graph_builder.build_graph(case.traces)

# 3. Initialize model
model = MultimodalRCAModel(
    metrics_encoder=ChronosMetricsEncoder(),
    logs_encoder=LogsEncoder(),
    traces_encoder=TracesGCNEncoder(),
    fusion_module=MultimodalFusionModule()
)

# 4. Inference
result = model(
    metrics=metrics_tensor,
    logs=log_templates,
    traces=(node_features, adj_matrix),
    service_names=service_list
)

# 5. Evaluate
metrics = evaluate_ranking(
    result.predicted_ranking,
    case.root_cause_service
)

print(f"AC@1: {metrics['AC@1']}, MRR: {metrics['MRR']}")
```

---

## 8. Configuration Management

### Config Schema
**Location**: `configs/model_config.yaml`

```yaml
model:
  metrics_encoder:
    type: "chronos"  # or "tcn"
    model_name: "amazon/chronos-bolt-tiny"
    embedding_dim: 256
    freeze_backbone: true

  logs_encoder:
    vocab_size: 1024
    embedding_dim: 256
    hidden_dim: 512

  traces_encoder:
    input_dim: 64
    hidden_dim: 128
    output_dim: 256
    n_layers: 2

  fusion:
    n_heads: 8
    dropout: 0.1

  causal:
    method: "pcmci"
    tau_max: 5
    alpha_level: 0.01

training:
  batch_size: 16
  learning_rate: 0.0001
  n_epochs: 50
  early_stopping_patience: 10
  device: "cuda"

data:
  train_ratio: 0.6
  val_ratio: 0.2
  random_seed: 42
```

---

## 9. Testing Interfaces

Each module should implement unit tests:

```python
# tests/test_encoders.py
def test_chronos_encoder():
    encoder = ChronosMetricsEncoder()
    metrics = torch.randn(2, 100, 10)  # (batch, seq_len, features)
    output = encoder(metrics)
    assert output.embedding.shape == (2, 256)

# tests/test_fusion.py
def test_multimodal_fusion():
    fusion = MultimodalFusionModule()
    metrics_emb = ModalityEmbedding('metrics', torch.randn(2, 256))
    logs_emb = ModalityEmbedding('logs', torch.randn(2, 256))
    traces_emb = ModalityEmbedding('traces', torch.randn(10, 256))

    fused = fusion(metrics_emb, logs_emb, traces_emb)
    assert fused.shape[1] == 256
```

---

## 10. Extensibility Points

### Adding New Encoders
1. Inherit from `nn.Module`
2. Implement `forward()` returning `ModalityEmbedding`
3. Register in `src/models/encoders/__init__.py`

### Adding New Baselines
1. Implement `rank_services()` method
2. Add to `src/baselines/`
3. Update evaluation pipeline

### Adding New Evaluation Metrics
1. Add to `src/evaluation/metrics.py`
2. Update `RCAResult` dataclass
3. Update visualization functions

---

## 11. Summary

This interface specification ensures:
- ✅ **Modularity**: Each component has clear inputs/outputs
- ✅ **Testability**: All modules can be unit tested independently
- ✅ **Extensibility**: Easy to add new encoders, baselines, metrics
- ✅ **Type Safety**: Dataclasses enforce structure
- ✅ **Documentation**: Each interface is self-documenting

**Next Steps**:
1. Implement preprocessing modules (Phase 2)
2. Implement encoder modules (Phase 3-5)
3. Implement fusion module (Phase 6)
4. Integrate all components (Phase 7)
5. Run ablation studies (Phase 8)
