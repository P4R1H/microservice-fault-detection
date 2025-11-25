**Table 7: Complete model architecture and hyperparameters**

| Component        | Specification     | Details                       |
|:-----------------|:------------------|:------------------------------|
| Metrics Encoder  | Chronos-Bolt-Tiny | 20M params, 98MB              |
| Logs Encoder     | Drain3 + TF-IDF   | 1247 templates                |
| Traces Encoder   | 2-layer GCN       | 128d hidden, mean aggregation |
| Causal Discovery | PCMCI             | tau_max=5, ParCorr test       |
| Fusion           | Cross-Attention   | 8 heads, 3 layers             |
| Training         | AdamW             | LR=0.0001, 50 epochs          |
| Total Parameters | -                 | 24.7M (4.7M trainable)        |