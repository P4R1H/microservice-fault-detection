**Table 2: Ablation study - incremental component contributions**

| Configuration                               |   AC@1 |   AC@3 |   AC@5 | Δ vs Baseline   |
|:--------------------------------------------|-------:|-------:|-------:|:----------------|
| Metrics Only (Chronos)                      |  0.581 |  0.734 |  0.823 | +0.0%           |
| Logs Only (Drain3)                          |  0.456 |  0.612 |  0.734 | -21.5%          |
| Traces Only (GCN)                           |  0.523 |  0.678 |  0.789 | -10.0%          |
| Metrics + Logs                              |  0.647 |  0.798 |  0.867 | +11.4%          |
| Metrics + Traces                            |  0.689 |  0.834 |  0.901 | +18.6%          |
| Logs + Traces                               |  0.612 |  0.756 |  0.845 | +5.3%           |
| All Modalities (No Causal)                  |  0.712 |  0.856 |  0.923 | +22.5%          |
| All + PCMCI (No Cross-Attention)            |  0.734 |  0.871 |  0.932 | +26.3%          |
| Full System (All + PCMCI + Cross-Attention) |  0.761 |  0.887 |  0.941 | +31.0%          |