**Table 2: Ablation study - component contributions**

| Configuration                               |   AC@1 |   AC@3 |   AC@5 | Δ vs Full       |
|:--------------------------------------------|-------:|-------:|-------:|:----------------|
| Full Multimodal                             |  0.630 |  0.815 |  1.000 | baseline        |
| Metrics Only (no logs/traces)               |  0.526 |  0.712 |  0.850 | -10.4%          |
| No Gated Fusion (concat)                    |  0.581 |  0.778 |  0.963 | -4.9%           |
| No Causal Weights                           |  0.593 |  0.789 |  0.963 | -3.7%           |
| No Cross-Service Attention                  |  0.556 |  0.741 |  0.926 | -7.4%           |