**Table 5: Comparison of different encoder architectures**

| Encoder Configuration                |   AC@1 |   AC@3 |   AC@5 |   Time (s) |
|:-------------------------------------|-------:|-------:|-------:|-----------:|
| Full System (Chronos + GCN + Drain3) |  0.761 |  0.887 |  0.941 |      0.923 |
| TCN (instead of Chronos)             |  0.743 |  0.876 |  0.934 |      0.456 |
| GAT (instead of GCN)                 |  0.768 |  0.891 |  0.945 |      1.123 |
| BERT (instead of Drain3)             |  0.754 |  0.883 |  0.938 |      2.345 |