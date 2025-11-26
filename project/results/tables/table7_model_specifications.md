**Table 7: Complete model architecture and hyperparameters**

| Component        | Specification          | Details                            |
|:-----------------|:-----------------------|:-----------------------------------|
| Metrics Encoder  | Depthwise Sep. TCN     | 2 layers, 48 hidden, 64 features   |
| Logs Encoder     | Depthwise Sep. TCN     | 2 layers, 48 hidden, 32 features   |
| Traces Encoder   | Depthwise Sep. TCN     | 2 layers, 48 hidden, 32 features   |
| Causal Discovery | PCMCI                  | tau_max=3, ParCorr test            |
| Fusion           | Gated + Cross-Attention| 4 heads, 2 layers, λ=0.3           |
| Training         | AdamW                  | LR=1e-3, 100 epochs, early stop    |
| Regularization   | Dropout + Weight Decay | 0.35 dropout, 0.01 weight decay    |
| Total Parameters | Large config           | 722K trainable                     |