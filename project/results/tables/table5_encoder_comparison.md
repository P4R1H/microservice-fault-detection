**Table 5: Comparison of different model configurations**

| Configuration                        |   AC@1 |   AC@3 |   AC@5 |   Time (ms) |
|:-------------------------------------|-------:|-------:|-------:|------------:|
| TCN Small (324K params)              |  0.611 |  0.824 |  1.000 |         3.2 |
| TCN Large (722K params)              |  0.667 |  0.877 |  1.000 |         3.3 |
| With Gated Fusion                    |  0.667 |  0.877 |  1.000 |         3.3 |
| Without Gated Fusion (concat)        |  0.581 |  0.778 |  0.963 |         3.1 |