#Layer Normalization

Logs summary for LN experiment with different combinations. 

|    Regularization   | Batch Size | L1 Factor | L2 Factor | Train Accuracy | Test Accuracy |
|:-------------------:|:----------:|:---------:|:---------:|:--------------:|:-------------:|
|      LayerNorm      |     64     |     0     |     0     |      98.80     |     99.48     |
|      LayerNorm+L2   |     64     |     0     |   0.001   |     <b> 98.94 <b>    |     <b> 99.56 <b>    |
|      LayerNorm+L2   |     64     |     0     |   0.002   |      98.86     |     99.42     |
|      LayerNorm+L1   |     64     |   0.001   |   0.001   |      98.30     |     99.36     |
| LayerNorm + L1 + L2 |     64     |   0.001   |     0     |      98.25     |     99.14     |
| LayerNorm + L1 + L2 |     64     |   0.001   |   0.002   |      98.29     |     99.38     |
|      LayerNorm+L1   |     64     |   0.002   |     0     |      97.69     |     99.09     |
| LayerNorm + L1 + L2 |     64     |   0.002   |   0.001   |      97.72     |     99.14     |
| LayerNorm + L1 + L2 |     64     |   0.002   |   0.002   |      97.78     |     99.14     |
|      LayerNorm      |     128    |     0     |     0     |      98.81     |     99.45     |
|      LayerNorm+L2   |     128    |     0     |   0.001   |      98.90     |     99.47     |
|      LayerNorm+L2   |     128    |     0     |   0.002   |      98.89     |     99.50     |
|      LayerNorm+L1   |     128    |   0.001   |     0     |      98.38     |     99.31     |
| LayerNorm + L1 + L2 |     128    |   0.001   |   0.001   |      98.55     |     99.43     |
| LayerNorm + L1 + L2 |     128    |   0.001   |   0.002   |      98.39     |     99.27     |
|      LayerNorm + L1 |     128    |   0.002   |     0     |      98.03     |     99.29     |
| LayerNorm + L1 + L2 |     128    |   0.002   |   0.001   |      97.94     |     99.23     |
| LayerNorm + L1 + L2 |     128    |   0.002   |   0.002   |      98.03     |     99.27     |
