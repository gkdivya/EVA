#Batch Normalization

Logs summary for BN experiment with different combinations. 

|    Regularization   | Batch Size | L1 Factor | L2 Factor | Train Accuracy | Test Accuracy |
|:-------------------:|:----------:|:---------:|:---------:|:--------------:|:-------------:|
|      BatchNorm      |     64     |     0     |     0     |      98.59     |     99.50     |
|      BatchNorm      |     128    |     0     |     0     |      98.56     |     99.48     |
|    BatchNorm + L1   |     64     |   0.001   |     0     |      97.96     |     99.43     |
|    BatchNorm + L1   |     64     |   0.002   |     0     |      97.36     |     99.16     |
|    BatchNorm + L1   |     128    |   0.001   |     0     |      98.35     |     99.47     |
|    BatchNorm + L1   |     128    |   0.002   |     0     |      97.72     |     99.36     |
|    BatchNorm + L2   |     64     |     0     |   0.001   |      98.73     |    <b> 99.61 <b>    |
|    BatchNorm + L2   |     64     |     0     |   0.002   |      98.61     |      99.5     |
|    BatchNorm + L2   |     128    |     0     |   0.001   |      98.7      |     99.53     |
|    BatchNorm + L2   |     128    |     0     |   0.002   |      98.69     |     99.48     |
| BatchNorm + L1 + L2 |     64     |   0.001   |   0.001   |      98.00     |     99.24     |
| BatchNorm + L1 + L2 |     64     |   0.001   |   0.002   |      97.99     |     99.43     |
| BatchNorm + L1 + L2 |     64     |   0.002   |   0.001   |      97.36     |     99.24     |
| BatchNorm + L1 + L2 |     64     |   0.002   |   0.002   |      97.35     |     99.09     |
| BatchNorm + L1 + L2 |     128    |   0.001   |   0.001   |      98.33     |     99.32     |
| BatchNorm + L1 + L2 |     128    |   0.001   |   0.002   |      98.22     |     99.43     |
| BatchNorm + L1 + L2 |     128    |   0.002   |   0.001   |      97.77     |     99.36     |
| BatchNorm + L1 + L2 |     128    |   0.002   |   0.002   |      97.88     |     99.24     |
