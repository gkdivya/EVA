# Advanced Convolution experiments
1. Basic skeleton with sequential format (with GAP), 15 epoch, reduce the number of parameters 2, with batch norm.

- Targets: Baseline working model with Batch Normalization and Global averaging pool.
- Results:

| Training Accuracy | 99.72% | GAP = Kernel\_Size (7)Epoch = 15.Batch Normalization = YesTrainable Parameters= 44,420 |
| --- | --- | --- |
| Test Accuracy | 99.38% |

- Analysis:
  - Increase in number of trainable parameters after adding multiple Batch-Norm layers.
  - For each Batch-Norm layer, you can notice number of parameters are double the number of output channels. Eg. For layer BatchNorm2d-2, there are 16 output channels hence corresponding to those trainable parameters are 32.
  - For Batch-Norm layer, we could notice input shape and output shape both are same.
  - Batch Normalisation can increase the model capacity in fewer training steps as compared to without Batch-Norm model and hence it making model learn faster.
  - Model is Overfitted.

- File Link:

1. Adding Drop out layers on Baseline model.

- Targets: Baseline working model with Batch Normalization, Global averaging pool &amp; Drop out layers.
- Results:

| Training Accuracy | 99.60% | GAP = Kernel\_Size (7)Drop out = 0.10Epoch = 15.Batch Normalization = YesTrainable Parameters=44,420 |
| --- | --- | --- |
| Test Accuracy | 99.23% |

- Analysis:
- File Link:

1. Adding Fully Connected Layers on top of Baseline model with Drop out layers.

- Targets: Baseline working model with Batch Normalization, Global averaging pool, Drop out layers &amp; FC layers.
- Results:

| Training Accuracy | 99.02% | GAP = Kernel\_Size (7)Epoch = 15.Batch Normalization =YesDrop out = 0.10FC Layer = YesTrainable Parameters=44,640 |
| --- | --- | --- |
| Test Accuracy | 99.28% |

- Analysis:
- File Link:
