## Finetune CNN Architecture on MNIST dataset
- Achieve 99.4% accuracy on test data (this must be consistently shown in last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 10000 or 8000 Parameters 
- Do this in exactly 3 or more steps
- Each File must have "target, result, analysis" TEXT block (either at the start or the end)
- You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 

## Receptive Field calculation

| Operation   | nin | Ch_in | Ch_Out | padding | kernel | stride | nout | jin | jout | rin | rout | Activation\_Size | Params# |
| ----------- | --- | ----- | ------ | ------- | ------ | ------ | ---- | --- | ---- | --- | ---- | ---------------- | ------- |
| Convolution | 28  | 1     | 8      | 0       | 3      | 1      | 26   | 1   | 1    | 1   | 3    | 784              | 72      |
| Convolution | 26  | 8     | 16     | 0       | 3      | 1      | 24   | 1   | 1    | 3   | 5    | 5408             | 1152    |
| Max-Pooling | 24  | 16    | 16     | 0       | 2      | 2      | 12   | 1   | 2    | 5   | 6    | 9216             | 0       |
| Convolution | 12  | 16    | 8      | 0       | 1      | 1      | 12   | 2   | 2    | 6   | 12   | 2304             | 128     |
| Convolution | 12  | 8     | 10     | 0       | 3      | 1      | 10   | 2   | 2    | 12  | 28   | 1152             | 720     |
| Convolution | 10  | 10    | 16     | 0       | 3      | 1      | 8    | 2   | 2    | 28  | 60   | 1000             | 1440    |
| Convolution | 8   | 16    | 16     | 0       | 3      | 1      | 6    | 2   | 2    | 60  | 124  | 1024             | 2304    |
| Convolution | 6   | 16    | 16     | 0       | 3      | 1      | 4    | 2   | 2    | 124 | 252  | 576              | 2304    |

## Steps:
1. basic skeleton with sequential format(with GAP), 15 epoch, reduce the number of parameter
2, with batch norm
3. with drop out
4. add fc layer
5. augmentation
6. lr scheduler
