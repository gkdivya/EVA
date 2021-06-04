# Finetune CNN Architecture on MNIST dataset

## Objective:
- Achieve 99.4% accuracy on test data (this must be consistently shown in last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 10000 or 8000 Parameters 
- Do this in exactly 3 or more steps
- Each File must have "target, result, analysis" TEXT block (either at the start or the end)
- You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 


Modified MNIST model architecture to have 6k params and trained for 15 epochs<br>
Achieved 99.4% accuracy on MNIST test data consistently in last few epochs<br>

| Experiment                                                                                                                        | Target                                                                                                                           | Parameters |BestTrain Accuracy | Best Test Accuracy | Analysis                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MNIST\_Basic\_Setup](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step1_BasicSetup.ipynb)        | • Get the set-up right<br>• Set Transforms<br>• Set Data Loader<br>• Set Basic Working Code<br>• Set Basic Training  & Test Loop | 6.3M       | 99.93             | 99.28         | •  Extremely Heavy Model for such a problem<br>•  Model is over-fitting because the training accuracy is 99.93, but we are changing our model in the next step                                                                                                                                      |
| [MNIST\_Base Skeleton Model](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown)                                          |                                                                                                                                  |            |                   |               |                                                                                                                                                                                                                                                                                                     |
| [MNIST\_With\_Batch Normalization](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown)                                    |                                                                                                                                  |            |                   |               |                                                                                                                                                                                                                                                                                                     |
| [MNIST\_With Dropout](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown)                                                 |                                                                                                                                  |            |                   |               |                                                                                                                                                                                                                                                                                                     |
| [MNIST\_With\_Augmentation](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown) | •Add various Image augmentation techniques, image rotation, randomaffine, colorjitter .  |      6124      |           97.61        |   99.32            |   •The model is under-fitting, that should be ok as we know we have made our train data harder. <br> • However, we haven't reached 99.4 accuracy yet.<br> •The model seems to be stuck at 99.2% accuracy, seems like the model needs some additional capacity towards the end.                                                                                                                                                                                                                                                                                                |
| [MNIST\_With\_LR Scheduler](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step7_LRScheduler.ipynb) | • Add some capacity (additional FC layer after GAP) to the model and added LR Scheduler                                                                           | 6720       | 99.43             | 99.53         | •The model parameters have increased<br>• The model is under-fitting. This is fine, as we know we have made our train data harder. <br> • LR Scheduler and the additional capacity after GAP helped getting to the desired target 99.4, Onecyclic LR is being used, this seemed to perform better than StepLR to achieve consistent accuracy in last few layerss |

## Receptive Field calculation for our final model

Formula reference:</br>
<p align="center"><img src="https://user-images.githubusercontent.com/17870236/120273908-c0481b00-c2cc-11eb-8b97-af4c8b9d5917.png"></p>


| Operation   | nin | Ch_in | Ch_Out | padding | kernel | stride | nout | jin | jout | rin | rout | Act_Size | Params# |
| ----------- | --- | ----- | ------ | ------- | ------ | ------ | ---- | --- | ---- | --- | ---- | -------- | ------- |
| Convolution | 28  | 1     | 8      | 0       | 3      | 1      | 26   | 1   | 1    | 1   | 3    | 784      | 72      |
| Convolution | 26  | 8     | 16     | 0       | 3      | 1      | 24   | 1   | 1    | 3   | 5    | 5408     | 1152    |
| Max-Pooling | 24  | 16    | 16     | 0       | 2      | 2      | 12   | 1   | 2    | 5   | 6    | 9216     | 0       |
| Convolution | 12  | 16    | 8      | 0       | 1      | 1      | 12   | 2   | 2    | 6   | 12   | 2304     | 128     |
| Convolution | 12  | 8     | 10     | 0       | 3      | 1      | 10   | 2   | 2    | 12  | 28   | 1152     | 720     |
| Convolution | 10  | 10    | 16     | 0       | 3      | 1      | 8    | 2   | 2    | 28  | 60   | 1000     | 1440    |
| Convolution | 8   | 16    | 16     | 0       | 3      | 1      | 6    | 2   | 2    | 60  | 124  | 1024     | 2304    |
| Convolution | 6   | 16    | 16     | 1       | 3      | 1      | 6    | 2   | 2    | 124 | 252  | 576      | 2304    |

## Training Log for final Model

      EPOCH: 1
      Loss=0.32070186734199524 Batch_id=468 Accuracy=66.93: 100%|██████████| 469/469 [00:53<00:00,  8.85it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.1617, Accuracy: 9634/10000 (96.34%)

      EPOCH: 2
      Loss=0.23211365938186646 Batch_id=468 Accuracy=94.22: 100%|██████████| 469/469 [00:53<00:00,  8.75it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0673, Accuracy: 9802/10000 (98.02%)

      EPOCH: 3
      Loss=0.22090421617031097 Batch_id=468 Accuracy=95.92: 100%|██████████| 469/469 [00:53<00:00,  8.69it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0455, Accuracy: 9844/10000 (98.44%)

      EPOCH: 4
      Loss=0.05350199341773987 Batch_id=468 Accuracy=96.72: 100%|██████████| 469/469 [00:53<00:00,  8.73it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0350, Accuracy: 9895/10000 (98.95%)

      EPOCH: 5
      Loss=0.05736066773533821 Batch_id=468 Accuracy=97.06: 100%|██████████| 469/469 [00:53<00:00,  8.72it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0297, Accuracy: 9906/10000 (99.06%)

      EPOCH: 6
      Loss=0.056373003870248795 Batch_id=468 Accuracy=97.32: 100%|██████████| 469/469 [00:53<00:00,  8.72it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0252, Accuracy: 9924/10000 (99.24%)

      EPOCH: 7
      Loss=0.11534460633993149 Batch_id=468 Accuracy=97.50: 100%|██████████| 469/469 [00:53<00:00,  8.70it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0246, Accuracy: 9923/10000 (99.23%)

      EPOCH: 8
      Loss=0.04017015919089317 Batch_id=468 Accuracy=97.66: 100%|██████████| 469/469 [00:54<00:00,  8.65it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0219, Accuracy: 9936/10000 (99.36%)

      EPOCH: 9
      Loss=0.018773594871163368 Batch_id=468 Accuracy=97.77: 100%|██████████| 469/469 [00:54<00:00,  8.64it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0219, Accuracy: 9929/10000 (99.29%)

      EPOCH: 10
      Loss=0.05798463150858879 Batch_id=468 Accuracy=97.95: 100%|██████████| 469/469 [00:53<00:00,  8.73it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0169, Accuracy: 9953/10000 (99.53%)

      EPOCH: 11
      Loss=0.020612243562936783 Batch_id=468 Accuracy=98.03: 100%|██████████| 469/469 [00:54<00:00,  8.67it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0175, Accuracy: 9949/10000 (99.49%)

      EPOCH: 12
      Loss=0.02381170354783535 Batch_id=468 Accuracy=98.18: 100%|██████████| 469/469 [00:53<00:00,  8.71it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0195, Accuracy: 9946/10000 (99.46%)

      EPOCH: 13
      Loss=0.10838382691144943 Batch_id=468 Accuracy=98.31: 100%|██████████| 469/469 [00:53<00:00,  8.72it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0164, Accuracy: 9953/10000 (99.53%)

      EPOCH: 14
      Loss=0.1120440661907196 Batch_id=468 Accuracy=98.29: 100%|██████████| 469/469 [00:53<00:00,  8.72it/s]
        0%|          | 0/469 [00:00<?, ?it/s]
      Test set: Average loss: 0.0170, Accuracy: 9945/10000 (99.45%)

      EPOCH: 15
      Loss=0.08451732248067856 Batch_id=468 Accuracy=98.43: 100%|██████████| 469/469 [00:53<00:00,  8.72it/s]
      Test set: Average loss: 0.0168, Accuracy: 9950/10000 (99.50%)


<p align="center"><img src="https://user-images.githubusercontent.com/42609155/120590215-d0d8cc80-c457-11eb-877e-2c904aa27f8f.png" width="800"></p>

<b>Finally, by fine tuning the model in a step by step approach, the model was able to reach best test accuracy of 99.53% in 15 epochs with just 6720 (6K parameters)!!!

## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
