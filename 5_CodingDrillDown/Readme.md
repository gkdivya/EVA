# Finetune CNN Architecture on MNIST dataset

Objective is to fine tune the base code in [MNIST Basic Code](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%200_BasicSetup.ipynb) to achieve **99.4%** validation accuracy with **less than 10k Parameters** in 15 Epochs. 

## How we did:

Inspired by set of TSAI - motivational posts, we picked up all the right highlighted secrets in it to achieve the mammoth task! 
![image1](https://user-images.githubusercontent.com/17870236/120790982-3bb9fe80-c551-11eb-9b42-2ee1b2ca05a3.png)

1. **Skeleton** - Reduced the number of parameters using Convolution Blocks with less number of output channels (removed 64, 128, 256 and 512 for every layer) and removed bias
2. **Max-Pooling** - Transition Block (max pooling followed by 1x1) after 5x5 receptive field added in network.
3. **Batch-Normalization** - Added after every convolution layer except the last one to normalize the values being passed between convolution layers
4. **Capacity** - With very less paramters ~4k parameters, even with all the right concepts in place model couldnt learn. Increased capacity a bit to increase the accuracy.
5. **Global Average pooling** - GAP followed by fully connected layer (1x1 is applied on 1d data) used just before prediction to give the network a little flexibility with the input image size.
6. **Augmentation** - Image augmentation technique like image rotation, color jitter and affine transformation are used
7. **Regularization** - Adding drop out, helped reduced the gap between training and test loss.
8. **Learning Rate** - Used OneCycleLR Learning Rate to tune the model

More details on these experiments can be found [here](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/README.md).

| Experiment                                                                                                                        | Target                                                                                                                                                                                                          | Parameters | BestTrain Accuracy | Best Test Accuracy | Analysis                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MNIST\_Basic\_Setup](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%200_BasicSetup.ipynb)        | • Get the set-up right<br>• Set Transforms<br>• Set Data Loader<br>• Set Basic Working Code<br>• Set Basic Training  & Test Loop                                                                                | 6.3M       | 99.93             | 99.28         | •  Extremely Heavy Model for such a problem<br>•  Model is over-fitting because the training accuracy is 99.93, but we are changing our model in the next step                                                                                                                                                                                                                                                                                                              |
| [MNIST\_Base Skeleton Model](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%201_BasicSkeleton.ipynb)                                          |  • Get the basic skeleton interms of convolution and placement of transition blocks (max pooling, 1x1's)<br>•  Reduce the number of parameters as low as possible<br>•  Add GAP and remove the last BIG kernel. | 4572       | 98.22             | 98.43         | •  We have structured our model in a readable way<br>•  The model is lighter with less number of parameters<br>•  The performace is reduced compared to previous models. Since we have reduced model capacity, this is expected, the model has capability to learn.<br>• Next, we will be tweaking this model further and increase the capacity to push it more towards the desired accuracy.                                                                               |
| [MNIST\_With\_Batch Normalization](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%202_Batch_Normalization.ipynb)                                    | •  Add Batch-norm to increase model efficiency.                                                                                                                                                                 | 5088       | 99.03             | 99.04         | •  There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer<br> • Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.                                                                                                                                                                                                                         |
| [MNIST\_With Dropout](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%203_Dropout.ipynb)                                                 | <br>•  Add Regularization Dropout to each layer except last layer                                                                                                                                               | 5088       | 97.94             | 98.64         | •  There is no overfitting at all. With dropout training will be harder, because we are droping the pixels randomly.<br>•  The performance has droppped, we can further improve it.<br>•  But with the current capacity,not possible to push it further.We can possibly increase the capacity of the model by adding a layer after GAP!                                                                                                                                     |
| [MNIST\_With\_FullyConnectedLayer](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%204_%20Fully%20Connected%20layer.ipynb)                                    | • Increase model capacity at the end (add layer after GAP)                                                                                                                                                      | 6124       | 99.07             | 99.22         | • The model parameters have increased<br>• There is no overfitting rather slight underfitting, thats fine dropout is doing its work , because we are adding dropout at each layer the model is able to capture the training accuracy<br>• However, we haven't reached 99.4 accuracy yet.<br>Observing the missclassified images its good to try out some augmentation techniques as few images seems to be slightly rotated, and also image contrast needs to be considered |
| [MNIST\_With\_Augmentation](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%205_Augmentation.ipynb)                                           | •Add various Image augmentation techniques, image rotation, randomaffine, colorjitter .                                                                                                                         | 6124       | 97.61             | 99.32         | •The model is under-fitting, that should be ok as we know we have made our train data harder.<br>• However, we haven't reached 99.4 accuracy yet.<br>•The model seems to be stuck at 99.2% accuracy, seems like the model needs some additional capacity towards the end.                                                                                                                                                                                                   |
| [MNIST\_With\_LR Scheduler](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%206_LRScheduler.ipynb) | • Add some capacity (additional FC layer after GAP) to the model and added LR Scheduler                                                                                                                         | 6720       | 99.43             | 99.53         | •The model parameters have increased<br>• The model is under-fitting. This is fine, as we know we have made our train data harder.<br>• LR Scheduler and the additional capacity after GAP helped getting to the desired target 99.4, Onecyclic LR is being used, this seemed to perform better than StepLR to achieve consistent accuracy in last few layers

# Final Model

By fine tuning the model with the step by step approach, the [final model](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/MNIST_BestModel.ipynb) was able to reach best test accuracy of **99.53%** in **15 epochs** with just **6720 (6K parameters)**!!!

![image](https://user-images.githubusercontent.com/17870236/120801028-46c75b80-c55e-11eb-9797-07532b18be0f.png)

## Receptive Field calculation:

Formula reference:</br>
<p align="center"><img src="https://user-images.githubusercontent.com/17870236/120273908-c0481b00-c2cc-11eb-8b97-af4c8b9d5917.png" width=1000></p>

| Operation   | nin | in\_ch | out\_ch | padding | kernel | stride | nout | jin | jout | rin | rout | Act-size | Params |
| ----------- | --- | ------ | ------- | ------- | ------ | ------ | ---- | --- | ---- | --- | ---- | -------- | ------ |
| Convolution | 28  | 1      | 8       | 0       | 3      | 1      | 26   | 1   | 1    | 1   | 3    | 784      | 72     |
| Convolution | 26  | 8      | 16      | 0       | 3      | 1      | 24   | 1   | 1    | 3   | 5    | 5408     | 1152   |
| Max-Pooling | 24  | 16     | 16      | 0       | 2      | 2      | 12   | 1   | 2    | 5   | 6    | 9216     | 0      |
| Convolution | 12  | 16     | 8       | 0       | 1      | 1      | 12   | 2   | 2    | 6   | 6    | 2304     | 128    |
| Convolution | 12  | 8      | 10      | 0       | 3      | 1      | 10   | 2   | 2    | 6   | 10   | 1152     | 720    |
| Convolution | 10  | 10     | 16      | 0       | 3      | 1      | 8    | 2   | 2    | 10  | 14   | 1000     | 1440   |
| Convolution | 8   | 16     | 18      | 0       | 3      | 1      | 6    | 2   | 2    | 14  | 18   | 1024     | 2592   |
| Convolution | 6   | 18     | 16      | 0       | 1      | 1      | 6    | 2   | 2    | 18  | 18   | 648      | 288    |
| Convolution | 6   | 16     | 10      | 0       | 1      | 1      | 6    | 2   | 2    | 18  | 18   | 576      | 160    |

## Training Log:

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

## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
