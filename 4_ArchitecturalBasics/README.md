
# Architectural Basics

Modified the code given in [MNIST Base Code](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx) to achieve 99.4% validation accuracy with *10k Parameters* and in less than 20 Epochs.

Below important concepts were used/considered while designing the network:
- All convolution Blocks were designed with 3 * 3 kernels, Batch Normalization and Dropout with ReLU activation. 
- Receptive Field calculated for each blocks and it was the main intuition behind the number of layers added in the network. Convolutions are performed to achieve at least the receptive field equal to the image size.
- Transition Block - Max pooling with 1 * 1 convolution is used to reduce the number of channels in turn to reduce the number of parameters used in the network
- GAP, followed by Fully connected layer used just before prediction to give the network a little flexibility with the input image size.
- And most important thing, we have ensured not to use Batch normalization, Activation function, Max pooling, dropout just before the last layer. 

Intuition behind the step by step approach we followed to reduce the number of parameters and to improve accuracy with less params:
- At first, we reduced the number of parameters using Convolution Blocks with less number of output channels (removed 64, 128, 256 and 512 for every layer) and removed bias
- Added a Transition Block (max pooling followed by 1x1).
- Added a GAP layer to convert 2d to 1d
- Added a FC layer after GAP i.e used 1x1 after GAP (Note: 1x1 is fully connected layer when applied on 1d data)
- We used Batch Normalization after every convolution layer except the last one
- Used Augmentation technique like image rotation
- Added Dropout after every layer, we added dropout after we tried out all possible options. Adding drop out, helped reduced the gap between training and test loss.
- We used OneCycleLR Learning Rate to tune the model
- Had to increase the number of channels to achieve 99.4 accuracy in 19 epochs
- Network was trained for 19 epochs with batch size of 128
- Achieved a test accuracy of 99.4% at 18th and 19th epoch

## Experiments

Batch Size = 128 <br>
Epochs = 19

|Experiment| #Parameters | Batch Normalization | Augmentation | Dropout | GAP & FC Layer | Learning Rate Scheduler | Validation Accuracy | 
|-------|---|---|---|---|---|---|---|
|[MNIST_With Less Params](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/Experiments/MNIST_Exp1_WithLessParams.ipynb) |5490|No|No|No|No|0.01|99%|
|[MNIST_With Transition Block & GAP](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/Experiments/MNIST_Exp2_WithTransitionBlock.ipynb) |5690|Yes|No|No|No|0.02|98.82%|
|[MNIST_With Batch Normalization](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/Experiments/MNIST_Exp3_WithBatchNormalization.ipynb) |5810|Yes|Yes|No|No|0.02| 99.11%|
|[MNIST_With DropOut_LRScheduler](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/Experiments/MNIST_Exp6_WithLRScheduler.ipynb)|5184|Yes|Yes|0.1|Yes|0.02| 99.41% at 18th Epoch|
|[MNIST_With_10k_Parameters](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/Experiments/MNIST_Exp9_With10kParams.ipynb)|10040|Yes|Yes|0.1|Yes|0.015|99.40% at 18th and 19th Epoch|

## Final Best Model

[GitHub Notebook Link](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/MNIST_Architecture_Basics.ipynb) <br>
[Colab Link](https://colab.research.google.com/github/gkdivya/EVA/blob/main/4_ArchitecturalBasics/MNIST_Architecture_Basics.ipynb)

### Model Architecture

      Net(
        (conv1): Sequential(
          (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): Dropout2d(p=0.1, inplace=False)
          (4): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
          (5): ReLU()
          (6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (7): Dropout2d(p=0.1, inplace=False)
          (8): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
          (9): ReLU()
          (10): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (11): Dropout2d(p=0.1, inplace=False)
        )
        (trans1): Sequential(
          (0): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        (conv2): Sequential(
          (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): Dropout2d(p=0.1, inplace=False)
          (4): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
          (5): ReLU()
          (6): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (7): Dropout2d(p=0.1, inplace=False)
          (8): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False)
          (9): ReLU()
          (10): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (11): Dropout2d(p=0.1, inplace=False)
        )
        (avg_pool): Sequential(
          (0): AvgPool2d(kernel_size=5, stride=1, padding=0)
        )
        (conv_4): Sequential(
          (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): ReLU()
          (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): Dropout2d(p=0.1, inplace=False)
          (4): Conv2d(16, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
      )


### Training Log

    epoch=1 Loss=0.5491642355918884 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.96it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.3586, Accuracy: 9277/10000 (92.77%)

    epoch=2 Loss=0.1878300905227661 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.13it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0970, Accuracy: 9723/10000 (97.23%)

    epoch=3 Loss=0.13212096691131592 batch_id=00468: 100%|██████████| 469/469 [00:15<00:00, 29.46it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0549, Accuracy: 9827/10000 (98.27%)

    epoch=4 Loss=0.07660138607025146 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.83it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0490, Accuracy: 9849/10000 (98.49%)

    epoch=5 Loss=0.10429932922124863 batch_id=00468: 100%|██████████| 469/469 [00:15<00:00, 29.37it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0383, Accuracy: 9875/10000 (98.75%)

    epoch=6 Loss=0.036357346922159195 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.06it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0321, Accuracy: 9896/10000 (98.96%)

    epoch=7 Loss=0.08324731141328812 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.52it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0300, Accuracy: 9911/10000 (99.11%)

    epoch=8 Loss=0.2502351701259613 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.04it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0279, Accuracy: 9917/10000 (99.17%)

    epoch=9 Loss=0.06680113822221756 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.13it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0252, Accuracy: 9927/10000 (99.27%)

    epoch=10 Loss=0.16524271667003632 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.05it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0272, Accuracy: 9923/10000 (99.23%)

    epoch=11 Loss=0.06911925971508026 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.25it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0257, Accuracy: 9922/10000 (99.22%)

    epoch=12 Loss=0.06363837420940399 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.79it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0234, Accuracy: 9933/10000 (99.33%)

    epoch=13 Loss=0.054308656603097916 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.67it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0226, Accuracy: 9932/10000 (99.32%)

    epoch=14 Loss=0.16232791543006897 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.68it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0215, Accuracy: 9933/10000 (99.33%)

    epoch=15 Loss=0.12649112939834595 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.08it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0213, Accuracy: 9935/10000 (99.35%)

    epoch=16 Loss=0.03541385754942894 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 29.27it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0203, Accuracy: 9941/10000 (99.41%)

    epoch=17 Loss=0.06020794436335564 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.59it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.38%)

    epoch=18 Loss=0.039867233484983444 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.62it/s]
      0%|          | 0/469 [00:00<?, ?it/s]Test set: Average loss: 0.0201, Accuracy: 9940/10000 (99.40%)

    epoch=19 Loss=0.08172362297773361 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.81it/s]
    Test set: Average loss: 0.0201, Accuracy: 9940/10000 (99.40%)

### Validation Accuracy

Achieved a test accuracy of 99.4% at 18th and 19th epoch

![accuracy_plot](https://user-images.githubusercontent.com/42609155/119974537-68f73180-bfd2-11eb-98d3-89db764d5959.png)


## Reference Links
[Kaggle Notebook]( https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99)

## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
