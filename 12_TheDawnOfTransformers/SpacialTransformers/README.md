# Spatial Transformer 

The spatial transformer module consists of layers of neural networks that can spatially transform an image. These spatial transformations include cropping, scaling, rotations, and translations etc

CNNs perform poorly when the input data contains so much variation. One of the solutions to this is the max-pooling layer. But then again, max-pooling layers do no make the CNN invariant to large transformations in the input data.

This gives rise to the concept of Spatial Transformer Networks. In STNs, the transformer module knows where to apply the transformation to properly scale, resize, and crop and image. We can apply the STN module to the input data directly, or even to the feature maps (output of a convolution layer). In simple words, we can say that the spatial transformer module acts as an attention mechanism and knows where to focus on the input data.


## Architecture

The architecture of a Spatial Transformer Network is based on three important parts.

- The localization network.
- Parameterized sampling grid.
- Differentiable image sampling.

![image](https://user-images.githubusercontent.com/42609155/127073287-08c80ce8-9686-4bdc-9933-cc6801f0f3cb.png)

### Localisation Network

The localization network takes the input feature map and outputs the parameters of the spatial transformations that should be applied to the feature map. The localization network is a very simple stacking of convolutional layers.

In the above figuare, U is the feature map input to the localization network. It outputs θ which are the transformation parameters that are regressed from the localization network. The final regression layers are fully-connected linear layers. Tθ is the transformation operation using the parameters θ.

###  Parameterized Sampling Grid

Parameterized Sampling Grid mainly generates a sampling grid that is consistent with the picture pixels, and multiplies it with theta matrix to gradually learn to fully correspond to the tilt recognition object

### Differentiable image sampling.

Differentable Image Sampling is mainly used to obtain the original image pixels corresponding to the sampling points to form a V feature map to complete the output of the V feature map

## Links to Code

Github Link - https://github.com/gkdivya/EVA/blob/main/12_TheDawnOfTransformers/SpacialTransformers/SpacialTransformers_CIFAR10.ipynb <br>
Colab Link - https://colab.research.google.com/github/gkdivya/EVA/blob/main/12_TheDawnOfTransformers/SpacialTransformers/SpacialTransformers_CIFAR10.ipynb

## Model Architecture

    Net(
      (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
      (conv2_drop): Dropout2d(p=0.5, inplace=False)
      (fc1): Linear(in_features=800, out_features=1024, bias=True)
      (fc2): Linear(in_features=1024, out_features=10, bias=True)
      (localization): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
        (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): ReLU(inplace=True)
      )
      (fc_loc): Sequential(
        (0): Linear(in_features=2048, out_features=256, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=256, out_features=6, bias=True)
      )
    )

## Training and Validation Log

    Train Epoch: 1 [0/50000 (0%)]	Loss: 2.323545
    Train Epoch: 1 [32000/50000 (64%)]	Loss: 2.059350

    Test set: Average loss: 1.8503, Accuracy: 3455/10000 (35%)

    Train Epoch: 2 [0/50000 (0%)]	Loss: 1.831888
    Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.886865

    Test set: Average loss: 1.6227, Accuracy: 4204/10000 (42%)

    Train Epoch: 3 [0/50000 (0%)]	Loss: 1.694754
    Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.606473

    Test set: Average loss: 1.5204, Accuracy: 4549/10000 (45%)

    Train Epoch: 4 [0/50000 (0%)]	Loss: 1.796990
    Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.633417

    Test set: Average loss: 1.4311, Accuracy: 4865/10000 (49%)

    Train Epoch: 5 [0/50000 (0%)]	Loss: 1.567344
    Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.531219

    Test set: Average loss: 1.3916, Accuracy: 4944/10000 (49%)

    Train Epoch: 6 [0/50000 (0%)]	Loss: 1.736797
    Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.657320

    Test set: Average loss: 1.3568, Accuracy: 5155/10000 (52%)

    Train Epoch: 7 [0/50000 (0%)]	Loss: 1.481957
    Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.382086

    Test set: Average loss: 1.3456, Accuracy: 5275/10000 (53%)

    Train Epoch: 8 [0/50000 (0%)]	Loss: 1.517952
    Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.226434

    Test set: Average loss: 1.3063, Accuracy: 5386/10000 (54%)

    Train Epoch: 9 [0/50000 (0%)]	Loss: 1.151374
    Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.206437

    Test set: Average loss: 1.2380, Accuracy: 5709/10000 (57%)

    Train Epoch: 10 [0/50000 (0%)]	Loss: 1.456446
    Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.354574

    Test set: Average loss: 1.4246, Accuracy: 5008/10000 (50%)

    Train Epoch: 11 [0/50000 (0%)]	Loss: 1.362190
    Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.169174

    Test set: Average loss: 1.1937, Accuracy: 5775/10000 (58%)

    Train Epoch: 12 [0/50000 (0%)]	Loss: 1.185801
    Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.029347

    Test set: Average loss: 1.2152, Accuracy: 5824/10000 (58%)

    Train Epoch: 13 [0/50000 (0%)]	Loss: 1.386696
    Train Epoch: 13 [32000/50000 (64%)]	Loss: 0.941119

    Test set: Average loss: 1.1461, Accuracy: 6003/10000 (60%)

    Train Epoch: 14 [0/50000 (0%)]	Loss: 1.049565
    Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.195351

    Test set: Average loss: 1.1265, Accuracy: 6176/10000 (62%)

    Train Epoch: 15 [0/50000 (0%)]	Loss: 1.247364
    Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.086525

    Test set: Average loss: 1.1538, Accuracy: 6003/10000 (60%)

    Train Epoch: 16 [0/50000 (0%)]	Loss: 1.444609
    Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.094858

    Test set: Average loss: 1.1510, Accuracy: 6055/10000 (61%)

    Train Epoch: 17 [0/50000 (0%)]	Loss: 1.088625
    Train Epoch: 17 [32000/50000 (64%)]	Loss: 0.968816

    Test set: Average loss: 1.0879, Accuracy: 6215/10000 (62%)

    Train Epoch: 18 [0/50000 (0%)]	Loss: 1.174545
    Train Epoch: 18 [32000/50000 (64%)]	Loss: 1.038991

    Test set: Average loss: 1.1363, Accuracy: 6135/10000 (61%)

    Train Epoch: 19 [0/50000 (0%)]	Loss: 1.330752
    Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.006819

    Test set: Average loss: 1.0857, Accuracy: 6275/10000 (63%)

    Train Epoch: 20 [0/50000 (0%)]	Loss: 1.163814
    Train Epoch: 20 [32000/50000 (64%)]	Loss: 1.170664

    Test set: Average loss: 1.1007, Accuracy: 6245/10000 (62%)

    Train Epoch: 21 [0/50000 (0%)]	Loss: 1.148362
    Train Epoch: 21 [32000/50000 (64%)]	Loss: 0.952622

    Test set: Average loss: 1.0823, Accuracy: 6238/10000 (62%)

    Train Epoch: 22 [0/50000 (0%)]	Loss: 1.132950
    Train Epoch: 22 [32000/50000 (64%)]	Loss: 1.108818

    Test set: Average loss: 1.0709, Accuracy: 6301/10000 (63%)

    Train Epoch: 23 [0/50000 (0%)]	Loss: 1.033309
    Train Epoch: 23 [32000/50000 (64%)]	Loss: 0.971754

    Test set: Average loss: 1.0409, Accuracy: 6449/10000 (64%)

    Train Epoch: 24 [0/50000 (0%)]	Loss: 1.043417
    Train Epoch: 24 [32000/50000 (64%)]	Loss: 1.073401

    Test set: Average loss: 1.0763, Accuracy: 6288/10000 (63%)

    Train Epoch: 25 [0/50000 (0%)]	Loss: 1.069654
    Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.232488

    Test set: Average loss: 1.0320, Accuracy: 6456/10000 (65%)

    Train Epoch: 26 [0/50000 (0%)]	Loss: 0.922884
    Train Epoch: 26 [32000/50000 (64%)]	Loss: 0.735497

    Test set: Average loss: 1.0782, Accuracy: 6271/10000 (63%)

    Train Epoch: 27 [0/50000 (0%)]	Loss: 1.016425
    Train Epoch: 27 [32000/50000 (64%)]	Loss: 0.827539

    Test set: Average loss: 1.1679, Accuracy: 6080/10000 (61%)

    Train Epoch: 28 [0/50000 (0%)]	Loss: 1.426343
    Train Epoch: 28 [32000/50000 (64%)]	Loss: 0.801315

    Test set: Average loss: 1.0189, Accuracy: 6519/10000 (65%)

    Train Epoch: 29 [0/50000 (0%)]	Loss: 0.952791
    Train Epoch: 29 [32000/50000 (64%)]	Loss: 0.965643

    Test set: Average loss: 0.9905, Accuracy: 6606/10000 (66%)

    Train Epoch: 30 [0/50000 (0%)]	Loss: 0.841032
    Train Epoch: 30 [32000/50000 (64%)]	Loss: 0.719915

    Test set: Average loss: 0.9853, Accuracy: 6630/10000 (66%)

    Train Epoch: 31 [0/50000 (0%)]	Loss: 0.857542
    Train Epoch: 31 [32000/50000 (64%)]	Loss: 1.049039

    Test set: Average loss: 0.9998, Accuracy: 6621/10000 (66%)

    Train Epoch: 32 [0/50000 (0%)]	Loss: 0.792736
    Train Epoch: 32 [32000/50000 (64%)]	Loss: 0.591469

    Test set: Average loss: 0.9940, Accuracy: 6653/10000 (67%)

    Train Epoch: 33 [0/50000 (0%)]	Loss: 0.692739
    Train Epoch: 33 [32000/50000 (64%)]	Loss: 0.686782

    Test set: Average loss: 1.0606, Accuracy: 6390/10000 (64%)

    Train Epoch: 34 [0/50000 (0%)]	Loss: 0.914352
    Train Epoch: 34 [32000/50000 (64%)]	Loss: 0.952755

    Test set: Average loss: 1.0841, Accuracy: 6254/10000 (63%)

    Train Epoch: 35 [0/50000 (0%)]	Loss: 1.229171
    Train Epoch: 35 [32000/50000 (64%)]	Loss: 0.891603

    Test set: Average loss: 1.0064, Accuracy: 6546/10000 (65%)

    Train Epoch: 36 [0/50000 (0%)]	Loss: 0.878139
    Train Epoch: 36 [32000/50000 (64%)]	Loss: 0.858399

    Test set: Average loss: 1.0306, Accuracy: 6459/10000 (65%)

    Train Epoch: 37 [0/50000 (0%)]	Loss: 1.085227
    Train Epoch: 37 [32000/50000 (64%)]	Loss: 0.924159

    Test set: Average loss: 1.0020, Accuracy: 6544/10000 (65%)

    Train Epoch: 38 [0/50000 (0%)]	Loss: 0.782105
    Train Epoch: 38 [32000/50000 (64%)]	Loss: 1.012582

    Test set: Average loss: 1.0327, Accuracy: 6499/10000 (65%)

    Train Epoch: 39 [0/50000 (0%)]	Loss: 0.905093
    Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.802818

    Test set: Average loss: 1.0074, Accuracy: 6524/10000 (65%)

    Train Epoch: 40 [0/50000 (0%)]	Loss: 0.965823
    Train Epoch: 40 [32000/50000 (64%)]	Loss: 0.845076

    Test set: Average loss: 0.9725, Accuracy: 6672/10000 (67%)

    Train Epoch: 41 [0/50000 (0%)]	Loss: 0.936192
    Train Epoch: 41 [32000/50000 (64%)]	Loss: 1.193512

    Test set: Average loss: 1.1972, Accuracy: 5926/10000 (59%)

    Train Epoch: 42 [0/50000 (0%)]	Loss: 1.121302
    Train Epoch: 42 [32000/50000 (64%)]	Loss: 1.015068

    Test set: Average loss: 0.9916, Accuracy: 6577/10000 (66%)

    Train Epoch: 43 [0/50000 (0%)]	Loss: 0.819756
    Train Epoch: 43 [32000/50000 (64%)]	Loss: 0.756327

    Test set: Average loss: 1.1255, Accuracy: 6124/10000 (61%)

    Train Epoch: 44 [0/50000 (0%)]	Loss: 0.998561
    Train Epoch: 44 [32000/50000 (64%)]	Loss: 0.771200

    Test set: Average loss: 1.0567, Accuracy: 6476/10000 (65%)

    Train Epoch: 45 [0/50000 (0%)]	Loss: 0.701021
    Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.824684

    Test set: Average loss: 0.9893, Accuracy: 6679/10000 (67%)

    Train Epoch: 46 [0/50000 (0%)]	Loss: 0.465935
    Train Epoch: 46 [32000/50000 (64%)]	Loss: 0.806973

    Test set: Average loss: 1.0120, Accuracy: 6638/10000 (66%)

    Train Epoch: 47 [0/50000 (0%)]	Loss: 0.661738
    Train Epoch: 47 [32000/50000 (64%)]	Loss: 0.797799

    Test set: Average loss: 1.0350, Accuracy: 6502/10000 (65%)

    Train Epoch: 48 [0/50000 (0%)]	Loss: 0.636714
    Train Epoch: 48 [32000/50000 (64%)]	Loss: 0.671531

    Test set: Average loss: 1.0672, Accuracy: 6411/10000 (64%)

    Train Epoch: 49 [0/50000 (0%)]	Loss: 0.863062
    Train Epoch: 49 [32000/50000 (64%)]	Loss: 0.735707

    Test set: Average loss: 1.0700, Accuracy: 6436/10000 (64%)

    Train Epoch: 50 [0/50000 (0%)]	Loss: 0.748093
    Train Epoch: 50 [32000/50000 (64%)]	Loss: 0.642411

    Test set: Average loss: 0.9894, Accuracy: 6711/10000 (67%)


## Visualize STN Results

![image](https://user-images.githubusercontent.com/42609155/127077962-9743cb5b-eebd-4f10-8276-3cec0caa75d9.png)


Spatial Transformer Network model has cropped and resized most of the images to the center. It has rotated many of the images to an orientation that it feels will be helpful. Although some of the orientations are not centered. Maybe a bit of more training will help.


## Reference

https://arxiv.org/pdf/1506.02025v3.pdf <br>
https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html
https://kevinzakka.github.io/2017/01/10/stn-part1/
https://kevinzakka.github.io/2017/01/18/stn-part2/
https://medium.com/@kushagrabh13/spatial-transformer-networks-ebc3cc1da52d
