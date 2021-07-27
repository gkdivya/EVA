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

### Differentiable image sampling.


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


## Visualize STN Results



The spatial transformations here are very prominent. Our Spatial Transformer Network model has cropped and resized most of the images to the center. It has rotated many of the images to an orientation that it feels will be helpful. Although some of the orientations are not centered. Maybe a bit of more training will help.


## Reference

https://arxiv.org/pdf/1506.02025v3.pdf <br>
https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html
