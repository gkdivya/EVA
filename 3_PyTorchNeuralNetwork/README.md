### Problem Statement:

Write a neural network that can take 2 inputs:
* an image from MNIST dataset and
* a random number between 0 and 9

and gives two outputs:
* the "number" that was represented by the MNIST image, and
* the "sum" of this number with the random number that was generated and sent as the input to the network

<p align="center"><img src="https://user-images.githubusercontent.com/42609155/118740404-6aee2180-b869-11eb-9a42-d72efbc4f132.png" width="600"></p>

Network can be written with fully connected layers and convolution layers. 


## Data Preparation:

The MNIST dataset is used as one of the input to the model, it contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels

Custom dataset is created using MNIST dataset and generates data in below format: 

|Mnist Image|Mnist Label|One hot encoded Random Number|Sum of random number and Mnist Number| 
|----|-----|------|-----|

A random integer, sum of MNIST label and the random integer along with MNIST Image and its label. 

## Model

<p align="center"><img src="https://github.com/gkdivya/EVA/blob/7b9feda284e2b2eb7342e1652f7efb5e95206e09/3_PyTorchNeuralNetwork/assets/MNIST_RandomAddition.png" width="800"></p>

* Using convolution blocks, MNIST image features are extracted 
* One hot encoded random number is concatenated with the MNIST image features, are further passed to fully connected layers to predict the sum
* MNIST features are flatten and passed to a softmax function directly to predict the MNIST number

      Net(
        (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (conv5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
        (conv6): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1))
        (conv7): Conv2d(1024, 10, kernel_size=(3, 3), stride=(1, 1))
        (fc1): Linear(in_features=20, out_features=128, bias=True)
        (fc2): Linear(in_features=128, out_features=19, bias=True)
      )

## Number of parameters
Model has 6,384,925 trainable parameters.


## Loss Function
Because both the MNIST classification and the sum(0-18) are multi class classification problem, we have used Negative Log-Likelihood Loss Function.

In NLL, the model is punished for making the correct prediction with smaller probabilities and encouraged for making the prediction with higher probabilities. The logarithm does the punishment. NLL does not only care about the prediction being correct but also about the model being certain about the prediction with a high score. 

The Pytorch NLL Loss is expressed as: log(x,y) = -log(y)
x represents the actual value and y the predicted value.


## Training Log

      Epoch 1 : 
      Train set: Average loss: 1.3315
      Test set: Average loss: 1.342, MNist Accuracy:94.0, Sum_Accuracy:11.36

      Epoch 2 : 
      Train set: Average loss: 1.1960
      Test set: Average loss: 1.235, MNist Accuracy:96.24, Sum_Accuracy:14.6

      Epoch 3 : 
      Train set: Average loss: 1.1768
      Test set: Average loss: 1.156, MNist Accuracy:97.82, Sum_Accuracy:21.1

      Epoch 4 : 
      Train set: Average loss: 1.1079
      Test set: Average loss: 1.112, MNist Accuracy:98.32, Sum_Accuracy:29.84

      Epoch 5 : 
      Train set: Average loss: 1.0855
      Test set: Average loss: 1.055, MNist Accuracy:98.48, Sum_Accuracy:41.94

      Epoch 6 : 
      Train set: Average loss: 0.9753
      Test set: Average loss: 0.980, MNist Accuracy:99.02, Sum_Accuracy:50.18

      Epoch 7 : 
      Train set: Average loss: 0.9036
      Test set: Average loss: 0.905, MNist Accuracy:98.76, Sum_Accuracy:59.54

      Epoch 8 : 
      Train set: Average loss: 0.8030
      Test set: Average loss: 0.821, MNist Accuracy:99.04, Sum_Accuracy:70.76

      Epoch 9 : 
      Train set: Average loss: 0.7155
      Test set: Average loss: 0.734, MNist Accuracy:98.92, Sum_Accuracy:78.32

      Epoch 10 : 
      Train set: Average loss: 0.6378
      Test set: Average loss: 0.648, MNist Accuracy:99.12, Sum_Accuracy:87.92

      Epoch 11 : 
      Train set: Average loss: 0.5266
      Test set: Average loss: 0.560, MNist Accuracy:99.04, Sum_Accuracy:91.62

      Epoch 12 : 
      Train set: Average loss: 0.4535
      Test set: Average loss: 0.476, MNist Accuracy:99.22, Sum_Accuracy:95.22

      Epoch 13 : 
      Train set: Average loss: 0.4048
      Test set: Average loss: 0.396, MNist Accuracy:99.22, Sum_Accuracy:97.14

      Epoch 14 : 
      Train set: Average loss: 0.3035
      Test set: Average loss: 0.327, MNist Accuracy:99.26, Sum_Accuracy:97.96

      Epoch 15 : 
      Train set: Average loss: 0.2906
      Test set: Average loss: 0.269, MNist Accuracy:99.34, Sum_Accuracy:98.92

      Epoch 16 : 
      Train set: Average loss: 0.2261
      Test set: Average loss: 0.226, MNist Accuracy:99.34, Sum_Accuracy:98.84

      Epoch 17 : 
      Train set: Average loss: 0.1648
      Test set: Average loss: 0.184, MNist Accuracy:99.42, Sum_Accuracy:99.1

      Epoch 18 : 
      Train set: Average loss: 0.1486
      Test set: Average loss: 0.162, MNist Accuracy:99.2, Sum_Accuracy:99.08

      Epoch 19 : 
      Train set: Average loss: 0.1223
      Test set: Average loss: 0.134, MNist Accuracy:99.38, Sum_Accuracy:99.22

      Epoch 20 : 
      Train set: Average loss: 0.0993
      Test set: Average loss: 0.120, MNist Accuracy:99.32, Sum_Accuracy:99.1

      Epoch 21 : 
      Train set: Average loss: 0.0927
      Test set: Average loss: 0.114, MNist Accuracy:99.18, Sum_Accuracy:99.02

      Epoch 22 : 
      Train set: Average loss: 0.0717
      Test set: Average loss: 0.094, MNist Accuracy:99.36, Sum_Accuracy:99.12

      Epoch 23 : 
      Train set: Average loss: 0.0772
      Test set: Average loss: 0.084, MNist Accuracy:99.26, Sum_Accuracy:99.36

      Epoch 24 : 
      Train set: Average loss: 0.0525
      Test set: Average loss: 0.075, MNist Accuracy:99.46, Sum_Accuracy:99.32

      Epoch 25 : 
      Train set: Average loss: 0.0493
      Test set: Average loss: 0.070, MNist Accuracy:99.32, Sum_Accuracy:99.24


<p align="center"><img src="https://user-images.githubusercontent.com/42609155/119095568-10eb8880-ba30-11eb-910d-f766d3a1d237.png" width="500"></p>

## Test Loss

      Test set: Average loss: 0.073, MNist Accuracy:99.3, Sum_Accuracy:99.26
      
## Inference

<p align="center">
    <img src="https://user-images.githubusercontent.com/42609155/119095810-60ca4f80-ba30-11eb-974f-e123a9e02fb3.png" width="200">
    <img src="https://user-images.githubusercontent.com/42609155/119095818-645dd680-ba30-11eb-980d-8fe1618ff0be.png" width="200">
    <img src="https://user-images.githubusercontent.com/42609155/119095826-66279a00-ba30-11eb-8720-495a8134885b.png" width="200">
    <img src="https://user-images.githubusercontent.com/42609155/119095834-6889f400-ba30-11eb-8c4d-e3aec3723442.png" width="200">
</p>


## References

[How to build custom Datasets for Images in Pytorch](https://youtu.be/ZoZHd0Zm3RY)<br>
[concatenate-layer-output-with-additional-input-data](https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462)<br>
[Is one-hot-encoding required for target label](https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function)<br>
[How to build a multimodal deep learning](https://www.drivendata.co/blog/hateful-memes-benchmark/)







