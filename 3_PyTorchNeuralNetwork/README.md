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
      Train set: Average loss: 1.2838
      Test set: Average loss: 1.333, MNist Accuracy:94.14, Sum_Accuracy:11.45

      Epoch 2 : 
      Train set: Average loss: 1.1986
      Test set: Average loss: 1.218, MNist Accuracy:97.0, Sum_Accuracy:16.58

      Epoch 3 : 
      Train set: Average loss: 1.1528
      Test set: Average loss: 1.152, MNist Accuracy:97.95, Sum_Accuracy:21.85

      Epoch 4 : 
      Train set: Average loss: 1.0859
      Test set: Average loss: 1.091, MNist Accuracy:98.64, Sum_Accuracy:35.26

      Epoch 5 : 
      Train set: Average loss: 1.0228
      Test set: Average loss: 1.038, MNist Accuracy:98.08, Sum_Accuracy:41.63

      Epoch 6 : 
      Train set: Average loss: 0.9620
      Test set: Average loss: 0.945, MNist Accuracy:98.82, Sum_Accuracy:55.11

      Epoch 7 : 
      Train set: Average loss: 0.8494
      Test set: Average loss: 0.854, MNist Accuracy:98.78, Sum_Accuracy:67.57

      Epoch 8 : 
      Train set: Average loss: 0.7657
      Test set: Average loss: 0.750, MNist Accuracy:98.98, Sum_Accuracy:79.26

      Epoch 9 : 
      Train set: Average loss: 0.6502
      Test set: Average loss: 0.650, MNist Accuracy:99.03, Sum_Accuracy:86.69

      Epoch 10 : 
      Train set: Average loss: 0.5678
      Test set: Average loss: 0.550, MNist Accuracy:99.07, Sum_Accuracy:91.18

      Epoch 11 : 
      Train set: Average loss: 0.4143
      Test set: Average loss: 0.452, MNist Accuracy:99.13, Sum_Accuracy:95.11

      Epoch 12 : 
      Train set: Average loss: 0.3541
      Test set: Average loss: 0.369, MNist Accuracy:98.98, Sum_Accuracy:97.21

      Epoch 13 : 
      Train set: Average loss: 0.2980
      Test set: Average loss: 0.295, MNist Accuracy:99.25, Sum_Accuracy:98.03

      Epoch 14 : 
      Train set: Average loss: 0.2284
      Test set: Average loss: 0.239, MNist Accuracy:99.19, Sum_Accuracy:98.75

      Epoch 15 : 
      Train set: Average loss: 0.1867
      Test set: Average loss: 0.198, MNist Accuracy:99.31, Sum_Accuracy:98.77

      Epoch 16 : 
      Train set: Average loss: 0.1342
      Test set: Average loss: 0.165, MNist Accuracy:99.35, Sum_Accuracy:98.94

      Epoch 17 : 
      Train set: Average loss: 0.1457
      Test set: Average loss: 0.144, MNist Accuracy:99.18, Sum_Accuracy:98.99

      Epoch 18 : 
      Train set: Average loss: 0.1146
      Test set: Average loss: 0.123, MNist Accuracy:99.15, Sum_Accuracy:99.03

      Epoch 19 : 
      Train set: Average loss: 0.0786
      Test set: Average loss: 0.108, MNist Accuracy:99.35, Sum_Accuracy:99.19

      Epoch 20 : 
      Train set: Average loss: 0.0823
      Test set: Average loss: 0.097, MNist Accuracy:99.26, Sum_Accuracy:99.02


<p align="center"><img src="https://user-images.githubusercontent.com/42609155/119068731-6610a580-ba02-11eb-955e-b47dbea188e9.png" width="500"></p>

## Inference



## References

[How to build custom Datasets for Images in Pytorch](https://youtu.be/ZoZHd0Zm3RY)<br>
[concatenate-layer-output-with-additional-input-data](https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462)<br>
[Is one-hot-encoding required for target label](https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function)<br>
[How to build a multimodal deep learning](https://www.drivendata.co/blog/hateful-memes-benchmark/)







