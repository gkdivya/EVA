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
        (fc2): Linear(in_features=128, out_features=30, bias=True)
        (fc3): Linear(in_features=30, out_features=19, bias=True)
      )

## Number of parameters
Model has 6,384,925 trainable parameters.


## Loss Function
Because both the MNIST classification and the sum(0-18) are multi class classification problem, we have used Negative Log-Likelihood Loss Function.

In NLL, the model is punished for making the correct prediction with smaller probabilities and encouraged for making the prediction with higher probabilities. The logarithm does the punishment. NLL does not only care about the prediction being correct but also about the model being certain about the prediction with a high score. 

The Pytorch NLL Loss is expressed as: log(x,y) = -log(y)
x represents the actual value and y the predicted value.


## Training Log



## Inference



## References

https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462







