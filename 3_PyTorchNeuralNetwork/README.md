
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
Because both the MNIST classification and the sum(0-18) are multi class classification problem, we have used Negative Log-Likelihood Loss Function for each of the outputs and averaged the loss:

 Loss = (Mnist NLL Loss + Addition NLL Loss)/2

In NLL, the model is punished for making the correct prediction with smaller probabilities and encouraged for making the prediction with higher probabilities. The logarithm does the punishment. NLL does not only care about the prediction being correct but also about the model being certain about the prediction with a high score. 

The Pytorch NLL Loss is expressed as: log(x,y) = -log(y)
x represents the actual value and y the predicted value.


## Training Log

      Epoch 1 : 
      Train set: Average loss: 1.3915
      val set: Average loss: 1.368, MNist Accuracy:93.04, Sum_Accuracy:10.88

      Epoch 2 : 
      Train set: Average loss: 1.2572
      val set: Average loss: 1.248, MNist Accuracy:96.16, Sum_Accuracy:15.22

      Epoch 3 : 
      Train set: Average loss: 1.1600
      val set: Average loss: 1.188, MNist Accuracy:96.9, Sum_Accuracy:17.76

      Epoch 4 : 
      Train set: Average loss: 1.1511
      val set: Average loss: 1.125, MNist Accuracy:97.82, Sum_Accuracy:29.12

      Epoch 5 : 
      Train set: Average loss: 1.0766
      val set: Average loss: 1.075, MNist Accuracy:98.24, Sum_Accuracy:38.64

      Epoch 6 : 
      Train set: Average loss: 0.9500
      val set: Average loss: 1.012, MNist Accuracy:98.04, Sum_Accuracy:41.8

      Epoch 7 : 
      Train set: Average loss: 0.9224
      val set: Average loss: 0.926, MNist Accuracy:98.38, Sum_Accuracy:57.8

      Epoch 8 : 
      Train set: Average loss: 0.7930
      val set: Average loss: 0.852, MNist Accuracy:98.4, Sum_Accuracy:66.42

      Epoch 9 : 
      Train set: Average loss: 0.7626
      val set: Average loss: 0.767, MNist Accuracy:98.52, Sum_Accuracy:77.82

      Epoch 10 : 
      Train set: Average loss: 0.6550
      val set: Average loss: 0.676, MNist Accuracy:98.68, Sum_Accuracy:84.98

      Epoch 11 : 
      Train set: Average loss: 0.6017
      val set: Average loss: 0.592, MNist Accuracy:98.62, Sum_Accuracy:90.2

      Epoch 12 : 
      Train set: Average loss: 0.4858
      val set: Average loss: 0.507, MNist Accuracy:98.8, Sum_Accuracy:93.72

      Epoch 13 : 
      Train set: Average loss: 0.4871
      val set: Average loss: 0.434, MNist Accuracy:98.82, Sum_Accuracy:95.58

      Epoch 14 : 
      Train set: Average loss: 0.3552
      val set: Average loss: 0.364, MNist Accuracy:98.72, Sum_Accuracy:97.52

      Epoch 15 : 
      Train set: Average loss: 0.2759
      val set: Average loss: 0.303, MNist Accuracy:98.82, Sum_Accuracy:98.24

      Epoch 16 : 
      Train set: Average loss: 0.2438
      val set: Average loss: 0.251, MNist Accuracy:98.92, Sum_Accuracy:98.38

      Epoch 17 : 
      Train set: Average loss: 0.1744
      val set: Average loss: 0.214, MNist Accuracy:99.02, Sum_Accuracy:98.62

      Epoch 18 : 
      Train set: Average loss: 0.1420
      val set: Average loss: 0.183, MNist Accuracy:98.86, Sum_Accuracy:98.64

      Epoch 19 : 
      Train set: Average loss: 0.1160
      val set: Average loss: 0.159, MNist Accuracy:99.0, Sum_Accuracy:98.96

      Epoch 20 : 
      Train set: Average loss: 0.1003
      val set: Average loss: 0.143, MNist Accuracy:98.88, Sum_Accuracy:98.82

      Epoch 21 : 
      Train set: Average loss: 0.0719
      val set: Average loss: 0.123, MNist Accuracy:99.06, Sum_Accuracy:98.82

      Epoch 22 : 
      Train set: Average loss: 0.0746
      val set: Average loss: 0.117, MNist Accuracy:98.88, Sum_Accuracy:98.64

      Epoch 23 : 
      Train set: Average loss: 0.0590
      val set: Average loss: 0.105, MNist Accuracy:99.06, Sum_Accuracy:98.82

      Epoch 24 : 
      Train set: Average loss: 0.0606
      val set: Average loss: 0.098, MNist Accuracy:99.0, Sum_Accuracy:98.76

      Epoch 25 : 
      Train set: Average loss: 0.0566
      val set: Average loss: 0.090, MNist Accuracy:99.0, Sum_Accuracy:98.84


<p align="center"><img src="https://user-images.githubusercontent.com/42609155/119095568-10eb8880-ba30-11eb-910d-f766d3a1d237.png" width="500"></p>

## Test Loss

      Test set: Average loss: 0.072, MNist Accuracy:99.27, Sum_Accuracy:99.24
      
## Inference

<p align="center">
    <img src="https://user-images.githubusercontent.com/42609155/119095810-60ca4f80-ba30-11eb-974f-e123a9e02fb3.png" width="200">
    <img src="https://user-images.githubusercontent.com/42609155/119095818-645dd680-ba30-11eb-980d-8fe1618ff0be.png" width="200">
    <img src="https://user-images.githubusercontent.com/42609155/119095826-66279a00-ba30-11eb-8720-495a8134885b.png" width="200">
    <img src="https://user-images.githubusercontent.com/42609155/119100042-2adb9a00-ba35-11eb-96a9-b3ef330a6954.png" width="200">
</p>

## References

[How to build custom Datasets for Images in Pytorch](https://youtu.be/ZoZHd0Zm3RY)<br>
[concatenate-layer-output-with-additional-input-data](https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462)<br>
[Is one-hot-encoding required for target label](https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function)<br>
[How to build a multimodal deep learning](https://www.drivendata.co/blog/hateful-memes-benchmark/)







