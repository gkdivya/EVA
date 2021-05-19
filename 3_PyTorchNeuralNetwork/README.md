### Problem Statement:


Write a neural network that can take 2 inputs:
* an image from MNIST dataset and
* a random number between 0 and 9

and gives two outputs:
* the "number" that was represented by the MNIST image, and
* the "sum" of this number with the random number that was generated and sent as the input to the network

![assign](https://user-images.githubusercontent.com/42609155/118740404-6aee2180-b869-11eb-9a42-d72efbc4f132.png)


Network can be written with fully connected layers and convolution layers. 


## Data Preparation:

The MNIST dataset is used as one of the input to the model, it contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels

MNIST Image is mapped to a random integer and the input dataset for training is prepared with below format

|Mnist Image|Mnist Label|One hot encoded Random Number|Sum of random number and Mnist Number| 
|----|-----|------|-----|

## Model



## Number of parameters



## Loss Function



## Training Log



## Inference







