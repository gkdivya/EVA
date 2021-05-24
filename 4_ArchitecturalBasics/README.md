
# Architectural Basics

Modified the code given in [MNIST BASE CODE](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx) to achieve
99.4% validation accuracy with less than 20k Parameters and in less than 20 Epochs

Below important concepts were used/considered while designing the network:
Normalization:

Convolution Block:

Transition Block: 
Max pooling followed 1*1 convolution

Receptive Field:

Kernels:

Dropout:
Initial network, we designed without dropout. Observation:
Why we decided to add dropout:

Batch Size:

Important Notes considered:
Not to use Batch normalization, Activation function, Max pooling, dropout just before the last layer

GitHub Notebook Link:
Colab Link: 

## Experiments


## Best Model


## Training Log


## Validation Accuracy


## Reference Links
[KAGGLE]( https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99)

## Collaborators
Divya Kamat (divya.r.kamat@gmail.com)
Divya G K (gkdivya@gmail.com)
