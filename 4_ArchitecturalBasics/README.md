
# Architectural Basics

Modified the code given in [MNIST Base Code](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx) to achieve 99.4% validation accuracy with less than 20k Parameters and in less than 20 Epochs.

Below important concepts were used/considered while designing the network:
- Convolution Block 
- Receptive Field
- Kernels
- Transition Block
- Batch Normalization
- Image Augmenation
- Dropout
- Batch Size

Intuition behind the step by step approach we followed to reduce the number of parameters and to improve accuracy with less params:
- At first, we reduced the number of parameters using Convolution Blocks with less number of output channels (removed 64, 128, 256 and 512 for every layer) and removed bias
- Transition Block (max pooling followed by 1x1) with Receptive Field Concept.
- Added a GAP layer to convert 2d to 1d
- We used Batch Normalization after every layer
- Use Augmentation techniques like image rotation
- Added Dropout after every layer, we added dropout after we tried out all possible options. Adding drop out, helped reduced the gap between training and test loss.
- Lastly, use Learning Rate to tune the model

<b>Important Notes considered<b>:
We have ensured not to use Batch normalization, Activation function, Max pooling, dropout just before the last layer. And to use always kernel size of 3 * 3


## Experiments

Batch Size = 128 <br>
Epochs = 19

|Experiment| #Parameters | Batch Normalization | Transition & GAP Layer | Augmentation | Dropout | Learning Rate| Validation Accuracy | 
|-------|---|---|---|---|---|---|---|
|[MNIST_With Less Params](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/Experiments/MNIST_Exp1_WithLessParams.ipynb) |5490|No|No|No|No|0.01|98%|


## Final Best Model

[GitHub Notebook Link](https://github.com/gkdivya/EVA/blob/main/4_ArchitecturalBasics/MNIST_Architecture_Basics.ipynb) <br>
[Colab Link](https://colab.research.google.com/github/gkdivya/EVA/blob/main/4_ArchitecturalBasics/MNIST_Architecture_Basics.ipynb#scrollTo=ZnerqwVwNxYa)

- Model has 17K parametes
- Added batch norm after every layer except last layer
- Added Transition layer (Max pool followed by 1x1) to reduce the number of channels after every block
- Add GAP layer
- Also added a FC layer after GAP
- Used Augmentation like image rotation
- Added Drop out of 0.1 after every layer except last layer
- The model was trained with a learning rate of 0.015 amd momentum of 0.9 
- Network was trained for 19 epochs with batch size of 128
- Achieved a test accuracy of 99.42% at 19th epoch


## Training Log


## Validation Accuracy


![accuracy_plot](https://user-images.githubusercontent.com/42609155/119747046-7c5dac00-beaf-11eb-988c-2c683bb6ba0a.png)


## Reference Links
[Kaggle Notebook]( https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99)

## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
