## MNIST Fine tuning experiments

Using the MNIST image dataset and CNN, follow a step by step approach to finetune the network to achieve 99.4% validation accuracy with less than *8k Parameters* in 15 Epochs.

## Step 0 : [Basic Setup](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step 0_BasicSetup.ipynb)

#### Target:

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop

#### Results:

- Parameters: 6.3M
- Best Training Accuracy: 99.93
- Best Test Accuracy: 99.28

#### Analysis:

- Extremely Heavy Model for such a problem
- Model is over-fitting because the training accuracy is 99.93, but we are changing our model in the next step

## Step 1 : [Basic Skeleton](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step 1_BasicSkeleton.ipynb)

#### Target:

- Get the basic skeleton interms of convolution and placement of transition blocks (max pooling, 1x1's)
- Reduce the number of parameters as low as possible
- Add GAP and remove the last BIG kernel.

**Transition Block**: Max pooling followed by 1*1 to observe the accuracy

**1\*1 Convolution:** Its used in a convolutional neural network mainly to control the number of feature maps. It helps the network to go deeper without compromising on the extracted features with very less params.

[![image](https://user-images.githubusercontent.com/17870236/120821875-746acf80-c573-11eb-8ac6-df8ca4b341c1.png)](https://user-images.githubusercontent.com/17870236/120821875-746acf80-c573-11eb-8ac6-df8ca4b341c1.png)

**Global Average Pooling**: Initially GAP was introduced as an alternative to a fully connected layer in CNN networks. GAP is basically used to collapse all important features to 1*1*n.

Main intuition behind GAP is to give network the flexibility with the input image size since the output of the GAP is always consistent irrespective of the input image size.

[nn.AvgPool2D](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) For example nn.AvgPool2d(kernel_size=3) can convert 3*3*n channels 1*1*n

#### Results:

- Parameters: 4572
- Best Training Accuracy: 98.22
- Best Test Accuracy: 98.43

#### Analysis:

- We have structured our model in a readable way
- The model is lighter with less number of parameters
- The performace is reduced compared to previous models. Since we have reduced model capacity, this is expected, the model has capability to learn.
- Next, we will be tweaking this model further and increase the capacity to push it more towards the desired accuracy.

## Step 2 : [Add BatchNormalization](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step 2_Batch_Normalization.ipynb)

#### Target:

- Add Batch-norm to increase model efficiency

**Batch Normalization**: Even though we normalize the input values with Image normalization transform, parameter values changes while training the network. Batch Normalization is a technique to normalize the values getting passed within the network. Its normally applied to activations of a prior convolutional layer or on the inputs directly. [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) It helps with faster and smoother training process.

#### Results:

- Parameters: 5088
- Best Training Accuracy: 99.03
- Best Test Accuracy: 99.04

#### Analysis:

- There is slight increase in the number of parameters, as batch norm stores a specific mean and std deviation for each layer
- Model overfitting problem is rectified to an extent. But, we have not reached the target test accuracy 99.40%.

## Step 3 : [Add Dropout](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step 3_Dropout.ipynb)

#### Target:

- Add Regularization Dropout to each layer except last layer

**Dropout**: Dropout is a regularization technique where randomly selected neurons are ignored during training process. So the dropped out neurons - weight updates are not applied in back propagation.

[![image](https://user-images.githubusercontent.com/17870236/120822804-69fd0580-c574-11eb-8424-db85179b66e4.png)](https://user-images.githubusercontent.com/17870236/120822804-69fd0580-c574-11eb-8424-db85179b66e4.png)

#### Results:

- Parameters: 5088
- Best Training Accuracy: 97.94
- Best Test Accuracy: 98.64

#### Analysis:

- There is no overfitting at all. With dropout training will be harder, because we are droping the pixels randomly.
- The performance has droppped, we can further improve it.
- But with the current capacity,not possible to push it further.We can possibly increase the capacity of the model by adding a layer after GAP!

## Step 4 : [Increase Capacity (add Fully Connected Layer)](https://github.com/divya-r-kamat/DeepVision/blob/main/CNN Optimization/MNIST_IncreaseCapacity_Step6.ipynb)

#### Target:

- Increase model capacity at the end (add layer after GAP)

**Intuition behind adding a Fully connected layer after GAP**: Earlier, CNN networks designed with Fully connected layers were expensive and without GAP, using a fully connected layer was restricting the input image size. GAP replaced by Fully connected layer made more sense in designing CNN networks. But in recent few papers, its found using a GAP followed by FC layer, network is able to learn more complex features.

[GAP And FC - VGG-GAP-Model architecture](https://www.researchgate.net/figure/VGG-GAP-model-architecture-The-CNNs-in-the-model-included-five-max-pooling-layers-and_fig1_337277062)

#### Results:

- Parameters: 6,124
- Best Training Accuracy: 99.07
- Best Test Accuracy: 99.22

#### Analysis:

- The model parameters have increased
- There is no overfitting rather slight underfitting, thats fine dropout is doing its work , because we are adding dropout at each layer the model is able to capture the training accuracy
- However, we haven't reached 99.4 accuracy yet.
- Observing the missclassified images its good to try out some augmentation techniques as few images seems to be slightly rotated, and also image contrast needs to be considered

## Step 5 : [Image Augmentation](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step 5_Augmentation.ipynb)

#### Target:

- Add various Image augmentation techniques, image rotation, randomaffine, colorjitter .

#### Results:

- Parameters: 6124
- Best Training Accuracy: 97.61
- Best Test Accuracy: 99.32%

#### Analysis:

- he model is under-fitting, that should be ok as we know we have made our train data harder.
- However, we haven't reached 99.4 accuracy yet.
- The model seems to be stuck at 99.2% accuracy, seems like the model needs some additional capacity towards the end.

## Step 6 : [LR Scheduler](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step 6_LRScheduler.ipynb)

#### Target:

- Add some capacity (additional FC layer after GAP) to the model and added LR Scheduler

#### Results:

- Parameters: 6720
- Best Training Accuracy: 99.43
- Best Test Accuracy: 99.53

#### Analysis:

- The model parameters have increased
- The model is under-fitting. This is fine, as we know we have made our train data harder.
- LR Scheduler and the additional capacity after GAP helped getting to the desired target 99.4, Onecyclic LR is being used, this seemed to perform better than StepLR to achieve consistent accuracy in last few layers
