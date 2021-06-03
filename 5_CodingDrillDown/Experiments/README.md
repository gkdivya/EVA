## MNIST Fine tuning experiments

Using the MNIST image dataset and CNN, follow a step by step approach to finetune the network to achieve 99.4% validation accuracy with less than _8k Parameters_ in 15 Epochs.

## Step 0 : Basic Setup

[Link to Notebook](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%200_BasicSetup.ipynb)

### Target:

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training  & Test Loop

### Results:
- Parameters: 6.3M
- Best Training Accuracy: 99.93
- Best Test Accuracy: 99.28

### Analysis:
- Extremely Heavy Model for such a problem
- Model is over-fitting because the training accuracy is 99.93, but we are changing our model in the next step

## Step 1 : Basic Skeleton

## Step 2 : Add BatchNormalization

## Step 3 : Add Dropout

## Step 4 : Increase Capacity (add Fully Connected Layer)

[Link to Notebook](https://github.com/divya-r-kamat/DeepVision/blob/main/CNN%20Optimization/MNIST_IncreaseCapacity_Step6.ipynb)

### Target:

- Increase model capacity at the end (add layer after GAP)

### Results:
- Parameters: 6,124
- Best Training Accuracy: 99.07
- Best Test Accuracy: 99.22

### Analysis:
- The model parameters have increased
- There is no overfitting rather slight underfitting, thats fine dropout is doing its work , because we are adding dropout at each layer the model is able to capture the training accuracy
- However, we haven't reached 99.4 accuracy yet.
- Observing the missclassified images its good to try out some augmentation techniques as few images seems to be slightly rotated, and also image contrast needs to be considered

## Step 5 : Image Augmentation

[Link to Notebook](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%205_Augmentation.ipynb)

### Target:

- Add various Image augmentation techniques, image rotation, randomaffine, colorjitter .

### Results:
- Parameters: 6124
- Best Training Accuracy: 97.61
- Best Test Accuracy: 99.32%

### Analysis:
- he model is under-fitting, that should be ok as we know we have made our train data harder. 
- However, we haven't reached 99.4 accuracy yet.
- The model seems to be stuck at 99.2% accuracy, seems like the model needs some additional capacity towards the end.

## Step 6 : LR Scheduler

[Link to Notebook](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/Experiments/MNIST_Step%206_LRScheduler.ipynb)

### Target:

- Add some capacity (additional FC layer after GAP) to the model and added LR Scheduler

### Results:
- Parameters: 6720
- Best Training Accuracy: 99.43
- Best Test Accuracy: 99.53

### Analysis:

- The model parameters have increased
- The model is under-fitting. This is fine, as we know we have made our train data harder.  
- LR Scheduler and the additional capacity after GAP helped getting to the desired target 99.4, Onecyclic LR is being used, this seemed to perform better than StepLR to achieve consistent accuracy in last few layers

