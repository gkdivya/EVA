## Data Analysis

Link to Exploratory Data Analysis on CIFAR10 dataset is [here](https://github.com/gkdivya/EVA/blob/main/7_AdvancedConcepts/dataanalysis/DataAnalysis_CIFAR10Dataset.ipynb)

- Data set consists of 60000 images
- There are 50000 training images and 10000 testing images
- All Images in the dataset are of same size, i.e of size 32 x 32
- Images are equally distributed across classes, there is no imbalance in the dataset

    <p align="left"><img src="https://user-images.githubusercontent.com/42609155/122554775-089a6380-d057-11eb-9f13-b798454d0702.png" width="400"></p>


- The dataset contains 10 classes, below are 10 sample images from each class

    <p align="left"><img src="https://user-images.githubusercontent.com/42609155/122555261-b1e15980-d057-11eb-8edf-5bf3a9e0256c.png" width="600"></p>
    
- From the images we can see that there are few images of dog and cat, which are black in color.
- Also, images of bird and aeroplane have some similarities

So, we have decided to apply following image augmentation techniques :
- HorizontalFlip
- ShiftScaleRotate
- CutOut Strategy
- Colorjitter
- Grayscale


