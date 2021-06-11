
The code is modularized and has seperate functions:
- We have a single model.py file that includes a function to choose the norm_type as either GN/LN/BN to decide which normalization to include and creates the network architecture accordingly
- then we have a training function
- a testing function
- a function to plot the metrics
- and also a few helper fucntions to load the data, print missclassified images


The base model was taken from the previous [assignment](https://github.com/gkdivya/EVA/blob/main/5_CodingDrillDown/MNIST_BestModel.ipynb)
