# ResNets and Higher Receptive Fields

Objective is to learn LR Schedulers, ResNets, and Higher Receptive Fields and to perform a convolution based on custom resnet architecture on CIFAR10 dataset. 


## Vision Library
Fully modularized library - [torch_cv_wrapper](https://github.com/gkdivya/torch_cv_wrapper) is built to perform object detection based on PyTorch on CIFAR10 dataset.

**Folder Structure**

    |── config
    |   ├── config.yaml    
    ├── dataloader  
    |   ├── albumentation.py 
    |   ├── load_data.py
    ├── model  
    |   ├── custommodel.py 
    |   ├── resnet.py
    ├── utils  
    |   ├── __init__.py 
    |   ├── train.py 
    |   ├── test.py 
    |   ├── plot_metrics.py 
    |   ├── helper.py 
    |   ├── gradcam.py 
    ├── main.py     
    ├── README.md  

### Final Model details

Epochs - 24 <br>
LR Scheduler - OneCycleLR

Following Data Augmentation is applied, refer [this](https://github.com/gkdivya/torch_cv_wrapper/blob/main/dataloader/albumentation.py) 
- RandomCrop(32, padding=4)
- CutOut(8x8)
- Horizontal Flip


### Training and Testing Logs

    

### Loss and Accuracy Plots




### Confusion Matrix and Accuracy by class





### Misclassified Images



### Model diagnostic with Grad-CAM


### Tensorboard Logs


### Collaborators

- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)



