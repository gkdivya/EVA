##  ADVANCED TRAINING CONCEPTS
Class activation maps, Weight Updates, Optimizers & LR Schedulers

### Experiments
| ResNet Model | Normalization       | Scheduler         | Params | Training Accuracy | Test Accuracy | Experiment Config File |
| ------------ | ------------------- | ----------------- | ------ | ----------------- | ------------- | ---------------------- |
| ResNet18     | Batch Normalization | OnceCycleLR       |        |                   |               |                        |
| ResNet18     | Batch Normalization | LRScheduler       |        |                   |               |                        |
| ResNet18     | Batch Normalization | ReduceLROnPlateau |        |                   |               |                        |
| ResNet18     | Layer Normalization | OnceCycleLR       |        |                   |               |                        |
| ResNet18     | Layer Normalization | LRScheduler       |        |                   |               |                        |
| ResNet18     | Layer Normalization | ReduceLROnPlateau |        |                   |               |                        |
| ResNet34     | Batch Normalization | OnceCycleLR       |        |                   |               |                        |
| ResNet34     | Batch Normalization | LRScheduler       |        |                   |               |                        |
| ResNet34     | Batch Normalization | ReduceLROnPlateau |        |                   |               |                        |
| ResNet34     | Layer Normalization | OnceCycleLR       |        |                   |               |                        |
| ResNet34     | Layer Normalization | LRScheduler       |        |                   |               |                        |
| ResNet34     | Layer Normalization | ReduceLROnPlateau |        |                   |               |                        |

### Final Model

### Training and Testing Logs


### Loss and Accuracy Plots

### Misclassified Images


### Confusion Matrix and Accuracy by class


### Model diagnostic with Grad-CAM

### Folder structure

The code is fully modularised, we have built a repo named [torch_cv_wrapper](https://github.com/gkdivya/torch_cv_wrapper) for this assignment, and all the libraries are imported from this: 

    |── config
    |   ├── config.yaml    
    ├── dataloader  
    |   ├── albumentation.py 
    |   ├── load_data.py
    ├── model  
    |   ├── model.py 
    ├── utils  
    |   ├── __init__.py 
    |   ├── train.py 
    |   ├── test.py 
    |   ├── plot_metrics.py 
    |   ├── helper.py 
    |   ├── gradcam.py 
    ├── main.py     
    ├── README.md  

### Reference

- [Grad-Cam with PyTorch](https://github.com/kazuto1011/grad-cam-pytorch)

### Collaborators

- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)


