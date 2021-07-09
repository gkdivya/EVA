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

The final notebook is [here](https://github.com/gkdivya/EVA/blob/main/9_ResNetsandHigherReceptiveFields/CIFAR10_Image_Classification_CustomResnet.ipynb) with a test accuracy of 92.89% at 24th epoch

Epochs - 24 <br>
LR Scheduler - OneCycleLR

Following Data Augmentation is applied, refer [this](https://github.com/gkdivya/torch_cv_wrapper/blob/main/dataloader/albumentation.py) 
- RandomCrop(32, padding=4)
- CutOut(8x8)
- Horizontal Flip

### LR Search Plot

![image](https://user-images.githubusercontent.com/42609155/125091377-f8255800-e0ed-11eb-8ca3-dd72e999ed12.png)


### Training and Testing Logs

    Epoch 1:
    Loss=1.2540031671524048 Batch_id=97 LR=0.01400 Accuracy=38.43: 100%|██████████| 98/98 [00:13<00:00,  7.23it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0026, Accuracy: 5399/10000 (53.99%)

    Epoch 2:
    Loss=1.309002161026001 Batch_id=97 LR=0.02870 Accuracy=55.39: 100%|██████████| 98/98 [00:13<00:00,  7.50it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0025, Accuracy: 5801/10000 (58.01%)

    Epoch 3:
    Loss=1.023582100868225 Batch_id=97 LR=0.04685 Accuracy=64.82: 100%|██████████| 98/98 [00:13<00:00,  7.44it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0021, Accuracy: 6498/10000 (64.98%)

    Epoch 4:
    Loss=0.7895511388778687 Batch_id=97 LR=0.06149 Accuracy=71.44: 100%|██████████| 98/98 [00:13<00:00,  7.46it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0015, Accuracy: 7241/10000 (72.41%)

    Epoch 5:
    Loss=0.7681843042373657 Batch_id=97 LR=0.06700 Accuracy=75.63: 100%|██████████| 98/98 [00:13<00:00,  7.34it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0015, Accuracy: 7501/10000 (75.01%)

    Epoch 6:
    Loss=0.517564058303833 Batch_id=97 LR=0.06653 Accuracy=79.26: 100%|██████████| 98/98 [00:13<00:00,  7.26it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0012, Accuracy: 7938/10000 (79.38%)

    Epoch 7:
    Loss=0.4779468774795532 Batch_id=97 LR=0.06517 Accuracy=81.12: 100%|██████████| 98/98 [00:13<00:00,  7.38it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0013, Accuracy: 7832/10000 (78.32%)

    Epoch 8:
    Loss=0.4683723449707031 Batch_id=97 LR=0.06294 Accuracy=82.59: 100%|██████████| 98/98 [00:13<00:00,  7.23it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0011, Accuracy: 8203/10000 (82.03%)

    Epoch 9:
    Loss=0.4887462258338928 Batch_id=97 LR=0.05990 Accuracy=83.93: 100%|██████████| 98/98 [00:13<00:00,  7.16it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0012, Accuracy: 7986/10000 (79.86%)

    Epoch 10:
    Loss=0.49753597378730774 Batch_id=97 LR=0.05615 Accuracy=84.81: 100%|██████████| 98/98 [00:13<00:00,  7.20it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0010, Accuracy: 8187/10000 (81.87%)

    Epoch 11:
    Loss=0.3796556890010834 Batch_id=97 LR=0.05178 Accuracy=85.33: 100%|██████████| 98/98 [00:13<00:00,  7.24it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0011, Accuracy: 8173/10000 (81.73%)

    Epoch 12:
    Loss=0.4380437731742859 Batch_id=97 LR=0.04691 Accuracy=85.75: 100%|██████████| 98/98 [00:13<00:00,  7.12it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0012, Accuracy: 7971/10000 (79.71%)

    Epoch 13:
    Loss=0.4117755889892578 Batch_id=97 LR=0.04167 Accuracy=86.66: 100%|██████████| 98/98 [00:13<00:00,  7.28it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0009, Accuracy: 8356/10000 (83.56%)

    Epoch 14:
    Loss=0.4038155972957611 Batch_id=97 LR=0.03621 Accuracy=87.44: 100%|██████████| 98/98 [00:13<00:00,  7.06it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0009, Accuracy: 8511/10000 (85.11%)

    Epoch 15:
    Loss=0.41934654116630554 Batch_id=97 LR=0.03068 Accuracy=87.71: 100%|██████████| 98/98 [00:13<00:00,  7.21it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0011, Accuracy: 8198/10000 (81.98%)

    Epoch 16:
    Loss=0.2969716191291809 Batch_id=97 LR=0.02522 Accuracy=89.03: 100%|██████████| 98/98 [00:13<00:00,  7.10it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0009, Accuracy: 8537/10000 (85.37%)

    Epoch 17:
    Loss=0.3367551863193512 Batch_id=97 LR=0.01999 Accuracy=89.68: 100%|██████████| 98/98 [00:13<00:00,  7.14it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0008, Accuracy: 8609/10000 (86.09%)

    Epoch 18:
    Loss=0.2761857509613037 Batch_id=97 LR=0.01513 Accuracy=90.82: 100%|██████████| 98/98 [00:13<00:00,  7.03it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0009, Accuracy: 8485/10000 (84.85%)

    Epoch 19:
    Loss=0.2435772716999054 Batch_id=97 LR=0.01077 Accuracy=92.27: 100%|██████████| 98/98 [00:13<00:00,  7.12it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0007, Accuracy: 8786/10000 (87.86%)

    Epoch 20:
    Loss=0.20730707049369812 Batch_id=97 LR=0.00703 Accuracy=93.71: 100%|██████████| 98/98 [00:13<00:00,  7.18it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0006, Accuracy: 9001/10000 (90.01%)

    Epoch 21:
    Loss=0.16077126562595367 Batch_id=97 LR=0.00401 Accuracy=95.45: 100%|██████████| 98/98 [00:13<00:00,  7.22it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0005, Accuracy: 9150/10000 (91.50%)

    Epoch 22:
    Loss=0.12195223569869995 Batch_id=97 LR=0.00180 Accuracy=96.68: 100%|██████████| 98/98 [00:13<00:00,  7.14it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0005, Accuracy: 9223/10000 (92.23%)

    Epoch 23:
    Loss=0.12343305349349976 Batch_id=97 LR=0.00045 Accuracy=97.60: 100%|██████████| 98/98 [00:13<00:00,  7.12it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0004, Accuracy: 9277/10000 (92.77%)

    Epoch 24:
    Loss=0.06058374419808388 Batch_id=97 LR=0.00000 Accuracy=97.87: 100%|██████████| 98/98 [00:13<00:00,  7.23it/s]
    Test set: Average loss: 0.0004, Accuracy: 9289/10000 (92.89%)

### Loss and Accuracy Plots

![image](https://user-images.githubusercontent.com/42609155/125091643-3753a900-e0ee-11eb-9253-a8619b454999.png)
![image](https://user-images.githubusercontent.com/42609155/125091723-4b97a600-e0ee-11eb-9cff-ac2831f55287.png)


### Confusion Matrix and Accuracy by class


![image](https://user-images.githubusercontent.com/42609155/125091862-6a963800-e0ee-11eb-810d-fd223602bca8.png)

    Accuracy of plane : 94 %
    Accuracy of   car : 96 %
    Accuracy of  bird : 89 %
    Accuracy of   cat : 84 %
    Accuracy of  deer : 93 %
    Accuracy of   dog : 87 %
    Accuracy of  frog : 96 %
    Accuracy of horse : 95 %
    Accuracy of  ship : 95 %
    Accuracy of truck : 95 %


### Misclassified Images

![image](https://user-images.githubusercontent.com/42609155/125092172-afba6a00-e0ee-11eb-95ad-ff0ac37f3b14.png)


### Model diagnostic with Grad-CAM

![image](https://user-images.githubusercontent.com/42609155/125092114-9dd8c700-e0ee-11eb-936f-160abc57fa8a.png)


### Tensorboard Logs

![image](https://user-images.githubusercontent.com/42609155/125092308-d5477380-e0ee-11eb-863d-271eb958a6e4.png)


### Collaborators

- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)



