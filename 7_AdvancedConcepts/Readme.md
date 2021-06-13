# Advanced Concepts

Advance Convolutions, Attention and Image Augmentation: Depthwise, Pixel Shuffle, Dilated, Transpose, Channel Attention, and Albumentations Library

**Objective** : To achieve 87% accuracy with total Params less than 100k in CIFAR10 dataset and
 
1.  To use GPU ✓
2.  To use architecture C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) 
3.  To use Dilated kernels here instead of MP or strided convolution
4.   To achieve total Receptive Field more than 52
5.   To use Depthwise Separable Convolution at least in 2 of the layers
6.   To use Dilated Convolution at least in one of the layers
7.   To use GAP (compulsory mapped to # of classes):- CANNOT add FC after GAP to target #of classes
8.   To use correct Normalization values by having computing mean and std value in Transform ✓
9.   To use albumentation library and apply:
> *   horizontal flip
> *   shiftScaleRotate 
> *   coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
> *   grayscale

## Albumentation Library

- Faster than TorchVision inbuilt augmentation
- Better support for segmentation and object detection dataset with "label preserving transformations"


## Receptive Field



## Misclassified Image Analysis



## References
- https://www.youtube.com/watch?v=rAdLwKJBvPM </br>
- https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py </br>


## Collaborators and the contribution details:

- Divya Kamat (divya.r.kamat@gmail.com)</br>
   Modularising the code and creation of skeleton for the complete project </br>

- Divya G K (gkdivya@gmail.com)</br>
   Albumentation </br>

- Sarang (jaya.sarangan@gmail.com)</br>
   Dilated Convolution </br>

- Garvit Garg (garvit.gargs@gmail.com)</br>
   Depthwise Seperable convolution </br>
   
- All collaborators</br>
   README file </br>

