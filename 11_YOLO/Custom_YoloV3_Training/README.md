# YoloV3 Training on Custom dataset

[Data Collection](#data_collection)</br>
[Data Annotation](#data_annotation)</br>
[Model Training](#model_training)</br>
[Model Inference](#model_inference)</br>

## Data_Collection

The dataset was downloaded from [here](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view) and additionally 100 images with Creative common license were collected from google for the below classes (25 images per class):
- hardhat
- vest
- boots
- mask

These 100 images where then merged with the above dataset.

## Data_Annotation

Annotation tool from this [repo](https://github.com/miki998/YoloV3_Annotation_Tool) and the installation steps as mentioned in the repo was followed to setup the tool and annotate the images with bounding boxes.

<img src="https://user-images.githubusercontent.com/17870236/127248717-cf045180-5342-443c-aada-205b1bb18d9b.png" width=600 height=400/>



    data
      --customdata
        --images/
          --img001.jpg
          --img002.jpg
          --...
        --labels/
          --img001.txt
          --img002.txt
          --...
        custom.data #data file
        custom.names #class names
        custom.txt #list of name of the images the network to be trained on. Currently we are using same file for test/train


## Model_Training
- Created a folder 'weights' in the root (YoloV3) folder and copied the 'yolov3-spp-ultralytics.pt' file downloaded from [link](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)
- In 'yolov3-custom.cfg' file, 
    - Changed the filters as ((4 + 1 + 4(No of classes))*3) 27
    - Changed all entries of classes as 4
    - Changed burn_in to 100
    - Changed max_batches to 5000
    - Changed steps to 4000,4500
- Changed the class names in custom.names</br>
    hardhat</br>
    vest</br>
    mask</br>
    boots</br>
- Visualization of Batch images - Image Augmentation with YOLO - Bag of Freebies
<img src="https://user-images.githubusercontent.com/17870236/127417094-e8dbd311-4cab-4bdd-a773-76a5f1f49ca6.png" />


### Logs
<img src="https://user-images.githubusercontent.com/17870236/127417190-3671a673-3676-4d1b-8781-8077b508e56c.png" width=800 height=600/>

## Model_Inference

### Inference on 16 Images: 


![image](https://user-images.githubusercontent.com/42609155/127581798-02667c0f-0e9c-4247-aefd-25c9dab9cf44.png)

![image](https://user-images.githubusercontent.com/42609155/127582330-8d018586-be3b-4e0e-9461-7a8f006d3cda.png)


### Inference on Youtube video listing the 4 classes 

[![IMAGE ALT TEXT](http://img.youtube.com/vi/RaxeV2HvKfo/0.jpg)](http://www.youtube.com/watch?v=RaxeV2HvKfo "Custom YoloV3 Training")

## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
