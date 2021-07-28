# YoloV3 Training on Custom dataset

## Data Collection

The dataset was download from [here](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view) and additionally 100 images were collected with Creative common license for the below classes (25 images per class):
- hardhat
- vest
- boots
- mask

These 25 images where then merged with the above dataset.

## Data Annotation

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
        custom.names #your class names
        custom.txt #list of name of the images you want your network to be trained on. Currently we are using same file for test/train


## Model Training
- Created a folder 'weights' in the root (YoloV3) folder and copied the 'yolov3-spp-ultralytics.pt' file downloaded from [link](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0)
- Intialized the model with pretrained weights

### Logs

## Model Inference


