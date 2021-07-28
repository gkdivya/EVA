# YoloV3 Training on Custom dataset

## Data Collection

The dataset was download from [here](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view) and additionally 100 images were collected for the below classes (25 images per class):
- hardhat
- vest
- boots
- mask

These 25 images where then merged with the above dataset.

## Annotation

Annotation tool from [this](https://github.com/miki998/YoloV3_Annotation_Tool) repo is and the installation steps as mentioned in the repo was followed to setup the tool and Annotate the images with bounding boxes.

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


## Initializing model with Pre-Trained weights

- Create a folder called weights in the root (YoloV3) folder
- Download from: https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0
- Place 'yolov3-spp-ultralytics.pt' file in the weights folder



