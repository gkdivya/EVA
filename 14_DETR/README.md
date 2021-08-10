# DETR : End-to-End Object Detection with Transformers

- Take a look at this [post](https://opensourcelibs.com/lib/finetune-detr), which explains how to fine-tune DETR on a custom dataset. 
- Replicate the process and train the model yourself. The objectives are:
  - to understand how fine-tuning works
  - to understand architectural related concepts

## What is object detection? 

Object detection is a task where we want our model to distinguish the foreground objects from the background and predict the locations and the categories for the objects present in the image. Given a image if you need to determine if the image has a single particular object (say cat or dog) , we can use classification. However, if we have to get the location of that object its called classification and localization. But if there are multiple objects in an image and we want the pixel location of each and every object, then that is object detection. Object Detection is a problem which is not only a bit complex but also computationally expensive, due to the number of components to it.

Some of the previous techniques such as the RCNN family, YOLO(You Look Only Once) and SSD(Single Shot Detection) perform object detection in multistep manner they try to get Region Proposal using Region proposal network to come up with potential regions that may contain the object and then the concept of anchor boxes, NMS(non-max-suppression)and IOU is used to generate relevant boxes and identify the object. Although these concepts work, its a bit complex but also computationally expensive, with has all kinds of hyperparameters and layers 

## DETR (Detection Transformer)

The researchers at Facebook AI have come up with DETR, an innovative and efficient approach to solve the object detection problem. Detection Transformer leverages the transformer network(both encoder and the decoder) for Detecting Objects in Images . Facebook's researchers argue that for object detection one part of the image should be in contact with the other part of the image for greater result especially with ocluded objects and partially visible objects, and what's better than to use transformer for it.

The main objective behind DETR is effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode prior knowledge about the task and makes the process complex and computationally expensive.

![image](https://user-images.githubusercontent.com/42609155/128793888-3a29e1ee-64c4-412a-8435-6dc988e259f8.png)


### Architecture of DETR

The overall DETR architecture is actually pretty easy to understand. It contains three main components:

- a CNN backbone
- an Encoder-Decoder transformer
- a simple a feed-forward network as a head

![image](https://user-images.githubusercontent.com/42609155/128794251-4e86a306-bd86-4f15-a413-87779edffe93.png)


- the CNN backbone generates a feature map from the input image. Then the output of the CNN backbone is converted into a one-dimensional feature map that is passed to the Transformer encoder as input. The output of this encoder are N number of fixed length embeddings (vectors), where N is the number of objects in the image assumed by the model.

- The Transformer decoder decodes these embeddings into bounding box coordinates with the help of self and encoder-decoder attention mechanism.

- Finally, the feed-forward neural networks predict the normalized center coordinates, height, and width of the bounding boxes and the linear layer predicts the class label using a softmax function.
