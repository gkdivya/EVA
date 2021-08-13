# DETR : End-to-End Object Detection with Transformers


- Take a look at this [post](https://opensourcelibs.com/lib/finetune-detr), which explains how to fine-tune DETR on a custom dataset. 
- The goal is to the process and train the model yourself. The objectives are:
  - to understand how fine-tuning works
  - to understand architectural related concepts

Let's first understand Object detection and architectural related concepts for DETR.

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

The loss is calculated by computing the bipartite matching loss. The model makes a predefined number of predictions, and each of the predictions are computed in parallel.

![image](https://user-images.githubusercontent.com/42609155/128794251-4e86a306-bd86-4f15-a413-87779edffe93.png)

### The CNN Backbone

The CNN backbone generates a feature map from the input image. Then the output of the CNN backbone is converted into a one-dimensional feature map and added to a positional encoding, which is fed into a Transformer consisting of an Encoder and a Decoder in a manner quite similar to the Encoder-Decoder transformer described in the original Transformer paper (http://arxiv.org/abs/1706.03762). The output of this encoder are N number of fixed length embeddings (vectors), where N is the number of objects in the image assumed by the model.

Assume that our input image xᵢₘ of height H₀, width W₀, and three input channels. CNN backbone consists of a (pretrained) CNN (usually ResNet), which we use to generate C lower dimensional features having width W and height H (In practice, we set C=2048, W=W₀/32 and H=H₀/32).
This leaves us with C two-dimensional features, and since we will be passing these features into a transformer, each feature must be reformatted in a way that will allow the encoder to process each feature as a sequence. This is done by flattening the feature matrices into an H⋅W vector, and then concatenating each one.

![image](https://user-images.githubusercontent.com/42609155/128952635-4ab55e9f-a230-45ee-b91d-4dcc96285303.png)

The flattened convolutional features are added to a spatial positional encoding which can either be learned, or pre-defined.

### The Transformer

The Transformer decoder decodes these embeddings into bounding box coordinates with the help of self and encoder-decoder attention mechanism. The transformer is nearly identical to the original encoder-decoder architecture. The difference is that each decoder layers decodes each of the N (the predefined number of) objects in parallel. The model also learns a set of N object queries which are (similar to the encoder) learned positional encodings.

![image](https://user-images.githubusercontent.com/42609155/128952782-c90eec14-7ef1-4f66-a792-9e0b9957058a.png)

#### Object Queries

An intuitive way of understanding the object queries is by imagining that each object query is a person. And each person can ask the, via attention, about a certain region of the image. So one object query will always ask about what is in the center of an image, and another will always ask about what is on the bottom left, and so on.

### Prediction heads

Finally, the output of the decoder is then fed into a fixed number of Prediction Heads which consist of a predefined number of feed forward networks.  The feed-forward neural networks predict the normalized center coordinates, height, and width of the bounding boxes and the linear layer predicts the class label using a softmax function.


Besides the transformer part in architecture, DETR also adopt two major components from previous research.
- Bipartite Matching Loss
- Parallel Decoding

## Bipartite Matching Loss 

Bipartite matching loss is designed based on Hungarian algorithm. Unlike other object detection models where multiple bounding boxes are matched to one ground truth box, DETR uses bipartite matching, which is one-vs-one matching. By performing one-vs-one matching, its able to significantly reduce low-quality predictions, and achieve eliminations of output reductions like NMS.

DETR frameworks uses a set based global loss that enforces unique prediction through bipartite matching. DETR always infers a fixed set of ‘N’ predictions where ‘N’ is significantly larger than the number of classes. Let y denote the ground truth set of objects and y-hat the set of N predictions. The bipartite matching between the ground truth and predicted is achieved by Hungarian algorithm which determines the optimal assignment between ground truth and prediction. 

The bipartite matching is denoted as the sum of matching loss Lmatch with optimal assignment denoted by

![image](https://user-images.githubusercontent.com/42609155/129122096-1dec3330-501c-4c1a-980d-3e992a2f8941.png)


Lmatch the matching loss is the sum of class prediction loss and bounding box difference loss.

![image](https://user-images.githubusercontent.com/42609155/129122121-f580f038-d047-4a70-8f67-835583a85478.png)

# FineTune DETR

Now, lets understand how fine-tuning works, here we are going to fine-tune Facebook's DETR (DEtection TRansformer) on balloon dataset (custom object detection dataset) The goal for the model is to recognize balloons in pictures.

## Dataset

DETR will be fine-tuned on a custom dataset: the balloon dataset. The dataset is taken from [here](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon). The balloon dataset comes in the VIA annotation format. However, to prepare the annotations for the model, DETR expects them to be in COCO format. 

The following [Github repo](https://github.com/woctezuma/VIA2COCO) is used to convert annotations from VIA format to COCO format

There are 61 images in the training set, and 13 images in the validation set.

The directory structure would be as following:

    path/to/coco/
    ├ annotations/  # JSON annotations
    │  ├ annotations/custom_train.json
    │  └ annotations/custom_val.json
    ├ train2017/    # training images
    └ val2017/      # validation images


## Training

Model was trained for 150 epochs and the link to the notebook can be found [here](https://github.com/gkdivya/EVA/blob/main/14_DETR/FineTuning_DETR(DEtection_TRansformer).ipynb)

## Metrics

Metrics to monitor the training include:

- the Average Precision (AP), which is the primary challenge metric for the COCO dataset,
- losses (total loss, classification loss, l1 bbox distance loss, GIoU loss),
- errors (cardinality error, class error).

![image](https://user-images.githubusercontent.com/42609155/129305315-53470ec2-8a9f-4c7f-9ebf-53cc9667c120.png)

![image](https://user-images.githubusercontent.com/42609155/129305341-b9c9129e-271f-4841-a8ab-c104975325f5.png)

![image](https://user-images.githubusercontent.com/42609155/129305376-951fa8a9-909e-4b2e-9904-f49a66f7db2e.png)


## Predictions on Validation image

![image](https://user-images.githubusercontent.com/42609155/129305729-0d601f4a-8b39-4198-b5f1-10716d521d6f.png)



## Reference

- https://www.cellstrat.com/2020/08/07/end-to-end-object-detection-with-transformers/
- https://github.com/woctezuma/finetune-detr

## Collaborators

- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
