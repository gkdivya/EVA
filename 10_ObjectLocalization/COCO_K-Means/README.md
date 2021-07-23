# Coco dataset format

Andrew Ng proposed on a recent webinar - [data centric approach](https://www.youtube.com/watch?v=06-AZXmwHjo&ab_channel=DeepLearningAI) to solve real problems in AI, stating the importance of data annotation.

Coco dataset format is one of the widely used annotation format in Computer vision tasks. And it supports annotation for various vision usecases:

- object detection
- keypoint detection
- stuff segmentation
- panoptic segmentation
- image captioning

Coco annotation JSON have mainly four building blocks
![image](https://user-images.githubusercontent.com/17870236/126727905-88bf6a9a-c68a-42c1-9e2f-df9b784954ad.png)

- info: High-level information about the dataset.

      {
          "description": "COCO 2017 Dataset",
          "url": "http://cocodataset.org",
          "version": "1.0",
          "year": 2017,
          "contributor": "COCO Consortium",
          "date_created": "2017/09/01",
      }
      
- licenses: License detail of the images in the dataset.

      [
          {
              "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
              "id": 1,
              "name": "Attribution-NonCommercial-ShareAlike License",
          },
          {
              "url": "http://creativecommons.org/licenses/by-nc/2.0/",
              "id": 2,
              "name": "Attribution-NonCommercial License",
          },
          ....
       ]
- categories: List of categories/class information mapped to the ids. Categories can belong to a supercategory

      [{'supercategory': 'person', 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'} ...]
      
- images: Lists all images with unique ids and image information like width, height in the dataset without labeling info. 

      {
          "license": 3,
          "file_name": "000000391895.jpg",
          "coco_url": "http://images.cocodataset.org/train2017/000000391895.jpg",
          "height": 360,
          "width": 640,
          "date_captured": "2013-11-14 11:18:45",
          "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
          "id": 391895,
      }

- annotations: list of every  object annotation from every image in the dataset. It have the below details
      
      {
          "segmentation": [
              [
                  239.97,
                  260.24,
                  222.04,
                  270.49,
                  199.84,
                  253.41,
                  213.5,
                  227.79,
                  259.62,
                  200.46,
                  274.13,
                  ...
                  ...
              ]
          ],
          "area": 2765.1486500000005,
          "iscrowd": 0,
          "image_id": 558840,
          "bbox": [199.84, 200.46, 77.71, 70.88],
          "category_id": 58,
          "id": 156,
      }
      
     | key          | What it represents                                                                                                                                  |
     | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
     | segmentation | x and y coordinates for the vertices of the polygon around every object instance for the segmentation masks.                                        |
     | area         | Area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes. |
     | iscrowd      | Instances with iscrowd=True will be ignored during evaluation.                                                                                      |
     | image\_id    | An image identifier. It should be unique between all the images in the dataset, and is used during evaluation                                       |
     | bbox         | Coordinates of the N bounding boxes in \[x, y, w, h\] format                                                                                        |
     | category\_id | Category label for each bounding box                                                                                                                |
     | id           | Unique id of the annotation                                                                                                                         |


 # Coco Dataset 
 
 'Common Objects in Context' dataset is a set of high quality datasets widely used in State of art computer vision tasks. And guess what, all the images in the dataset are annotated using Pascal data format, Nah kidding.. **COCO data format**!
 
 ![image](https://user-images.githubusercontent.com/17870236/126731543-a51c58eb-51be-4026-9d1f-639d1ccc59a5.png)

## Class distribution
![image](https://user-images.githubusercontent.com/17870236/126731485-1ee00975-131c-4167-a7d3-719a0d7bff17.png)

# YOLO anchor boxes

## Anchor box
Anchor boxes are nothing but template bounding boxes. In the sense, Object detection models utilize the anchor boxes to make beter bounding box predictions.
First step is to identify good candidate anchor boxes in YOLO. </br>

YOLO V2 and V3 comes with a set of pre-defined anchor boxes which may not work out of box for custom data. Defining anchor boxes for the custom data will tune the model better and increase object detection accuracy 

## K-means clustering

K -means clustering algorithm is very famous algorithm in data science. This algorithm aims to partition n datapoints to k clusters. 
It mainly includes :
- Initialization : K means (i.e centroid) are generated at random.
- Assignment : Clustering formation by associating each datapoint with nearest centroid.
- Updating Cluster : Centroid of a newly created cluster becomes mean.
Assignment and Update are repetitively done until convergence. The final result is that the sum of squared errors minimized between points and their respective centroids.

### What it really does in determining anchor box?

In general, bounding boxes for objects are given by tuples of the form (x,y,w,h), we extract width and height from these coordinates, and normalize data with respect to image width and height.
There are two Metrics for K-means : 
- Euclidean distance
- IoU (Jaccard index)

Jaccard index = (Intersection between selected box and cluster head box)/(Union between selected box and cluster head box)

At initialization we can choose k random boxes as our cluster heads. Assign anchor boxes to respective clusters based on IoU value > threshold and calculate mean IoU of cluster.
This process would be repeated until convergence.


## Different K-Values and anchor boxes
| K=3 | K=4 |
| --- | --- |
|  ![image](https://user-images.githubusercontent.com/17870236/126734558-9daf823f-c1e4-4151-a1a5-28af0b11613f.png)     |  ![image](https://user-images.githubusercontent.com/17870236/126735193-e9670d6b-d05a-4cee-84bd-1f44602f5acd.png)   |
| K=5 | K=6 |
| ![image](https://user-images.githubusercontent.com/17870236/126734611-61520cf8-04a8-4ecd-9d10-b7d043006736.png)     |    ![image](https://user-images.githubusercontent.com/17870236/126734633-6a79cdf2-84e6-486e-92a1-f112478f0f86.png)  |

## Elbow Method
Elbow method is used to find the optimal number of clusters which helps choosing the right K-value. 

![image](https://user-images.githubusercontent.com/17870236/126769319-2f647f0a-4c4d-42e7-8c7a-5dfdfe832028.png)

## Clusters Based on IOU

Based on Average IOU the number of cluster is 4

![image](https://user-images.githubusercontent.com/42609155/126766050-6e5754a4-4691-43e5-adde-fac63064620a.png)


## Reference
https://cocodataset.org/ </br>
https://towardsdatascience.com/getting-started-with-coco-dataset-82def99fa0b8 </br>
https://github.com/svdesai/eda-coco </br>
https://towardsdatascience.com/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9 </br>
https://www.programmersought.com/article/8007145472/</br>
https://www.kaggle.com/backtracking/anchor-boxes-analysis-using-k-means</br>
https://github.com/matplotlib/matplotlib/blob/401aa2f7f2c155165b77d8e5f7cb98270f42054a/examples/statistics/errorbars_and_boxes.py</br>

