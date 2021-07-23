# Coco dataset format

Andrew Ng proposed on a recent webinar - [data centric approach](https://www.youtube.com/watch?v=06-AZXmwHjo&ab_channel=DeepLearningAI) to solve real problems in AI, stating that in his experience time spent on good dataset creation, data labeling is always worth it. 

Coco dataset format is widely used annotation format in Computer vision tasks. And it supports annotation for

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


 # Coco Dataset - Class distribution
![image](https://user-images.githubusercontent.com/17870236/126727272-752c4d9e-d626-4265-93f7-f7d732239ef8.png)

@ K-Means
