# Coco dataset format

First step in any computer vision task is to label or annotate the data. And coco dataset format is widely used one.
Coco annotation type supports below tasks.

- object detection
- keypoint detection
- stuff segmentation
- panoptic segmentation
- image captioning

Coco annotation JSON have mainly four building blocks
![image](https://user-images.githubusercontent.com/17870236/126727905-88bf6a9a-c68a-42c1-9e2f-df9b784954ad.png)

- info: High-level information about the dataset.
- licenses: License detail of the images in the dataset.
- categories: List of categories/class information mapped to the ids. Categories can belong to a supercategory
- images: Lists all images with unique ids and image information like width, height in the dataset without labeling info.
- annotations: list of every  object annotation from every image in the dataset

      boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
      labels (Int64Tensor[N]): the label for each bounding box
      image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
      area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
      iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
      (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
      (optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible.
      
![image](https://user-images.githubusercontent.com/17870236/126727272-752c4d9e-d626-4265-93f7-f7d732239ef8.png)

@ K-Means
