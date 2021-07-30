# Object Detection using YOLO v3

## OpenCV Yolo:  
- Followed the steps listed [here](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)
- Took a image with object which is there in COCO data set (search for COCO classes to learn). 
- Ran this image through the code above
- Uploaded the annotated image by YOLO OpenCV. 



## Training Custom Dataset on Colab for YoloV3
- Refer to [this](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS) Colab File
- Refer to [this](https://github.com/theschoolofai/YoloV3) GitHub Repo
- Download [this](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view) dataset
- Collect and add 25 images for the following 4 classes into the dataset shared:
  - class names are hardhat, vest, mask and boots, its also listed in custom.names file 
  - follow exact rules to make sure that you can train the model. Steps are explained in the [README.md](https://github.com/theschoolofai/YoloV3/blob/master/README.md) file on github repo link above.
- Once additional 100 images are added, train the model

Next,

- Download a very small (~10-30sec) video from youtube which shows the 4 classes. 
- Use ffmpeg to extract frames from the video. 
- Infer on these images using detect.py file.  
- python detect.py --conf-three 0.3 --output output_folder_name
- Use  ffmpeg  to convert the files in your output folder to video
- Upload the video to YouTube. 



 ## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
