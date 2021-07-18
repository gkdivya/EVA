OpenCV Yolo:  SOURCE
- Run this above code on your laptop or Colab. 
- Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
- Run this image through the code above
- load the link to GitHub implementation of this
- Upload the annotated image by YOLO. 

Training Custom Dataset on Colab for YoloV3
Refer to this Colab File:  LINK
Refer to this GitHub  Repo
Download this dataset (Links to an external site.). This was annotated by EVA5 Students. Collect and add 25 images for the following 4 classes into the dataset shared:
- class names are in custom.names file. 
- you must follow exact rules to make sure that you can train the model. Steps are explained in the README.md file on github repo link above.
- Once you add your additional 100 images, train the model
Once done:

-Download a very small (~10-30sec) video from youtube which shows your classes. 
-Use ffmpeg to extract frames from the video. 
-Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
-Infer on these images using detect.py file. **Modify** detect.py file if your file names do not match the ones mentioned on GitHub. 
-python detect.py --conf-three 0.3 --output output_folder_name
-Use  ffmpeg  to convert the files in your output folder to video
-Upload the video to YouTube. 

Also run the model on 16 images that you have collected (4 for each class)
-Share the link to your GitHub project with the steps mentioned above - 1000 pts (only if all the steps were done, and it resulted in a trained model that you could run on video/images)
-Share the link to your YouTube video - 500 pts
-Share the link of your YouTube video shared on LinkedIn, Instagram, medium, etc! You have no idea how much you'd love people complimenting you! [TOTALLY OPTIONAL] - bonus points
-Share the link to the readme file where we can find the result of your model on YOUR 16 images. - 500 pts
-Bonus: YoloV4 Training on Colab! (Links to an external site.)

 
