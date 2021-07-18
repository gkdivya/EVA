import cv2
import numpy as np

class OpenCVYolo:
  def __init__(self):
    self.yolo_weights = self.load_yolo()
    self.coco_classes = self.load_class_details()
 
  def load_yolo(self):
    # Load Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    return net

  def load_class_details(self):
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

  def identify_objects(self, image):
    # Loading image
    img = cv2.imread(image)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    
    layer_names = self.yolo_weights.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in self.yolo_weights.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(self.coco_classes), 3))

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    self.yolo_weights.setInput(blob)
    outs = self.yolo_weights.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(self.coco_classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # Saving the image
    cv2.imwrite('output/annotated_yolo_output.jpg', img)
    return None

if __name__ == "__main__":
   yolo = OpenCVYolo() 
   yolo.identify_objects('input/bag_scissors.jpg')