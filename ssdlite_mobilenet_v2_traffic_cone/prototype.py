# Object Detection script without implementing Pi Camera v2

import cv2 as cv
import numpy as np
import math

# Load model
# Change frozen inference graph and label map path if needed
print ("[INFO] loading model...")
frozen_infer_graph_path = "frozen_inference_graph.pb"
label_map_path = "graph.pbtxt"
model = cv.dnn.readNet(frozen_infer_graph_path, label_map_path)


# Initialise classes and color box
CLASSES = ["traffic cone"]
COLORS = (0, 255, 255)

# Load image to model
image_path = "dataset/test_set/33.jpg"
image = cv.imread(image_path)

(h, w) = image.shape[:2]

image = cv.resize(image, (300, 300))
blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), 0)

# pass blob through the network
print("[INFO] detecting object")
model.setInput(blob)
detections = model.forward()
confidence_level = 0.5 # minimum confidence level for object to be detected

 #loop over the detections
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confidence_level:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
     
    	# display the prediction
        label = "{}: {:.2f}%".format(CLASSES, confidence * 100)
        print("[INFO] {}".format(label))
        cv.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        # Find midpoint
        middle_x = (startX - endX) / 2
        middle_x = int(math.sqrt(middle_x * middle_x))
        middle_x = startX + middle_x
        middle_y = (startY - endY) / 2
        middle_y = int(math.sqrt(middle_y * middle_y))
        middle_y = startY + middle_y
        #cv.rectangle(image, (start_middle_x, start_middle_y), (end_middle_x, end_middle_y), COLORS[idx], 1)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv.putText(image, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
    
        
        #cv.rectangle(image, (start_middle_x, start_middle_y), (end_middle_x, end_middle_y), (255,255,0), 1)
        
# show the output image
cv.imshow("Output", image)
cv.waitKey(0)
	



