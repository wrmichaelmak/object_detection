# Object Detection script without implementing Pi Camera v2

import cv2 as cv
import numpy as np

# Load model
# Change frozen inference graph and label map path if needed
print ("[INFO] loading model...")
frozen_infer_graph_path = "frozen_inference_graph.pb"
label_map_path = "graph.pbtxt"
model = cv.dnn.readNet(frozen_infer_graph_path, label_map_path)
 

# Initialise classes and color box
classes = ["traffic cone"]
color = (0, 255, 255)

# Load image to model
#image_path = "dataset/test_set/33.jpg"
#image = cv.imread(image_path)
#
#(h, w) = image.shape[:2]
#
#image = cv.resize(image, (300, 300))
#blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), 0)
#
## pass blob through the network
#print("[INFO] detecting object")
#model.setInput(blob)
#detections = model.forward()
# confidence_level = 0.5 # minimum confidence level for object to be detected

# loop over the detections
# for i in np.arrange(0, detections.shape[2]):
	# confidence = detections[0, 0, i, 2]
	# if confidence > confidence_level
	



