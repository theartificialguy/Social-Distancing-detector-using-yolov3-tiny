import cv2
import numpy as np
import random
import os
import time
from scipy.spatial import distance as dist

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny-custom-social.cfg")

distance_threshold = 50

cap = cv2.VideoCapture('humans.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, ((int(cap.get(3)), int(cap.get(4))))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
 
MIN_DISTANCE = 50
ret = True

while ret:

    ret, img = cap.read()
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id!=0:
                continue
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    results = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    violate = set()
		if len(results) >= 2:
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
					if D[i, j] < MIN_DISTANCE:
						violate.add(i)
						violate.add(j)

		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			if i in violate:
				color = (0, 0, 255)

			cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
			cv2.circle(img, (cX, cY), 5, color, 1)

		txt = "Social Distancing Violations Detected: {}".format(len(violate))
		cv2.putText(img, txt, (10, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)


    frame = cv2.flip(img, 0)
    out.write(frame)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
