import cv2
import numpy as np
import sys
import glob

import time
import torch

class YoloDetector():

    def __init__(self):
        #Using yolov5s for our purposes of object detection, you may use a larger model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', self.device)
    
    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2

        frame = frame[...,::-1]

        results = self.model(frame, size=640)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def update_boxes(self, results, frame, height, width, confidence=0.3, plot_on_img = True):

        labels, cord = results

        detections = []

        n = len(labels)
        x_shape, y_shape = width, height
        for i in range(n):
            row = cord[i]
            if row[4]>=confidence:
              
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                #In this demonstration, we will only be detecting persons. You can add classes of your choice
                if self.class_to_label(labels[i]) == 'person':
                    x_center = x1 + (x2-x1)
                    y_center = y1 + ((y2-y1) / 2)

                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype = np.float32)
                    feature = 'person'

                    bbox = [x1, y1, int(x2-x1), int(y2-y1)]
                    detections.append((bbox, row[4].item(), 'person'))
                    if plot_on_img:
                        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)

        return frame, detections