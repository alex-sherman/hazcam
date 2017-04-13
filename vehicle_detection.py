from __future__ import print_function

import cv2
import numpy as np

EDGE_DILATION = 2
CASCADE_SRC = '../opencv_dashcam_car_detection/cascade_dir/cascade.xml'

def rectangleSimilarity(r1, r2):
    a1 = r1[2] * r1[3]
    a2 = r2[2] * r2[3]
    dx = abs(r1[0] - r2[0])
    dy = abs(r1[1] - r2[1])
    w1 = r1[2]
    h1 = r1[3]
    area_ratio = float(min(a1, a2)) / max(a1,a2)

    x_sim = float(dx) / w1
    y_sim = float(dy) / h1
    similarity = area_ratio-x_sim-y_sim
    #print(area_ratio, x_sim, y_sim, similarity)
    return similarity

class VehicleDetector(object):
    def __init__(self):
        self.img = None
        self.car_cascade = cv2.CascadeClassifier(CASCADE_SRC)
        self.detection_history = []
        self.last_rects = []

    def run_step(self, img):
        self.img = img

        self.detection_history.append(self.last_rects)
        if(len(self.detection_history) > 5):
            self.detection_history = self.detection_history[1:]

        print("new frame", self.detection_history)
    
    def draw_frame(self, debug, vis, thrs1, thrs2):
        height, width, c = vis.shape
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.2, thrs1)
        self.last_rects = cars
        current_rects = []

        for frame in self.detection_history[::-1]:
            for rect in frame:
                if(len(current_rects) != 0):
                    match = max([(rectangleSimilarity(r1[0], rect), index) for index,r1 in enumerate(current_rects)])
                    if(match[0] > 0.7):
                        current_rects[index][1] += 1
                    else:
                        current_rects.append([rect,1])
                else:
                    current_rects.append([rect,1])

        
        for (x,y,w,h) in cars:
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1) 

        for ((x,y,w,h), weight) in current_rects:
            if(weight > 1):
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2) 
        return vis

