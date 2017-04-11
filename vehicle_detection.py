import cv2
import numpy as np

EDGE_DILATION = 2
CASCADE_SRC = './cascade.xml'

class VehicleDetector(object):
    def __init__(self):
        self.img = None
        self.car_cascade = cv2.CascadeClassifier(CASCADE_SRC)
        self.detection_history = []

    def run_step(self, img):
        self.img = img
    
    def draw_frame(self, debug, vis, thrs1, thrs2):
        height, width, c = vis.shape
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.2, thrs1)

        self.detection_history.append(cars)
        if(len(self.detection_history) > 5):
            self.detection_history = self.detection_history[1:]

        for (x,y,w,h) in cars:
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2) 
        return vis

