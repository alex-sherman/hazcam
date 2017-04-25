from __future__ import print_function

import cv2
import numpy as np

EDGE_DILATION = 2
MAX_WEIGHT = 7
WEIGHT_THRESH = 4
CASCADE_SRC = 'cascade.xml'

def rectangleSimilarity(r1, r2):
    a1 = r1[2] * r1[3]
    a2 = r2[2] * r2[3]
    
    left_max = max(r1[0], r2[0])
    right_min = min(r1[0]+ r1[2], r2[0] + r2[2])
    top_min = min(r1[1]+r1[3], r2[1]+r2[3])
    bottom_max = max(r1[1], r2[1])
    x_overlap = right_min - left_max
    y_overlap = top_min - bottom_max


    if x_overlap <= 0 or y_overlap <= 0:
        return 0

    area = x_overlap * y_overlap

    similarity = area / ((a1+a2)/2.0)
    #print("Overlap area:", area, "Total Area:", (a1+a2)/2.0, "Similarity: ", similarity)

    #print(area_ratio, x_sim, y_sim, similarity)
    return similarity

def rectAverage(r_new, r_old, alpha):
    av = tuple(map(lambda x: int(x[0] * alpha + x[1] * (1-alpha)), zip(r_new, r_old)))
    return av

class VehicleDetector(object):
    def __init__(self):
        self.img = None
        self.car_cascade = cv2.CascadeClassifier(CASCADE_SRC)
        self.prev_rects = []
        self.latest_rects = []
        self.latest_filtered_rects = []

    def run_step(self, img):
        self.img = img
        self.prev_rects = self.latest_rects

    
    def draw_frame(self, debug, vis, thrs1 = 5):
        height, width, c = vis.shape
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.2, thrs1)

        searchRects = list(cars)
        self.latest_rects = []

        for i,[r1, weight] in enumerate(self.prev_rects):
            match = max([(0,0)]+[(rectangleSimilarity(r1, r2), j) for j,r2 in enumerate(searchRects)])
            if(match[0] > 0.2):
                if weight < MAX_WEIGHT:
                    weight += 1
                avRect = rectAverage(searchRects[match[1]], r1, 0.7)
                self.latest_rects.append([avRect, weight])
                #remove r2 from current frame search space
                del searchRects[match[1]]
            else:
                #if no match was found in the current frame, roll forward with weight-1
                #otherwise do not roll forward
                if weight > 1:
                    self.latest_rects.append([r1, weight-1])

        #Add unmatched rects from this frame to latest with weight 1
        self.latest_rects += [[rect, 1] for rect in searchRects]

        # for (x,y,w,h) in cars:
        #     cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1) 

        self.latest_filtered_rects = [r for r in self.latest_rects if r[1] > WEIGHT_THRESH]

        for [(x,y,w,h), weight] in self.latest_filtered_rects:
            #print(weight)
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2) 
        return vis

