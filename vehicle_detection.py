import cv2
import numpy as np

EDGE_DILATION = 2

class VehicleDetector(object):
    def __init__(self):
        self.contours = None
        self.hierarchy = None
        self.img = None

    def run_step(self, img):
        self.img = img
    
    def draw_frame(self, debug, vis, thrs1, thrs2, vdsize):
        height, width, c = vis.shape
        imgray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        #imgray = cv2.medianBlur(imgray,3)

        #thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        thresh = cv2.Canny(imgray, thrs1, thrs2, apertureSize=5)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.array([[1] * vdsize] *vdsize))

        image, self.contours, self.hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [cv2.convexHull(cnt) for cnt in self.contours]
        #print [cv2.contourArea(contour) for contour in self.contours]
        #self.contours= [contour for contour in self.contours if cv2.contourArea(contour)>vdsize]

        mask = np.zeros((height, width, 3), np.uint8)
        mask = cv2.drawContours(mask, self.contours, -1, (0,255,0), 1)


        
        
        vis = mask
        return vis

