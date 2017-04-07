import cv2

class VehicleDetector(object):
    def __init__(self):
        self.contours = None
        self.hierarchy = None

    def run_step(self, img, thrs1, thrs2):
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,thrs1,thrs2,0)
        image, self.contours, self.hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    def draw_frame(self, debug, vis):
        imgout = cv2.drawContours(vis, self.contours, -1, (0,255,0), 1)
        return imgout

