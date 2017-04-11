#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np


# built-in module
import sys
import math

from lane_detect import LaneDetector
from vehicle_detection import VehicleDetector

IMAGE_START = 200
IMAGE_HEIGHT = 370



if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 4000, 10000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 4000, 10000, nothing)
    cv2.createTrackbar('angle res', 'edge', 180, 360, nothing)
    cv2.createTrackbar('thrs4', 'edge', 35, 50, nothing)
    cv2.createTrackbar('thrs5', 'edge', 35, 100, nothing)
    cv2.createTrackbar('debug', 'edge', 0, 31, nothing)
    cv2.createTrackbar('vd1', 'edge', 4, 10, nothing)
    cv2.createTrackbar('vd2', 'edge', 4525, 10000, nothing)

    cap = cv2.VideoCapture(fn)
    ld = LaneDetector()
    vd = VehicleDetector()

    paused = True
    step = True

    while True:
        if not paused or step:
            flag, img = cap.read()
            if img == None: break
            img = img[IMAGE_START:IMAGE_START + IMAGE_HEIGHT, :]
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        thrs4 = cv2.getTrackbarPos('thrs4', 'edge') * 2
        thrs5 = cv2.getTrackbarPos('thrs5', 'edge') * 2
        debug = cv2.getTrackbarPos('debug', 'edge')
        angle_res = cv2.getTrackbarPos('angle res', 'edge')

        vd1 = cv2.getTrackbarPos('vd1', 'edge')
        vd2 = cv2.getTrackbarPos('vd2', 'edge')

        if not paused or step:
            ld.run_step(img, thrs1, thrs2, thrs4, thrs5, debug, angle_res)
            vd.run_step(img)

        vis = ld.draw_frame(debug, img.copy())
        vis = vd.draw_frame(debug, vis, vd1, vd2)
        step = False
        cv2.imshow('edge', vis)
        ch = cv2.waitKey(5)
        if ch == 13:
            step = True
        if ch == 32:
            paused = not paused
        if ch == 27:
            break
    cv2.destroyAllWindows()
