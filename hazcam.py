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
from lane_plane import *
import json

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
    cv2.createTrackbar('thrs4', 'edge', 35, 50, nothing)
    cv2.createTrackbar('thrs5', 'edge', 35, 100, nothing)
    cv2.createTrackbar('debug', 'edge', 0, 31, nothing)
    cv2.createTrackbar('vd1', 'edge', 5, 10, nothing)
    cv2.createTrackbar('vd2', 'edge', 4525, 10000, nothing)

    cap = cv2.VideoCapture(fn)
    ld = LaneDetector()
    vd = VehicleDetector()
    lp = None

    paused = True
    step = True

    while True:
        if not paused or step:
            flag, img = cap.read()
            if img == None: break
            img = img[IMAGE_START:IMAGE_START + IMAGE_HEIGHT, :]
            if lp == None:
                lp = LanePlane(img.shape[1], img.shape[0])
                with open("mat.json", 'r') as f:
                    lp.mat = json.load(f)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        thrs4 = cv2.getTrackbarPos('thrs4', 'edge') * 2
        thrs5 = cv2.getTrackbarPos('thrs5', 'edge') * 2
        debug = cv2.getTrackbarPos('debug', 'edge')

        vd1 = cv2.getTrackbarPos('vd1', 'edge')
        vd2 = cv2.getTrackbarPos('vd2', 'edge')

        if not paused or step:
            ld.run_step(img, thrs1, thrs2, thrs4, thrs5, debug)
            vd.run_step(img)
            lp.endpoint_iterate(ld.left_line, ld.right_line)
            if len(ld.depth_pairs) > 0:
                delta_depths = sorted(map(lambda p: p[2], filter(lambda d: abs(d[0]) < 0.01 and d[2] < -0.001, [lp.pair_3d_delta(dp[0], dp[1]) for dp in ld.depth_pairs])))
                if len(delta_depths) > 4:
                    print(sum(delta_depths[len(delta_depths) / 2 - 1:len(delta_depths) / 2 + 2]) / 3.)
                


        vis = ld.draw_frame(debug, img.copy())
        vis = vd.draw_frame(debug, vis, vd1, vd2)
        if lp != None:
            lp.draw(vis, (0,255,0))
        step = False
        cv2.imshow('edge', vis)
        ch = cv2.waitKey(1)
        if ch == 116:
            #lp.mat = deepcopy(IDENTITY)
            lp.endpoint_iterate(ld.left_line, ld.right_line)
            with open("mat.json", 'w') as f:
                json.dump(lp.mat, f)
        if ch == 13:
            step = True
        if ch == 32:
            paused = not paused
        if ch == 27:
            break
    cv2.destroyAllWindows()
