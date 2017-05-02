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
IMAGE_BOTTOM = 570
VANISHING_HEIGHT = 150
THRESH = 4000
FRAME_REPEAT = 0

def drawText(vis, text, position, scale, thickness, padding = 2, color = (255, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    size = (size[0] + padding * 2, -size[1] - padding * 2)
    cv2.rectangle(vis, tuple(diff(position, (padding, -padding * 2))), tuple(add(position, size)), (0,0,0), thickness = -1)
    cv2.putText(vis, text, position, font, scale, color, thickness, bottomLeftOrigin = False)

if __name__ == '__main__':
    print(__doc__)

    fn = sys.argv[1]
    if len(sys.argv) > 2:
        THRESH = int(sys.argv[2])
    if len(sys.argv) > 3:
        IMAGE_START = int(sys.argv[3])
    if len(sys.argv) > 4:
        IMAGE_BOTTOM = int(sys.argv[4])
    if len(sys.argv) > 5:
        VANISHING_HEIGHT = int(sys.argv[5])
    if len(sys.argv) > 6:
        FRAME_REPEAT = int(sys.argv[6])

    def nothing(*arg):
        pass

    cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', THRESH, 10000, nothing)
    cv2.createTrackbar('thrs2', 'edge', THRESH, 10000, nothing)
    cv2.createTrackbar('debug', 'edge', 0, 31, nothing)

    cap = cv2.VideoCapture(fn)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ld = LaneDetector(VANISHING_HEIGHT)
    vd = VehicleDetector()
    lp = None

    paused = True
    step = True
    cur_speed = 0
    draw_overlay = True
    frame = 1
    while True:
        if frame == FRAME_REPEAT:
            frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ld = LaneDetector(VANISHING_HEIGHT)
            lp = LanePlane(img.shape[1], img.shape[0])
            with open("mat.json", 'r') as f:
                lp.mat = json.load(f)
        if not paused or step:
            flag, img = cap.read()
            if img == None: break
            width = img.shape[1]
            yslice = slice(IMAGE_START, IMAGE_BOTTOM) if IMAGE_BOTTOM > IMAGE_START else slice(IMAGE_START, IMAGE_BOTTOM, -1)
            img = img[yslice, :]
            frame += 1
            if lp == None:
                lp = LanePlane(img.shape[1], img.shape[0])
                with open("mat.json", 'r') as f:
                    lp.mat = json.load(f)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        debug = cv2.getTrackbarPos('debug', 'edge')

        if not paused or step:
            ld.run_step(img, thrs1, thrs2, debug)
            vd.run_step(img)
            if ld.left_line != [[0,0], [0,0]] and ld.right_line != [[0,0], [0,0]]:
                lp.endpoint_iterate(ld.left_line, ld.right_line)
                if len(ld.depth_pairs) > 0:
                    delta_depths = sorted(map(lambda p: p[2], filter(lambda d: abs(d[0]) < 0.01 and d[2] > 0.001, [lp.pair_3d_delta(dp[1], dp[0]) for dp in ld.depth_pairs])))
                    if len(delta_depths) > 4:
                        est_speed = sum(delta_depths[len(delta_depths) / 2 - 1:len(delta_depths) / 2 + 2]) / 3.
                        ratio = abs(cur_speed - est_speed) / cur_speed if cur_speed != 0 else 0
                        if ratio > 0.9 and ratio < 1.1:
                            cur_speed = est_speed
                        else:
                            cur_speed = cur_speed * 0.5 + est_speed * 0.5
        vis = img.copy()
        drawText(vis,"{:2.4f}".format(cur_speed * 10), (img.shape[1] // 2, img.shape[0] - 5), 1, 2)
        vis = ld.draw_frame(debug, vis)
        if draw_overlay:
            vis = ld.draw_overlay(vis)
            vis = vd.draw_frame(debug, vis)
            for rect in vd.latest_filtered_rects:
                bottom_middle = (rect[0][0] + rect[0][2] / 2, rect[0][1] + rect[0][3])
                p3d = lp.get_plane_point(*bottom_middle)
                same_lane = p3d[0] > -1 and p3d[0] < 1
                depth = p3d[2]
                time = depth / cur_speed / fps
                if time < 2:
                    if time < 0.5 and same_lane:
                        drawText(vis, "{:2.2f}s".format(time), tuple(rect[0][:2]), 1, 2, color = (0,0,255))
                    elif time < 0.9 and same_lane or time < 0.2:
                        drawText(vis, "{:2.2f}s".format(time), tuple(rect[0][:2]), 1, 2, color = (0, 255, 255))
                    else:
                        drawText(vis, "{:2.2f}s".format(time), tuple(rect[0][:2]), 1, 2)
            if lp != None:
                lp.draw(vis, (0,255,0))
        step = False
        cv2.imshow('edge', vis)
        ch = cv2.waitKey(1)
        if ch == ord('t'):
            #lp.mat = deepcopy(IDENTITY)
            lp.endpoint_iterate(ld.left_line, ld.right_line)
            with open("mat.json", 'w') as f:
                json.dump(lp.mat, f)
        if ch == ord('o'):
            draw_overlay = not draw_overlay
        if ch == 13:
            step = True
        if ch == 32:
            paused = not paused
        if ch == 27:
            break
    cv2.destroyAllWindows()
