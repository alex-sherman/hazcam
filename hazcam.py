#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np


# built-in module
import sys
import math

DILATE_COUNT = 3
HEIGHT_TARGET = 12 + DILATE_COUNT * 2
HEIGHT_THRESH = 10
MIN_LINE_SLOPE = 0.5
MAX_LINE_SLOPE = 5

def filter_lines(lines):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in matplotlib, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    output = []
    for x1,y1,x2,y2 in lines[:, 0]:
        if x1 == x2: continue
        m = (float(y2) - y1) / (x2 - x1)
        if abs(m) < MIN_LINE_SLOPE or abs(m) > MAX_LINE_SLOPE: continue
        output.append([x1, y1, x2, y2, m])
    
    return output

def find_lane_markers(image):
    levels = 10
    image = image.copy()
    for _ in range(DILATE_COUNT):
        image = cv2.dilate(image, None)
    
    image /= 10
    _, contours0, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = [cv2.approxPolyDP(cnt, 8, True) for cnt in contours0]
    contours = [cv2.convexHull(cnt, 30, True) for cnt in contours0]
    boxes = [cv2.minAreaRect(cnt) for cnt in contours0]
    boxes = [box for box in boxes if abs(box[1][1] - HEIGHT_TARGET) < HEIGHT_THRESH or abs(box[1][0] - HEIGHT_TARGET) < HEIGHT_THRESH]
    return boxes

def draw_lane_markers(boxes, vis):
    draw_boxes = [np.int0(cv2.boxPoints(box)) for box in boxes]
    cv2.drawContours(vis, draw_boxes, -1, (255, 255, 0), 2)
MASK_WEIGHT = 0.05
MASK_THRESH = 40
MASK_WIDTH = 10
if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 2500, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 2500, 5000, nothing)
    cv2.createTrackbar('angle res', 'edge', 180, 360, nothing)
    cv2.createTrackbar('thrs4', 'edge', 20, 50, nothing)

    cap = cv2.VideoCapture(fn)
    paused = True
    segment_history = []
    previous_mask = None
    step = True
    while True:
        if not paused or step:
            flag, img = cap.read()
            img = img[270:500, :]
            height, width, c = img.shape
            if(previous_mask == None):
                previous_mask = np.zeros((height, width, 1), np.uint8)
            current_mask = np.zeros((height, width, 1), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge') * 2
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge') * 2
        thrs4 = cv2.getTrackbarPos('thrs4', 'edge') * 2
        angle_res = cv2.getTrackbarPos('angle res', 'edge')
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        vis = img.copy()
        lines = cv2.HoughLinesP(edge, 1, np.pi / angle_res, thrs4, minLineLength = 10, maxLineGap = 200)
        if lines == None: continue
        lines = filter_lines(lines)
        if not paused or step:
            for line in lines:
                x1,y1,x2,y2,m = line
                cv2.line(current_mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
            current_mask = cv2.addWeighted(current_mask, MASK_WEIGHT, previous_mask, 1 - MASK_WEIGHT, 0)
            #current_mask *= int(255.0 / current_mask.max())
            previous_mask = current_mask
            _, current_mask = cv2.threshold(current_mask, 40, 255,cv2.THRESH_BINARY)
            segment_history.append(lines)
        masked_edges = cv2.dilate(cv2.bitwise_and(edge, edge, mask = current_mask), None)
        boxes = find_lane_markers(masked_edges)
        vis = np.uint8(vis)
        vis[current_mask != 0] = (0, 255, 0)
        for line in lines:
            x1,y1,x2,y2,m = line
            cv2.line(vis, (x1,y1),(x2,y2), (0, 0, 255), 3)
        vis[masked_edges != 0] = (255, 0, 0)
        vis[edge != 0] = (255, 255, 255)
        draw_lane_markers(boxes, vis)
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
