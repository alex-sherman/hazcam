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

DEBUG = False
EDGE_DILATION = 3
MIN_LINE_SLOPE = 0.5
MAX_LINE_SLOPE = 5
VANISHING_HEIGHT = 60
IMAGE_HEIGHT = 300

def filter_lines(lines):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in matplotlib, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    output = []
    if lines != None:
        for x1,y1,x2,y2 in lines[:, 0]:
            if x1 == x2: continue
            m = (float(y2) - y1) / (x2 - x1)
            if abs(m) < MIN_LINE_SLOPE or abs(m) > MAX_LINE_SLOPE: continue
            b = y1 - m * x1
            y1 = VANISHING_HEIGHT
            x1 = int((y1 - b) / m)
            y2 = IMAGE_HEIGHT
            x2 = int((y2 - b) / m)
            output.append([x1, y1, x2, y2])
    
    return output

def find_lane_markers(image):
    levels = 10
    image = image.copy()
    
    image /= 10
    _, contours0, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.convexHull(cnt) for cnt in contours0]
    boxes = [cv2.minAreaRect(cnt) for cnt in contours0]
    return contours

def draw_lane_markers(boxes, vis):
    cv2.drawContours(vis, boxes, -1, (255, 255, 0), 1)
MASK_WEIGHT = 0.1
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
    cv2.createTrackbar('thrs4', 'edge', 25, 50, nothing)
    cv2.createTrackbar('thrs5', 'edge', 30, 100, nothing)

    cap = cv2.VideoCapture(fn)
    paused = True
    segment_history = []
    previous_mask = None
    step = True
    while True:
        if not paused or step:
            flag, img = cap.read()
            img = img[270:270 + IMAGE_HEIGHT, :]
            height, width, c = img.shape
            if(previous_mask == None):
                previous_mask = np.zeros((height, width, 1), np.uint8)
            current_mask = np.zeros((height, width, 1), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge') * 2
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge') * 2
        thrs4 = cv2.getTrackbarPos('thrs4', 'edge') * 2
        thrs5 = cv2.getTrackbarPos('thrs5', 'edge') * 2
        angle_res = cv2.getTrackbarPos('angle res', 'edge')
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        vis = img.copy()
        lines = cv2.HoughLinesP(edge, 1, np.pi / angle_res, thrs4, minLineLength = 10, maxLineGap = 200)
        lines = filter_lines(lines)
        if not paused or step:
            for line in lines:
                x1,y1,x2,y2 = line
                cv2.line(current_mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
            current_mask = cv2.addWeighted(current_mask, MASK_WEIGHT, previous_mask, 1 - MASK_WEIGHT, 0)
            #current_mask *= int(255.0 / current_mask.max())
            previous_mask = current_mask
            _, current_mask = cv2.threshold(current_mask, 40, 255,cv2.THRESH_BINARY)
            segment_history.append(lines)
            masked_edges = cv2.dilate(cv2.bitwise_and(edge, edge, mask = current_mask), np.array([[1] * EDGE_DILATION] *EDGE_DILATION))
            lines2 = cv2.HoughLinesP(masked_edges, 1, np.pi / angle_res, thrs5, minLineLength = 10, maxLineGap = 200)
            lines2 = filter_lines(lines2)
            for line in lines2:
                x1,y1,x2,y2 = line
                cv2.line(current_mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
        boxes = find_lane_markers(masked_edges)
        vis = np.uint8(vis)
        if DEBUG:
            vis[current_mask != 0] = (0, 50, 0)
            for line in lines:
                x1,y1,x2,y2 = line
                cv2.line(vis, (x1,y1),(x2,y2), (0, 0, 255), MASK_WIDTH)
            for line in lines2:
                x1,y1,x2,y2 = line
                cv2.line(vis, (x1,y1),(x2,y2), (0, 255, 255), 1)
            vis[masked_edges != 0] = (255, 0, 0)
            vis[edge != 0] = (255, 255, 255)
        draw_lane_markers(boxes, vis)
        step = False
        cv2.imshow('edge', vis)
        cv2.imshow('edge', vis)
        ch = cv2.waitKey(5)
        if ch == 13:
            step = True
        if ch == 32:
            paused = not paused
        if ch == 27:
            break
    cv2.destroyAllWindows()
