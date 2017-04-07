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
EDGE_DILATION = 20
MIN_LINE_SLOPE = 0.5
MAX_LINE_SLOPE = 5
IMAGE_START = 200
VANISHING_HEIGHT = 130
IMAGE_HEIGHT = 370

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

def similar(a, b):
    d = norm(a[2])
    diff0 = length(diff(a[0], b[0]))
    diff1 = length(diff(a[1], b[1]))

    return -(diff0 + diff1)

def find_lane_markers(image):
    levels = 10
    image = image.copy()
    
    image /= 10
    _, contours0, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = [[]] if hierarchy == None else hierarchy
    # Remove any children, h[3] == h[-1] is the parent index and is -1 if it has no parent
    contours = [cv2.convexHull(cnt) for cnt, h in zip(contours0, hierarchy[0]) if h[-1] == -1]
    ep = find_endpoints(contours)
    return contours, ep

def maximize(values, predicate, default):
    if(len(values) == 0): return default()
    return max([(predicate(value), value) for value in values])[1]

def draw_lane_markers(markers, last_lane_markers, vis):
    contours, endpoints = markers
    combined = [maximize([(a, b) for b in last_lane_markers], lambda p: similar(p[0], p[1]), list) for a in endpoints]
    #cv2.drawContours(vis, contours, -1, (200, 200, 0), 1)
    for pair in combined:
        if len(pair) > 0:
            cv2.line(vis, pair[0][1], tuple(add(pair[0][2], pair[0][1])), (255, 0, 255), 1)
            cv2.line(vis, pair[0][0],pair[1][0], (255, 0, 0), 2)
            cv2.line(vis, pair[0][1],pair[1][1], (255, 0, 0), 2)
            cv2.circle(vis, pair[0][0], 5, (255, 255, 0))
            cv2.circle(vis, pair[0][1], 5, (255, 0, 255))


def add(a, b):
    return [a[0] + b[0], a[1] + b[1]]

def diff(a, b):
    return [a[0] - b[0], a[1] - b[1]]

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def length(a):
    return (a[0] ** 2 + a[1] ** 2) ** 0.5

def norm(a):
    l = length(a)
    return [a[0] / l, a[1] / l]

def find_endpoints(hulls):
    end_points = []
    for hull in hulls:
        points = list(np.squeeze(hull, 1))
        direction = [0, 1]
        a = max([(dot(direction, p), tuple(p)) for p in points])[1]
        b = min([(dot(direction, p), tuple(p)) for p in points])[1]
        direction = (a[0] - b[0], a[1] - b[1])
        a = max([(dot(direction, p), tuple(p)) for p in points])[1]
        b = min([(dot(direction, p), tuple(p)) for p in points])[1]
        direction = (a[0] - b[0], a[1] - b[1])
        end_points.append([a, b, direction])
    return end_points


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
    cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 2000, 5000, nothing)
    cv2.createTrackbar('angle res', 'edge', 180, 360, nothing)
    cv2.createTrackbar('thrs4', 'edge', 35, 50, nothing)
    cv2.createTrackbar('thrs5', 'edge', 35, 100, nothing)
    cv2.createTrackbar('debug', 'edge', 0, 31, nothing)

    cap = cv2.VideoCapture(fn)
    paused = True
    segment_history = []
    previous_mask = None
    step = True
    boxes = ([],[])
    while True:
        if not paused or step:
            flag, img = cap.read()
            if img == None: break
            img = img[IMAGE_START:IMAGE_START + IMAGE_HEIGHT, :]
            height, width, c = img.shape
            if(previous_mask == None):
                previous_mask = np.zeros((height, width, 1), np.uint8)
            current_mask = np.zeros((height, width, 1), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge') * 2
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge') * 2
        thrs4 = cv2.getTrackbarPos('thrs4', 'edge') * 2
        thrs5 = cv2.getTrackbarPos('thrs5', 'edge') * 2
        debug = cv2.getTrackbarPos('debug', 'edge')
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
            masked_edges = cv2.morphologyEx(cv2.bitwise_and(edge, edge, mask = current_mask), cv2.MORPH_CLOSE, np.array([[1] * EDGE_DILATION] *EDGE_DILATION))
            lines2 = cv2.HoughLinesP(masked_edges, 1, np.pi / angle_res, thrs5, minLineLength = 10, maxLineGap = 200)
            lines2 = filter_lines(lines2)
            for line in lines2:
                x1,y1,x2,y2 = line
                cv2.line(current_mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
            segment_history = boxes[1]
        boxes = find_lane_markers(masked_edges)
        vis = np.uint8(vis)
        if debug & 0x10:
            vis[current_mask != 0] = (0, 50, 0)
        if debug & 0x8:
            for line in lines:
                x1,y1,x2,y2 = line
                cv2.line(vis, (x1,y1),(x2,y2), (0, 0, 255), MASK_WIDTH)
        if debug & 0x4:
            for line in lines2:
                x1,y1,x2,y2 = line
                cv2.line(vis, (x1,y1),(x2,y2), (0, 255, 255), 1)
        if debug & 0x2:
            vis[masked_edges != 0] = (255, 0, 0)
        if debug & 0x1:
            vis[edge != 0] = (255, 255, 255)
        draw_lane_markers(boxes, segment_history, vis)
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
