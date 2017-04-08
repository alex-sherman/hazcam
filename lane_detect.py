# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np


# built-in module
import sys
import math
from vector_math import *


EDGE_DILATION = 20
MIN_LINE_SLOPE = 0.5
MAX_LINE_SLOPE = 5
VANISHING_HEIGHT = 130

MASK_WEIGHT = 0.1
MASK_THRESH = 40
MASK_WIDTH = 10

def filter_lines(lines, top, bottom):
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
            y1 = top
            x1 = int((y1 - b) / m)
            y2 = bottom
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

class LaneDetector(object):
    def __init__(self):
        self.segment_history = []
        self.masked_edges = None
        self.previous_mask = None
        self.boxes = ([],[])
        self.edge = None

    def run_step(self, img, thrs1, thrs2, thrs4, thrs5, debug, angle_res):
        height, width, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if(self.previous_mask == None):
            self.previous_mask = np.zeros((height, width, 1), np.uint8)
        self.current_mask = np.zeros((height, width, 1), np.uint8)
        
        self.edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        self.lines = cv2.HoughLinesP(self.edge, 1, np.pi / angle_res, thrs4, minLineLength = 10, maxLineGap = 200)
        self.lines = filter_lines(self.lines, VANISHING_HEIGHT, height)

        for line in self.lines:
            x1,y1,x2,y2 = line
            cv2.line(self.current_mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
        self.current_mask = cv2.addWeighted(self.current_mask, MASK_WEIGHT, self.previous_mask, 1 - MASK_WEIGHT, 0)
        #self.current_mask *= int(255.0 / self.current_mask.max())
        self.previous_mask = self.current_mask
        _, self.current_mask = cv2.threshold(self.current_mask, 40, 255,cv2.THRESH_BINARY)
        self.masked_edges = cv2.morphologyEx(cv2.bitwise_and(self.edge, self.edge, mask = self.current_mask), cv2.MORPH_CLOSE, np.array([[1] * EDGE_DILATION] *EDGE_DILATION))
        self.lines2 = cv2.HoughLinesP(self.masked_edges, 1, np.pi / angle_res, thrs5, minLineLength = 10, maxLineGap = 200)
        self.lines2 = filter_lines(self.lines2, VANISHING_HEIGHT, height)
        for line in self.lines2:
            x1,y1,x2,y2 = line
            cv2.line(self.current_mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
        self.segment_history = self.boxes[1]
        self.boxes = find_lane_markers(self.masked_edges)

    def draw_frame(self, debug, vis):
        self.boxes = find_lane_markers(self.masked_edges)
        vis = np.uint8(vis)
        if debug & 0x10:
            vis[self.current_mask != 0] = (0, 50, 0)
        if debug & 0x8:
            for line in self.lines:
                x1,y1,x2,y2 = line
                cv2.line(vis, (x1,y1),(x2,y2), (0, 0, 255), MASK_WIDTH)
        if debug & 0x4:
            for line in self.lines2:
                x1,y1,x2,y2 = line
                cv2.line(vis, (x1,y1),(x2,y2), (0, 255, 255), 1)
        if debug & 0x2:
            vis[self.masked_edges != 0] = (255, 0, 0)
        if debug & 0x1:
            vis[self.edge != 0] = (255, 255, 255)
        draw_lane_markers(self.boxes, self.segment_history, vis)
        return vis