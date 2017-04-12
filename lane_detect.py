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

def x_at_y(target_y, x1, y1, x2, y2):
    m = (float(y2) - y1) / (x2 - x1)
    b = y1 - m * x1
    return int((target_y - b) / m)
def filter_lines(lines, top, bottom, slope_sign):
    """ Takes an array of hough lines and separates them by +/- slope.
        The y-axis is inverted in matplotlib, so the calculated positive slopes will be right
        lane lines and negative slopes will be left lanes. """
    output = []
    if lines != None:
        for x1,y1,x2,y2 in lines[:, 0]:
            if x1 == x2: continue
            m = (float(y2) - y1) / (x2 - x1)
            if m * slope_sign < MIN_LINE_SLOPE or m * slope_sign > MAX_LINE_SLOPE: continue
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
    return ep

def maximize(values, predicate, default):
    if(len(values) == 0): return default()
    return max([(predicate(value), value) for value in values])[1]

def draw_lane_markers(combined, vis):
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

def combine_eps(cur, past):
    return [maximize([(a, b) for b in past], lambda p: similar(p[0], p[1]), list) for a in cur]

class LaneDetector(object):
    def __init__(self):
        self.segment_history = []
        self.masked_edges_right = None
        self.masked_edges_left = None
        self.previous_mask_left = None
        self.previous_mask_right = None
        self.left_line = [[0,0], [0,0]]
        self.right_line = [[0,0], [0,0]]
        self.boxes = [[], []]
        self.edge = None
        self.eps = [[], []]

    def update_edge_mask(self, previous_mask, previous_line, slope_sign, thrs1, thrs2, thrs4, thrs5, debug, angle_res):
        lines = cv2.HoughLinesP(self.edge, 1, np.pi / angle_res, thrs4, minLineLength = 10, maxLineGap = 200)
        lines = filter_lines(lines, VANISHING_HEIGHT, self.edge.shape[0], slope_sign)
        mask = np.zeros(self.edge.shape, np.uint8)
        for line in lines:
            x1,y1,x2,y2 = line
            cv2.line(mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
        mask = cv2.addWeighted(mask, MASK_WEIGHT, previous_mask, 1 - MASK_WEIGHT, 0)
        #self.current_mask *= int(255.0 / self.current_mask.max())
        previous_mask = mask.copy()
        _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        masked_edges = cv2.morphologyEx(cv2.bitwise_and(self.edge, self.edge, mask = mask), cv2.MORPH_CLOSE, np.array([[1] * EDGE_DILATION] *EDGE_DILATION))
        lines2 = cv2.HoughLinesP(masked_edges, 1, np.pi / angle_res, thrs5, minLineLength = 10, maxLineGap = 200)
        lines2 = filter_lines(lines2, VANISHING_HEIGHT, self.edge.shape[0], slope_sign)
        for line in lines2:
            x1,y1,x2,y2 = line
            cv2.line(mask, (x1,y1),(x2,y2), 255, MASK_WIDTH)
            previous_line[0] = add(previous_line[0], (x2,y2))
            previous_line[1] = add(previous_line[1], (x_at_y(self.edge.shape[0]/2, x1, y1, x2, y2), self.edge.shape[0]/2))
        previous_line[0] = scale(previous_line[0], 1.0 / (len(lines2) + 1))
        previous_line[1] = scale(previous_line[1], 1.0 / (len(lines2) + 1))
        return masked_edges, mask, previous_mask, previous_line

    def run_step(self, img, thrs1, thrs2, thrs4, thrs5, debug, angle_res):
        height, width, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if(self.previous_mask_left == None):
            self.previous_mask_left = np.zeros((height, width, 1), np.uint8)
        if(self.previous_mask_right == None):
            self.previous_mask_right = np.zeros((height, width, 1), np.uint8)
        self.edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        self.masked_edges_left, self.current_mask_left, self.previous_mask_left, self.left_line \
            = self.update_edge_mask(self.previous_mask_left, self.left_line, -1, thrs1, thrs2, thrs4, thrs5, debug, angle_res)
        self.masked_edges_right, self.current_mask_right, self.previous_mask_right, self.right_line \
            = self.update_edge_mask(self.previous_mask_right, self.right_line, 1, thrs1, thrs2, thrs4, thrs5, debug, angle_res)
        
        self.segment_history = self.boxes
        self.boxes = [find_lane_markers(self.masked_edges_left), find_lane_markers(self.masked_edges_right)]
        self.eps = [ep[-2:] + combine_eps(cur, past)[:2] for cur, past, ep in zip(self.boxes, self.segment_history, self.eps)]

    def draw_frame(self, debug, vis):
        #self.boxes = find_lane_markers(self.masked_edges_left)
        vis = np.uint8(vis)
        if debug & 0x10:
            vis[self.current_mask_left | self.current_mask_right != 0] = (0, 50, 0)
        #if debug & 0x8:
        #    for line in self.lines:
        #        x1,y1,x2,y2 = line
        #        cv2.line(vis, (x1,y1),(x2,y2), (0, 0, 255), MASK_WIDTH)
        #if debug & 0x4:
        #    for line in self.lines2:
        #        x1,y1,x2,y2 = line
        #        cv2.line(vis, (x1,y1),(x2,y2), (0, 255, 255), 1)
        if debug & 0x2:
            vis[self.masked_edges_left | self.masked_edges_right != 0] = (255, 0, 0)
        if debug & 0x1:
            vis[self.edge != 0] = (255, 255, 255)
        cv2.line(vis, tuple(map(int, self.left_line[0])), tuple(map(int, self.left_line[1])), (0, 0, 255), MASK_WIDTH)
        cv2.line(vis, tuple(map(int, self.right_line[0])), tuple(map(int, self.right_line[1])), (0, 0, 255), MASK_WIDTH)
        draw_lane_markers(self.eps[0], vis)
        draw_lane_markers(self.eps[1], vis)
        return vis