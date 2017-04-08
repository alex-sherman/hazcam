from vector_math import *
import math
import cv2
import numpy as np

eps = [[[[398, 326], [501, 240], [-103, 86]], [[434, 292], [512, 228], [-78, 64]]], [[[852, 307], [751, 238], [101, 69]], [[815, 284], [737, 227], [78, 57]]], [[[550, 188], [562, 177], [-12, 11]], [[554, 185], [564, 176], [-10, 9]]], [[[680, 184], [666, 174], [14, 10]], [[676, 182], [664, 173], [12, 9]]], [[[649, 161], [645, 158], [4, 3]], [[648, 161], [644, 158], [4, 3]]], [[[580, 161], [580, 158], [0, 3]], [[582, 161], [581, 158], [1, 3]]], [[[593, 156], [640, 145], [-47, 11]], [[593, 156], [641, 146], [-48, 10]]]]
left = eps[0]
right = eps[1]
# Known 2D coordinates of our rectangle
i0 = left[0][1]
i1 = left[0][0]
i2 = right[0][1]
i3 = right[0][0]

print(i0, i1, i2, i3)

# 3D coordinates corresponding to i0, i1, i2, i3
r0 = (0, 0, 1)
r1 = (0, 0, 0)
r2 = (1, 0, 1)
r3 = (1, 0, 0)

mat = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]

def project(p, mat):
    x = mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2] * p[2] + mat[0][3] * 1
    y = mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2] * p[2] + mat[1][3] * 1
    w = mat[3][0] * p[0] + mat[3][1] * p[1] + mat[3][2] * p[2] + mat[3][3] * 1
    return (720 * (x / w + 1) / 2., 576 - 576 * (y / w + 1) / 2.)

# The squared distance between two points a and b
def norm2(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (dx * dx + dy * dy)

def evaluate(mat):
    c0 = project(r0, mat)
    c1 = project(r1, mat)
    c2 = project(r2, mat)
    c3 = project(r3, mat)
    return norm2(i0, c0) + norm2(i1, c1) + norm2(i2, c2) + norm2(i3, c3)

epsilon = 0.00001
def perturb(mat, amount, est):
    from copy import deepcopy
    from random import randrange, choice, uniform
    mat2 = deepcopy(mat)
    i, j = choice([(i, j) for i in [0, 1, 3] for j in [0, 2, 3]])
    #print(i, j)
    mat2[i][j] += epsilon
    est2 = evaluate(mat2)
    de = (est2 - est) / epsilon
    mat2[i][j] -= epsilon + uniform(0, amount) * de / abs(de)
    est2 = evaluate(mat2)
    return mat2, est2

def approximate(mat, amount = 0.01, n=10000):
    est = evaluate(mat)
    for i in xrange(n):
        mat2, est2 = perturb(mat, amount, est)
        if est2 < est:
            mat = mat2
            est = est2

    return mat, est

if __name__ == '__main__':
    mat, est = approximate(mat)
    print(mat, est)

    vis = np.zeros((370, 1280, 3), np.uint8)
    for pair in eps:
        if len(pair) > 0:
            cv2.line(vis, tuple(pair[0][1]), tuple(add(pair[0][2], pair[0][1])), (255, 0, 255), 1)
            cv2.line(vis, tuple(pair[0][0]), tuple(pair[1][0]), (255, 0, 0), 2)
            cv2.line(vis, tuple(pair[0][1]), tuple(pair[1][1]), (255, 0, 0), 2)
            cv2.circle(vis, tuple(pair[0][0]), 5, (255, 255, 0))
            cv2.circle(vis, tuple(pair[0][1]), 5, (255, 0, 255))
    zs = [z for z in range(0, 80)]
    for z1, z2 in zip(zs[:-1], zs[1:]):
        cv2.line(vis, tuple(map(int, project((0,0,z1), mat))), tuple(map(int, project((1,0,z1), mat))), (100, 100, 100), 1)

    cv2.imshow('herp', vis)
    while True:
#
        ch = cv2.waitKey(5)
        if ch == 27:
            break