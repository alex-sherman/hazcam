from vector_math import *
import math
import cv2
import numpy as np
import numpy.linalg as lin
import time


# The squared distance between two points a and b
def norm2(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return (dx * dx + dy * dy + dz * dz * 100)

class LanePlane(object):
    epsilon = 0.00001
    def __init__(self):
        self.mat = [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
        self.error1 = float('inf')
        self.error2 = float('inf')

    def project(self, p, mat = None):
        mat = mat or self.mat
        x, y, z = self.projectRaw(p, mat)
        return (WIDTH * (x + 1) / 2., HEIGHT * (1 - (y + 1) / 2.), z)

    def project2d(self, p):
        x, y, z = self.project(p, self.mat)
        return (x, y)

    def projectRaw(self, p, mat):
        x = mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2] * p[2] + mat[0][3] * 1
        y = mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2] * p[2] + mat[1][3] * 1
        z = mat[2][0] * p[0] + mat[2][1] * p[1] + mat[2][2] * p[2] + mat[2][3] * 1
        w = mat[3][0] * p[0] + mat[3][1] * p[1] + mat[3][2] * p[2] + mat[3][3] * 1
        return (x / w, y / w, z / w)

    def unproject(self, p):
        inv_mat = lin.inv(self.mat)
        p3 = [p[0] * 2. / WIDTH - 1, (p[1] / HEIGHT - 1) * -2. - 1, p[2]]
        return self.projectRaw(p3, inv_mat)

    def get_plane_point(self, x, y):
        a = self.project((0,0,0))
        b = self.project((1,0,0))
        c = self.project((1,0,1))
        n = norm(cross(diff(a, b), diff(c, b)))
        if n[2] == 0:
            z = 0
        else:
            d = dot(n, a)
            z = (d - dot(n[:2], [x, y]) / n[2])
        output = plane.unproject([x, y, z])
        return (output[0], 0, output[2])

    def evaluate(self, mat, constraints):
        return sum([norm2(c[0], self.project(c[1], mat)) for c in constraints])

    def perturb(self, constraints, amount, est):
        from copy import deepcopy
        from random import randrange, choice, uniform
        mat2 = deepcopy(self.mat)
        i, j = choice([(i, j) for i in [0, 1, 3] for j in [0, 2, 3]])
        mat2[i][j] += LanePlane.epsilon
        est2 = self.evaluate(mat2, constraints)
        if(LanePlane.epsilon == 0 or est2 == est): return self.mat, est
        de = (est2 - est) / LanePlane.epsilon
        mat2[i][j] -= LanePlane.epsilon + uniform(0, amount) * de / abs(de)
        est2 = self.evaluate(mat2, constraints)
        return mat2, est2

    def approximate(self, constraints, amount = 0.01, n=100):
        est = self.evaluate(self.mat, constraints)
        if(amount > 0 and n > 0):
            for i in xrange(n):
                mat2, est2 = self.perturb(constraints, amount, est)
                if est2 < est:
                    self.mat = mat2
                    est = est2
        return est

    def pair_constraint(self, cur_point, past_point, x3d):
        p3d = self.get_plane_point(*cur_point)
        screen_point = self.project(p3d)
        past_p3d = (x3d, 0, p3d[2] + 1)
        return ((past_point[0], past_point[1], screen_point[2] + 1), past_p3d)

    def endpoint_iterate(self, left_eps, right_eps, supports):
        constraints = supports
        self.error1 = self.evaluate(self.mat, constraints)
        amount1 = 0.01 * (1 - max(0, min(1, (1 - self.error1 / 2000))))
        number1 = 20 * (1 - max(0, min(1, (1 - self.error1 / 2000))))
        self.error1 = self.approximate(constraints, n = int(number1), amount = amount1)
        constraints += [plane.pair_constraint(ep[0][0], ep[1][0], 0) for ep in left_eps]
        constraints += [plane.pair_constraint(ep[0][1], ep[1][1], 0) for ep in left_eps]
        constraints += [plane.pair_constraint(ep[0][0], ep[1][0], 1) for ep in right_eps]
        constraints += [plane.pair_constraint(ep[0][1], ep[1][1], 1) for ep in right_eps]
        self.error2 = self.evaluate(self.mat, constraints)
        amount2 = 0.01 * (1 - max(0, min(1, (1 - self.error2 / 10000))))
        number2 = 200 * (1 - max(0, min(1, (1 - self.error2 / 10000))))
        self.error2 = self.approximate(constraints, n = int(number2), amount = amount2)


def on_click(plane, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(plane.get_plane_point(x, y))

if __name__ == '__main__':

    eps = [[[[398, 326], [501, 240], [-103, 86]], [[434, 292], [512, 228], [-78, 64]]], [[[852, 307], [751, 238], [101, 69]], [[815, 284], [737, 227], [78, 57]]], [[[550, 188], [562, 177], [-12, 11]], [[554, 185], [564, 176], [-10, 9]]], [[[680, 184], [666, 174], [14, 10]], [[676, 182], [664, 173], [12, 9]]], [[[649, 161], [645, 158], [4, 3]], [[648, 161], [644, 158], [4, 3]]], [[[580, 161], [580, 158], [0, 3]], [[582, 161], [581, 158], [1, 3]]], [[[593, 156], [640, 145], [-47, 11]], [[593, 156], [641, 146], [-48, 10]]]]
    left = eps[0]
    right = eps[1]
    WIDTH = 720
    HEIGHT = 1280
    # Known 2D coordinates of our rectangle
    i0 = left[0][1] + [5]
    i1 = left[0][0] + [0]
    i2 = right[0][1] + [5]
    i3 = right[0][0] + [0]
    depth_pairs = [p for ep in eps for p in zip(ep[0], ep[1])[:2]]
    print(depth_pairs)
    # 3D coordinates corresponding to i0, i1, i2, i3
    r0 = (0, 0, 5)
    r1 = (0, 0, 0)
    r2 = (1, 0, 5)
    r3 = (1, 0, 0)
    constraints = [(i0, r0), (i1, r1), (i2, r2), (i3, r3)]

    plane = LanePlane()
    error1 = float('inf')
    error2 = float('inf')
    #plane.approximate(constraints, n = 5000)
    while True:
        plane.endpoint_iterate(eps[0:4:2], eps[1:4:2], [(i1, r1), (i3, r3)])
        vis = np.zeros((370, 1280, 3), np.uint8)
        for pair in eps:
            if len(pair) > 0:
                cv2.line(vis, tuple(pair[0][1]), tuple(add(pair[0][2], pair[0][1])), (255, 0, 255), 1)
                cv2.line(vis, tuple(pair[0][0]), tuple(pair[1][0]), (255, 0, 0), 2)
                cv2.line(vis, tuple(pair[0][1]), tuple(pair[1][1]), (255, 0, 0), 2)
                cv2.circle(vis, tuple(pair[0][0]), 5, (255, 255, 0))
                cv2.circle(vis, tuple(pair[0][1]), 5, (255, 0, 255))
        zs = [z for z in range(0, 20)]
        for z1, z2 in zip(zs[:-1], zs[1:]):
            cv2.line(vis, tuple(map(int, plane.project2d((0,0,z1)))), tuple(map(int, plane.project2d((1,0,z1)))), (100, 100, 100), 1)

        cv2.imshow('herp', vis)
        cv2.setMouseCallback("herp", lambda *args: on_click(plane, *args))
        ch = cv2.waitKey(5)
        if ch == 27:
            break