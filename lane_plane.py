from vector_math import *

# Known 2D coordinates of our rectangle
i0 = (318, 247)
i1 = (326, 312)
i2 = (418, 241)
i3 = (452, 303)

# 3D coordinates corresponding to i0, i1, i2, i3
r0 = (0, 0, 0)
r1 = (0, 0, 1)
r2 = (1, 0, 0)
r3 = (1, 0, 1)

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
    return dx * dx + dy * dy

def evaluate(mat):
    c0 = project(r0, mat)
    c1 = project(r1, mat)
    c2 = project(r2, mat)
    c3 = project(r3, mat)
    return norm2(i0, c0) + norm2(i1, c1) + norm2(i2, c2) + norm2(i3, c3)

epsilon = 0.0001
def perturb(mat, amount, est):
    from copy import deepcopy
    from random import randrange, choice, uniform
    mat2 = deepcopy(mat)
    i, j = (choice([0, 1, 3]), choice([0, 2, 3]))
    mat2[i][j] += epsilon
    est2 = evaluate(mat2)
    de = (est2 - est) / epsilon
    mat2[i][j] -= epsilon + amount * de
    return mat2, est

def approximate(mat, amount, n=400):
    est = evaluate(mat)
    for i in xrange(n):
        mat, est = perturb(mat, amount, est)

    return mat, est

est = 0
for i in range(1, 10)[::-1]:
    mat, est = approximate(mat, 0.5, n = 100)
    mat, est = approximate(mat, 0.01, n = 300)
    #mat, est = approximate(mat, .01)

print(mat, est)