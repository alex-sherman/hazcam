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