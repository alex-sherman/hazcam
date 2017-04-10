def add(a, b):
    return [ta + tb for ta, tb in zip(a, b)]

def diff(a, b):
    return [ta - tb for ta, tb in zip(a, b)]

def dot(a, b):
    return sum([ta * tb for ta, tb in zip(a, b)])

def length(a):
    return sum(t ** 2 for t in a) ** 0.5

def norm(a):
    l = length(a)
    return [t / l for t in a]

def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c