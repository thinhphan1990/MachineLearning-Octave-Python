from __future__ import print_function
import numpy as np
from time import time  # for comparing running time

d, N = 1000, 10000  # dimension, number of training points

X = np.random.rand(N, d)
z = np.random.rand(d)


# naively compute square distance between two vector
def dist_pp(z, x):
    d = z - x.reshape(z.shape)  # force x and z to have the same dims
    return np.sum(d * d)


# from one point to each point in a set, naive
def dist_ps_naive(z, X):
    n = X.shape[0]
    res = np.zeros((1, n))
    for i in range(n):
        res[0][i] = dist_pp(z, X[i])
    return res


# from one point to each point in a set, fast
def dist_ps_fast(z, X):
    x2 = np.sum(X * X, 1)  # square of l2 norm of each ROW of X
    z2 = np.sum(z * z)  # square of l2 norm of z
    return x2 + z2 - 2 * X.dot(z)  # z2 can be ignored


t1 = time()
D1 = dist_ps_naive(z, X)
print('naive point2set, running time:', time() - t1, 's')
t1 = time()
D2 = dist_ps_fast(z, X)
print('fast point2set , running time:', time() - t1, 's')
print('Result difference:', np.linalg.norm(D1 - D2))
