import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles


def get_synthetic_data(type, size, size2=None, noise_frac=0.0):
    if type == 'circles':
        X, y = make_circles(n_samples=size)
    elif type == 'moons':
        X, y = make_moons(n_samples=size)
    else:
        if size2 is None:
            size = int(size/2)
            size2 = size
        c1 = np.random.multivariate_normal([1, 1], np.diag([1, 1]), size=size)
        c2 = np.random.multivariate_normal([3, 3], np.diag([1, 1]), size=size2)
        X = np.vstack((c2, c1))
        y = np.zeros(len(X))
        y[-size:] = 1
    if noise_frac > 0.0:
        n = len(y)
        nidx = np.random.choice(n, int(noise_frac * n), replace=False)
        y[nidx] = 1 - y[nidx]
    else:
        nidx = None
    return X, y, nidx
