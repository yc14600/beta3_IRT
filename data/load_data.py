from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.examples.tutorials.mnist import input_data

# train size + test size <= 55000 #
def load_mnist_fashion_data(train_size,test_size,dpath,noise_frac=0.,train_noise_frac=0.,class1=7,class2=9,seed=42):
    np.random.seed(seed)
    D = input_data.read_data_sets(dpath) 

    X = D.train.images
    Y = D.train.labels

    ids = (Y==class1)|(Y==class2) 
    X, y = X[ids], Y[ids]

    y[y==class1] = 0 # 
    y[y==class2] = 1 # ankle boots

    X_train = X[:train_size]
    y_train = y[:train_size]
    #print(y_train.sum()*1./len(y_train))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    #print(y_test.sum()*1./len(y_test))

    if train_noise_frac > 0.:
        n = len(y_train)
        idx = np.random.choice(n, int(noise_frac * n), replace=False)
        y_train[idx] = 1 - y_train[idx]
        #print(y_train.sum()*1./len(y_train))
        #print(idx)

    if noise_frac > 0.0:
        n = len(y_test)
        nidx = np.random.choice(n, int(noise_frac * n), replace=False)
        y_test[nidx] = 1 - y_test[nidx]
        #print(y_test.sum()*1./len(y_test))
        #print(np.sort(nidx))
    else:
        nidx = None

    return X_train,y_train,X_test,y_test,nidx