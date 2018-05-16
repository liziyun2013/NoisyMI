import keras.datasets.mnist
from collections import namedtuple
import numpy as np
import scipy.io as sio  
def get_mnist(trainN=None, testN=None):

    K = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    xtrain = np.reshape(x_train, [x_train.shape[0], -1]).astype("float32") / 255.
    xtest = np.reshape(x_test, [x_test.shape[0], -1]).astype("float32") / 255.
    xtrain = xtrain * 2.0 - 1.0
    xtest = xtest * 2.0 - 1.0

    if trainN is not None:
        xtrain = xtrain[0:trainN]
        ytrain = y_train[0:trainN]
    else:
        ytrain=y_train

    if testN is not None:
        xtest = xtest[0:testN]
        ytest = y_test[0:testN]
    else:
        ytest = y_test

    ytrain_onehot = keras.utils.np_utils.to_categorical(ytrain, K)
    ytest_onehot = keras.utils.np_utils.to_categorical(ytest, K)

    Dataset = namedtuple('Dataset', ["X", "Y_oh", "y", "K"])
    trn = Dataset(xtrain, ytrain_onehot, ytrain, K)
    tst = Dataset(xtest, ytest_onehot, ytest, K)

    return trn, tst

#trn, tst = get_mnist()

def get_full_data():
    return trn.X, trn.Y_oh

def get_full_data3():
    d = sio.loadmat('/Users/ziyun/Documents/Research/Information Theory on Deep Learning/NoisyMI3/NoisyMI2/data/var_u.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    trn = C()
    trn.X = F
    trn.Y_oh = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    return trn

trn = get_full_data3()

def get_full_data2():
    #d = sio.loadmat('/home/ms17/ziyunli4/NoisyMI2/NoisyMI2/data/var_u.mat')
    d = sio.loadmat('/Users/ziyun/Documents/Research/Information Theory on Deep Learning/NoisyMI3/NoisyMI2/data/var_u.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    trn = C()
    trn.X = F
    trn.Y_oh = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    return trn.X, trn.Y_oh

def get_subsample_data(samp_num):
    ind = [i for i in range(len(trn.X))]
    np.random.shuffle(ind)
    ind = ind[0:samp_num]
    return trn.X[ind], trn.Y_oh[ind]

def get_train_data():
    train_data = sio.loadmat('train.mat')

def get_test_data():
    test_data = sio.loadmat('test.mat')

