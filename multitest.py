import multiprocessing as mp
import numpy as np

def trans(X):
    return X.T

def trun1(X):
    Y = np.array([1,2,3,4]).T
    return X+Y

def multi():
    p=mp.Pool(4)
    X=np.array([1,1,1,1]).T
    res = [p.apply_async(model,(X,)) for model in [trans,trun1]]
    print([re.get() for re in res])

