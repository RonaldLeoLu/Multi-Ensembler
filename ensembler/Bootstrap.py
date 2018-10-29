#! python3
# -*- coding:utf-8 -*-
import numpy as np

def Bootstrap(data, label, weight=None, N=None, factor=None):
    if weight is None:
        weight = [1/data.shape[0]]*data.shape[0]
    if ((N is not None) & (factor is not None)):
        raise Exception('You should give only one of \'N\' and \'factor\'.')
    if N is not None:
        num = N
    if factor is not None:
        num = round(factor * data.shape[0])
    
    if weight is None:
        idxes = [np.random.choice(range(data.shape[0])) for i in range(num)]
        
    idxes = [np.random.choice(range(data.shape[0]),p=weight) for i in range(num)]
    
    return data[idxes], label[idxes]