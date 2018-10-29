import numpy as np
from random import choice

# 多数投票学习器
class MajorVoting:
    def __init__(self, weights = None):
        self.weights = weights
       
    @classmethod
    def fit_transform(cls, X):
        return [MajorVoting.majorCnt(x) for x in X]
        
    @staticmethod
    def majorCnt(value):
        return np.argmax(np.bincount(value.astype('int')))