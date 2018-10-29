#!python3
# -*- coding:utf-8 -*-
# auther : leolu
from .base import RandomDecisionTree

class RandomDecisionTreeRegressor(RandomDecisionTree):
    def __init__(self):
        pass
        
    def predict(self, X):
        return [np.mean(self.traverse(self.mytree, x)) for x in X]