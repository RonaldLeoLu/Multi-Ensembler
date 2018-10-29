#!python3
# -*- coding:utf-8 -*-
# auther : leolu
from .base import RandomDecisionTree
import numpy as np

class RandomDecisionTreeClassifier(RandomDecisionTree):
    def __init__(self):
        pass

    def predict(self, X):
        return [self.majorCnt(self.traverse(self.mytree, x)) for x in X]

    def predict_prob(self, X):
        return [self.prob(self.traverse(self.mytree, x)) for x in X]

    def majorCnt(self, l):
        return np.argmax(np.bincount(l))

    def prob(self, l):
        lenth = np.max(self.y)
        l.append(lenth)
        prob_l = list(np.bincount(l) /  (len(l) - 1))
        prob_l[-1] = prob_l[-1] - 1 / (len(l) - 1)
        return [prob_l[i] for i in range(lenth + 1)]

