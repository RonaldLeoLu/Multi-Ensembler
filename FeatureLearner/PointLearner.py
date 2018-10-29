#!python3
#coding:utf-8
#author: leolu

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Lasso


class PointTrainer:
    def __init__(self, n_cols = 7, n_sample = 100, n_model = 5, model = None):
        self.n_cols = n_cols
        self.n_sample = n_sample
        self.n_model = n_model
        self.model = model

    def random_dict(self, X):
        rows = np.random.choice(list(range(X.shape[0])), size = self.n_sample, replace = True)
        cols = np.random.choice(list(range(X.shape[1])), size = self.n_cols, replace = False)

        point_local = {}
        point_local['rows'] = rows
        point_local['cols'] = cols

        return point_local

    def fit(self, X, y):
        self.point_local = self.random_dict(X)
        self.models_ = []

        tmp = []
        lbl = y[self.point_local['rows']]
        for i in self.point_local['cols']:
            tmp.append(X[self.point_local['rows'], i])

        tmp = np.column_stack(tmp)
        for i in range(self.n_model):
            instance = clone(self.model)
            instance.fit(tmp, lbl)
            self.models_.append(instance)

        return self

    def predcit(self, X):
        pass


class PointClassifier(PointTrainer):

    def predict(self, X):

        tmp = []
        for i in range(self.n_model):
            tmp.append(self.models_[i].predict(X[:, self.point_local['cols']]))

        return np.column_stack(tmp).sum(axis=1) / self.n_model


class PointRegressioner(PointTrainer):

    def predict(self, X):

        tmp = []
        for i in range(self.n_model):
            tmp.append(self.models_[i].predict(X[:, self.point_local['cols']]))

        return np.column_stack(tmp).mean(axis=1)