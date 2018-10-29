#!python3
#coding:utf-8
#author: leolu

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

LR_example = LogisticRegression()


class SampleTrainer:
    def __init__(self, n_col = 7, n_model = 10, model = None):
        self.n_sample = n_col
        self.n_model = n_model
        self.model = model

    def random_matrix(self, X):
        cols = X.shape[1]

        selections = []
        for i in range(self.n_model):
            selections.append(np.random.choice(list(range(cols)), 
                                              size = self.n_sample, replace = False))
        return np.array(selections)

    def fit(self, X, y):
        self.selections = self.random_matrix(X)
        self.models_ = []

        for index in self.selections:
            tmp = X[:,index]
            lbl = y

            instance = clone(self.model)
            instance.fit(tmp, lbl)
            self.models_.append(instance)

        return self

    def predict(self, X):
        pass



class SampleClassifier(SampleTrainer):

    def predict(self, X):

        tmp = []
        for i in range(self.n_model):
            tmp.append(self.models_[i].predict(X[:, self.selections[i]]))

        return np.column_stack(tmp).sum(axis=1) / self.n_model


class SampleRegressioner(SampleTrainer):

    def predict(self, X):

        tmp = []
        for i in range(self.n_model):
            tmp.append(self.models_[i].predict(X[:, self.selections[i]]))

        return np.column_stack(tmp).mean(axis=1)