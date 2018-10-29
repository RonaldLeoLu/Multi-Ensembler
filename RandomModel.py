#! python3
# -*- coding: utf-8 -*-
# @leolu
import numpy as np
from functools import reduce
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class RandomBase(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self, select_num=5):
        self.num = select_num

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.cols = []
        while len(self.cols) < (self.X.shape[1] / 2):
            k = np.random.randint(X.shape[1])
            if k not in self.cols:
                self.cols.append(k)

        print('Cols:')
        print(self.cols)

        #print(self.X[:,-1])

        return self

    def predict_one(self, X):
        pass

    def predict(self, X):
        return np.array([self.predict_one(xx) for xx in X])


class RandomClassifier(RandomBase):

    def predict_one(self, X):
        lbl_list = []
        for i in self.cols:
            #print('i: %s'%i)
            lbl_index = np.where(self.X[:,i] == X[i])[0]
            #print(lbl_index)

            if len(lbl_index) == 0:
                #print('Doing continuous calc..')
                fea = self.X[:,i]
                sigma = np.sqrt(np.var(fea))

                condition = (fea < (X[i] + sigma)) & (fea > (X[i] - sigma))
                lbl_index = np.where(condition)[0]

            if len(lbl_index) == 0:
                print('Cannot find in one sigma..')
                lbl_list.append(np.bincount(self.y.astype('int'))[-1] / len(self.y))
            else:
                c1 = np.bincount((self.y[lbl_index]).astype('int'))
                if len(c1) == 0:
                    print('cols : {}'.format(self.X.shape[1]))
                    print('Error in {}..'.format(i))
                    print('index length is {}..'.format(len(lbl_index)))
                    print(lbl_index)
                
                if len(c1) == 1:
                    lbl_list.append(0)
                else:
                    lbl_list.append(c1[-1] / np.sum(c1))

        return reduce(lambda x,y : x*y, lbl_list)


class RandomRegressor(RandomBase):
    
    def predict_one(self, X):
        lbl_list = []
        for i in self.cols:
            lbl_index = np.where(self.X[:,i] == X[i])[0]

            if len(lbl_index) == 0:
                return np.mean(self.y)

            lbl_list.append(np.mean(self.y[lbl_index]))

        return np.mean(lbl_list)