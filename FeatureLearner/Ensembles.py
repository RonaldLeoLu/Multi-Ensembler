#! python3
# -*- coding:utf-8 -*-
import numpy as np
from random import choice
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from skelarn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from Randomlearners import RandomDecisionTreeClassifier
from Randomlearners import MajorVoting

class StackUnit:
    def __init__(self):
        lr = LogisticRegression(solver='sag', multi_class='multinomial')
        dtc = DecisionTreeClassifier()
        rdtc = RandomDecisionTreeClassifier()
        self.models = [lr, dtc, rdtc]
        self.meta_model = MajorVoting()
    
    @classmethod
    def split_n_parts(cls, n, length):
        idx = [list() for x in range(n)]
        idx_list = list(range(length))
        while len(idx_list) != 0:
            for i in range(n):
                if len(idx_list) == 0:
                    break
                tmp_idx = choice(idx_list)
                idx[i].append(tmp_idx)
                idx_list.remove(tmp_idx)
        return idx
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        times =5
        idx = cls.split_n_parts(times, X.shape[0])
        self.idx_tr = [idx[0] + idx[1] + idx[2] + idx[3],
                       idx[0] + idx[1] + idx[2] + idx[4],
                       idx[0] + idx[1] + idx[3] + idx[4],
                       idx[0] + idx[2] + idx[3] + idx[4],
                       idx[1] + idx[2] + idx[3] + idx[4]]
        self.idx_te = [idx[4], idx[3], idx[2], idx[1], idx[0]]
        self.models_ = [list() for model in self.models]
        
        for idx,model in enumerate(self.models):
            
            for i in range(times):
                x_tr, y_tr = X[self.idx_tr[i]], y[self.idx_tr[i]]
                #x_te = X[idx_te[i]]
            
                instance = clone(model)
                instance.fit(x_tr, y_tr)
                self.models_[idx].append(instance)
                
        return self
        
    def prepred(self, X = self.X):
        prd = np.zeros((self.y.shape[0], 3))
        for idx,models in enumerate(self.models_):
            for j,model in enumerate(models):
                x_te = self.X[self.idx_te[j]]
                prd[self.idx_te[j],idx] = model.predict(x_te)
        self.pred = prd       
        return self.meta_model.fit_transform(prd) # 若换做其他模型，此处需要进行大改。
        
    def predict(self, X):
        prd = np.zeros((X.shape[0], 3))
        for idx,models in enumerate(self.models_):
            subprd = np.zeros((X.shape[0], 5))
            for j,model in enumerate(models):
                subprd[:,j] = model.predict(X)
                
            prd[:,idx] = self.meta_model.fit_transform(subprd) # 若换做其他模型，从这里开始，以下大改。
        
        self.prd = prd
        
        return self.meta_model.fit_transform(prd)
    
    def train_features(self):
        return self.pred
        
    def test_features(self):
        return self.prd
    
    def lowest_score(self):
        score = []
        for i in range(3):
            score.append(accuracy_score(self.prd[:,i], self.y))
        
        return score.index(min(score))
        
    def wrong_idx(self):
        tmp = self.pred[:,self.lowest_score()]
        
        tf = [1 if tmp[i] == self.y[i] else 0 for i in range(tmp.shape[0])]
        
        return [idx for idx,value in enumerate(tf) if value == 0]
        
    def error(self):
        return len(self.wrong_idx) / self.X.shape[0]