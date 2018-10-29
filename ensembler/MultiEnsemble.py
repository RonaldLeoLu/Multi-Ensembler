#!python3
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.base import clone
from .Bootstrap import Bootstrap
from .MajorVoting import MajorVoting

class MultiEnsemble:
    def __init__(self, models, meta_models, weights, n=5):
        self.models = models
        self.meta_models = meta_models
        self.models_ = []
        self.meta_models_ = []
        self.weights = weights
        self.cnt = n
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        second_input = self.layer1(X,y)
        
        final_output = self.layer2(second_input,y)
        
        return self
        
    def predict(self, X):
        if len(self.models) > 10:
            mtd = 'avg'
        else:
            mtd = 'stack'
            
        if mtd == 'avg':
            mid = np.column_stack([np.sum([np.array(model.predict_proba(X)) for model in models],axis=0) / self.cnt for models in self.models_])
        if mtd == 'stack':
            mid = np.column_stack([np.column_stack([np.array(model.predict_proba(X)) for model in models]) for models in self.models_])
        #print('predict mid : {}'.format(mid.shape))    
        self.cen = mid
        output = np.sum([np.array(model.predict_proba(mid)) for model in self.meta_models_],axis=0) / len(self.meta_models_)
        
        return output.argmax(axis=1)
        
    def layer1(self, X, y):
        self.models_ = [list() for model in self.models]
        
        if len(self.models) > 10:
            mtd = 'avg'
        else:
            mtd = 'stack'
            
        for idx,model in enumerate(self.models):
            for i in range(self.cnt):
                xtmp,ytmp = Bootstrap(X, y, weight=self.weights, factor=.7)
                instance = clone(model)
                instance.fit(xtmp,ytmp)
                #print('xtmp:{}'.format(xtmp.shape))
                self.models_[idx].append(instance)
                
        if mtd == 'avg':
            #output = np.zeros((X.shape[0],len(self.models)))
            
            output = np.column_stack(
            [np.sum([np.array(model.predict_proba(self.X)) for model in models],axis=1) / self.cnt for models in self.models_])
                    
        if mtd == 'stack':
            #output = np.zeros((X.shape[0],len(set(y))*len(self.models)))
            
            output = np.column_stack(
            [np.column_stack([np.array(model.predict_proba(self.X)) for model in models]) for models in self.models_])
        #print('fit layer1 output : {}'.format(output.shape))
        return output
        
    def layer2(self, X, y):
        
        for meta_model in self.meta_models:
            instance1 = clone(meta_model)
            instance1.fit(X,y)
            self.meta_models_.append(instance1)
        
        return np.column_stack([np.array(meta.predict(X)) for meta in self.meta_models_])
        
    def output(self):
        return np.column_stack([np.array(model.predict_proba(self.cen)) for model in self.meta_models_])