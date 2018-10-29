#! python3
# -*- coding:utf-8 -*-
# @leolu

import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV,train_test_split

class StackingAverageModels(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(self,base_models,meta_model,n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        
    def fit(self,X,y):
        self.base_models_ = [list() for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds,shuffle=True,random_state=156)
        
        out_of_fold_predictions = np.zeros((X.shape[0],len(self.base_models)))
        for i,model in enumerate(self.base_models):
            for train_index,holdout_index in kfold.split(X,y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index,i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions,y)
        return self
    
    def predict(self,X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

def unit(nonlinem, linem, train, label, test):
    nonli_model = nonlinem
    linear_model = linem

    nonli_model.fit(train, label)
    linear_model.fit(train, label)

    non_train_fea = nonli_model.predict(train)
    non_test_fea = nonli_model.predict(test)

    lin_train_fea = linear_model.predict(train)
    lin_test_fea = linear_model.predict(test)

    train = np.column_stack([train, non_train_fea, lin_train_fea])
    test = np.column_stack([test, non_test_fea, lin_test_fea])

    return train, test


def acc_titanic(model):
    model.fit(train, label)
    y_pred = model.predict(test)
    return accuracy_score(ytest,y_pred)



'''
if __name__ == '__main__':
    score_list = []
    keep_train = True
    train = # #
    test = # #
    label = # #
    ytest = # #
    linearmodel = # #
    nonlinearmodel = # #
    RF = RandomForestClassifier() # RandomForestRegressor()
    layers = 0

    while keep_train:
        train, test = unit(nonlinearmodel, linearmodel, train, label, test)
        score = acc_titanic(RF)

        if abs(score - np.max(score_list)) < 1e-5:
            keep_train = False
        else:
            score_list.append(score)
            layers += 1

    print('Layers: %d'%layers)
    print('Score: %d'%score)
    '''







