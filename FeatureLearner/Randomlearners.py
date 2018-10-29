#! python3
# -*- coding:utf-8 -*-

import numpy as np
#from collections import Counter
from random import choice

class RandomDecisionTree:
    def __init__(self):
        pass
        
    def fit(self, X, y):
        self.tmp = np.column_stack([X,y])
        
        self.tree = self.buildTree(self.tmp)
        
        return self
        
    def predict(self, X):
        return [self.traverse(self.tree, x) for x in X]
        
    def buildTree(self, dataset):
        #print('Current dataset has {} columns.'.format(dataset.shape[1]))
        #print(dataset)
        tmp_ds = dataset[:]
        classlist = dataset[:,-1].tolist()
        #print(classlist)
        if len(classlist) == classlist.count(classlist[0]):
            return classlist[0]
            
        if dataset.shape[1] == 1:
            return self.majorCnt(classlist)
            
        cols = list(range(dataset.shape[1] - 1))
        col = choice(cols)
        #print('Random choose column {} to split.'.format(col))
        cols.remove(col)
        
        myTree = {col:{}}
        
        tmp_ds = np.delete(tmp_ds, col, 1)
        
        for value in set(dataset[:, col]):
            #print('Now is the loop based on {} columns'.format(dataset.shape[1]))
            tmp_low = tmp_ds[np.where(dataset[:,col] == value)[0],:]
            #print('Now the new of origin dataset is based on value: {}'.format(value))
            #print(tmp_low)
            myTree[col][value] = self.buildTree(tmp_low)
            
        return myTree
            
    def majorCnt(self, classlist):
        return np.argmax(np.bincount(classlist))
            
    def traverse(self, mytree, x):
        #print(mytree)
        print(x)
        #print(type(mytree))
        if type(mytree) != dict:
            return mytree
        col = list(mytree.keys())[0]
        #print(col)
        value = x[col]
        print('col:{},value:{}'.format(col,value))
        x_new = np.delete(x, col)
        if value not in list(mytree[col].keys()):
            v = list(mytree[col].keys())[0]
            return self.traverse(mytree[col][v], x_new)
        return self.traverse(mytree[col][value], x_new)