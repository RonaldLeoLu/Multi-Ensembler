#!python3
# -*- coding:utf-8 -*-
# auther : leolu

#############
#   This .py file is written for the basic part
# of our complete random decision tree.
#   In this .py file, we will define some useful
# functions that will be applied into
# 'RandomDecisionTreeClassifier.py' and 
# 'randomDecisionTreeRegressor.py'.
#############
import numpy as np


# Basic class of Random Decision Tree
class RandomDecisionTree:
    def __init__(self):
        pass

    def fit(self, X, y):
        # X: trainset
        # y: trainlable
        # return: self
        self.y = y
        tmp = np.column_stack([X, y])

        self.mytree = self.createTree(tmp)

        return self


    def predict(self, X):
        #################
        # X: testset
        # return: prediction
        # This part is depend on its real method.
        #################
        pass

    def createTree(self, data):
        #################
        # data: make sure data[:,-1] is train label
        # return: mytree <type: dict>
        #################
        tmp_ds = data[:]
        classlist = data[:,-1].tolist()
        #print(classlist)
        # stop condition 1 
        if len(classlist) == classlist.count(classlist[0]):
            return classlist
        # stop condition 2
        if data.shape[1] == 1:
            return classlist

        # choose the split feature
        cols = list(range(data.shape[1] - 1))
        col = np.random.choice(cols)
        cols.remove(col)
        # struct my tree
        mytree = {col:{}}

        if type(tmp_ds[:,col][0]) == 'object':
            for value in set(tmp_ds[:,col]):
                tmp = tmp_ds[np.where(tmp_ds[:,col] == value)[0],:]
                tmp = np.delete(tmp, col, 1)

                mytree[col][value] = self.createTree(tmp)

        else:
            # find the random split point
            MAX = tmp_ds[:,col].max()
            MIN = tmp_ds[:,col].min()

            key = np.random.uniform(low=MIN, high=MAX)

            for eql in ['gt', 'lt']:
                value = eql + str(key)
                if eql == 'gt':
                    tmp = tmp_ds[np.where(tmp_ds[:,col] > key)[0],:]
                    if tmp.shape[0] == 0:
                        continue
                    tmp = np.delete(tmp, col, 1)
                    mytree[col][value] = self.createTree(tmp)
                else:
                    tmp = tmp_ds[np.where(tmp_ds[:,col] <= key)[0],:]
                    if tmp.shape[0] == 0:
                        continue
                    tmp = np.delete(tmp, col, 1)
                    mytree[col][value] = self.createTree(tmp)
        return mytree

    # Traverse my tree
    def traverse(self, mytree, x):
        #################
        # tree: node of the decision tree
        # x: the sample we need to classify
        # return: prediction of the sample
        #################
        if type(mytree) != dict:
            return mytree

        col = list(mytree.keys())[0]
        v = x[col]
        x_new = np.delete(x, col)

        if type(v) == 'object':
            if v not in list(mytree[col].keys()):
                v = list(mytree[col].keys())[0]
                return self.traverse(mytree[col][v], x_new)

            return self.traverse(mytree[col][v], x_new)

        else:
            jgd = float(list(mytree[col].keys())[0][2:])
            if v > jgd:
                v = 'gt' + str(jgd)
            else:
                v = 'lt' + str(jgd)
            return self.traverse(mytree[col][v], x_new)
