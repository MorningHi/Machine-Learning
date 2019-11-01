#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/20
@Author  : Rezero
'''
# 利用sklearn.ensemble.AdaBoostClassifier 实现分类

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

def loadDataSet(path):
    dataSet = np.loadtxt(path)
    X = dataSet[:, :-1]
    y = dataSet[:, -1]
    return X, y

if __name__ == "__main__":
    trainX, trainY = loadDataSet('data/horseColicTraining.txt')   # 加载数据集
    testX, testY = loadDataSet('data/horseColicTest.txt')

    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=4), learning_rate=0.3)
    abc.fit(trainX, trainY)
    test_predict = abc.predict(testX)
    error = len(np.where(test_predict != testY)[0]) / len(testY)
    print("错误率：%f" % error)
