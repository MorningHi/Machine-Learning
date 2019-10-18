#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/1
@Author  : Rezero
'''

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


def loadData(fileName):
    df = read_csv(fileName, sep='\t')
    dataSet = df.values[:, :-1]
    labels = df.values[:, -1]
    le = LabelEncoder()
    for i in range(dataSet.shape[1]):
        dataSet[:, i] = le.fit_transform(dataSet[:, i])
    print(dataSet)
    return dataSet, labels



if __name__ == "__main__":
    dataSet, labels = loadData('data/lenses.txt')
    decisionTree = tree.DecisionTreeClassifier(max_depth=4)
    decisionTree = decisionTree.fit(dataSet, labels)
    clas = decisionTree.predict([[0, 0, 1, 1]])
    print(clas)

