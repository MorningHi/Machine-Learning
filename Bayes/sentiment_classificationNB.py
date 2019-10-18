#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/8
@Author  : Rezero

用朴素贝叶斯方法进行情绪分类，数据集：sentiment labelled sentences
'''


import os
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def loadDataSet(path):
    fileList = os.listdir(path)
    dataSet = []
    labels = []
    for file in fileList:
        with open(path + file) as fr:
            for line in fr.readlines():
                dataSet.append(line.split('\t')[0])
                labels.append(int(line.strip().split('\t')[1]))
    return dataSet, labels

if __name__ == "__main__":
    dataSet, labels = loadDataSet('sentences/')
    trainX, testX, trainY, testY = train_test_split(dataSet, labels, test_size=0.15, random_state=20)   # 划分训练集和测试机

    # 文本特征向量化
    # vec = CountVectorizer()
    # trainX = vec.fit_transform(trainX)
    # testX = vec.transform(testX)

    tvec = TfidfVectorizer()
    trainX = tvec.fit_transform(trainX)
    testX = tvec.transform(testX)

    mnb = MultinomialNB()     # 多项式朴素贝叶斯
    mnb.fit(trainX, trainY)

    predict_Y = mnb.predict(testX)
    error = np.sum(np.abs(predict_Y - testY)) / len(predict_Y)

    print(error)
    print(classification_report(testY, predict_Y))
