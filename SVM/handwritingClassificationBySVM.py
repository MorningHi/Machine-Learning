#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/17
@Author  : Rezero
'''
# 使用sklearn中的SVM模块实现手写数据识别

import os
import numpy as np
from sklearn.svm import SVC

def loadData(path):
    fileList = os.listdir(path)
    Num = len(fileList)  # 样本数
    dataX = []
    labels = []
    for i in range(len(fileList)):
        fileName = fileList[i]
        clas = int(fileName.split('_')[0])  # 当前样本的类别
        labels.append(clas)
        dataX.append(img2vector(path + fileName))  # 把每个样本读取为一个向量
    return np.array(dataX), np.array(labels)

def img2vector(fileName):
    returnVect = np.zeros(1024)
    with open(fileName) as fr:
        i = 0
        for lineStr in fr:  # 按行读取文件
            for j in range(32):
                returnVect[32 * i + j] = int(lineStr[j])
            i += 1
    return returnVect


if __name__ == "__main__":
    trainingX, trainingLabels = loadData('data/trainingDigits/')
    testX, testLabels = loadData('data/testDigits/')

    model = SVC()
    model.fit(trainingX, trainingLabels)
    test_predict = model.predict(testX)

    error_count = len(np.where((test_predict - testLabels) != 0)[0])
    print("错误率：%.2f%%" % (100 * error_count / len(testLabels)))
