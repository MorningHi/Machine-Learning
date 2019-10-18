#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/9/27
@Author  : Rezero
'''

import numpy as np
import os

def img2vector(fileName):
    returnVect = np.zeros((1, 1024))
    with open(fileName) as fr:
        i = 0
        for lineStr in fr:   # 按行读取文件
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
            i += 1
    return returnVect

def getDistance(testX, trainingSet):
    return np.sqrt(np.sum((trainingSet - testX)**2, axis=1))  # 欧式距离

def classify(testX, trainingSet, labels, k):
    '''
    testX: 测试样本的特征向量
    trainingSet: 训练集特征
    labels: 训练集类别标签
    k: 最邻近样本个数
    '''
    m = trainingSet.shape[0]  # 训练样本的数量
    distance = getDistance(testX, trainingSet)  # 计算测试样本到每个训练样本的距离
    nearestIndices = np.argsort(distance)[:k]  # 距离最小的k个样本的索引
    # 计算前k个样本中每一类的数量
    classCount = {}
    maxCount = 0
    nearestClass = None
    for i in range(k):
        voteIlabel = labels[nearestIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        if classCount[voteIlabel] > maxCount:   # 更新最多的计数和其对应的标签
            maxCount = classCount[voteIlabel]
            nearestClass = voteIlabel
    return nearestClass

def main():
    labels = []
    trainingFiles = os.listdir('data/trainingDigits')  # 训练集路径
    trainNum = len(trainingFiles)  # 训练集样本数
    trainingMat = np.zeros((trainNum, 1024))
    for i in range(trainNum):
        fileName = trainingFiles[i]
        clas = int(fileName.split('_')[0])   # 当前样本的类别
        labels.append(clas)
        trainingMat[i, :] = img2vector('data/trainingDigits/' + fileName) # 把每个样本所表示的数字读取为一个1*1024的向量


    testFiles = os.listdir('data/testDigits')  # 测试集路径
    errorCount = 0   # 分类错误计数
    testNum = len(testFiles)
    for i in range(testNum):
        fileName = testFiles[i]
        clas = int(fileName.split('_')[0])  # 样本的真实类别
        vectorUnderTest = img2vector('data/testDigits/' + fileName)  # 当前测试样本的向量
        # 使用kNN分类器对当前样本进行分类，设置k=3
        classifierResult = classify(vectorUnderTest, trainingMat, labels, 3)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, clas))
        if clas != classifierResult:
            errorCount += 1

    print("The total number of errors is: %d" % errorCount)
    print("the total error rate is %f: " %(errorCount/testNum))


if __name__ == "__main__":
    main()