#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/20
@Author  : Rezero
'''

import numpy as np

def loadDataSet(path):
    dataSet = np.loadtxt(path)
    X = dataSet[:, :-1]
    y = 2 * dataSet[:, -1] - 1   # 注意将类标签换成-1， 1
    return X, y

def stumpClassify(X, dimen, threshVal, inqual):
    '''根据阈值和当前维度(dimen)的特征值返回样本分类结果'''
    returnArr = np.ones(X.shape[0])
    if inqual == 'lt':
        returnArr[X[:, dimen] <= threshVal] = -1.0
    else:
        returnArr[X[:, dimen] >= threshVal] = -1.0
    return returnArr

def buildStump(X, y, D):
    '''
    根据数据集构建一个决策树桩
    :param X: 训练集的特征
    :param y: 训练集的类别标签
    :param D: 样本的权重向量
    :return: 训练出的决策树桩，误差，分类结果
    '''
    m, n = X.shape
    numStep = 10
    bestStump = {}  # 训练出的决策树桩用字典存储
    bestClassEst = np.zeros(m)  # 样本的类别估计值
    minError = np.inf
    for i in range(n):  # 遍历特征
        rangeMin = np.min(X[:, i])   # 特征取值的区间范围
        rangeMax = np.max(X[:, i])
        stepSize = (rangeMax - rangeMin) / numStep   # 根据步数和范围求步长
        for j in range(-1, numStep+1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + (j * stepSize)  # 第j个区间的上边界,用来作为当前分类的阈值
                predictedVals = stumpClassify(X, i, threshVal, inequal)
                error = np.ones(m)
                error[predictedVals == y] = 0  # 通过错误分类计算当前决策树桩的分类错误率
                weightedError = np.dot(D, error)  # 加权错误率
                if weightedError < minError:     # 记录误差最小的决策树桩
                    minError = weightedError
                    bestClassEst = predictedVals
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return  bestStump, minError, bestClassEst





def adaBoostTrainDS(X, y, numIt=40):
    '''
    :param X: 训练集的特征
    :param y: 训练集的类别标签
    :param numIt: 最大迭代次数，且每次迭代会生成一个弱分类器
    :return: 弱分类器序列
    '''
    weakClassArr = []
    m = X.shape[0]  # 训练样本数目
    D = np.ones(m) / m   # 初始每个样本权重相同
    aggClassEst = np.zeros(m)  # 样本的类别估计值
    for i in range(numIt):
        # 训练一个决策树桩分类器
        bestStump, error, classEst = buildStump(X, y, D)
        # print("样本权重: ", D.T)
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-16))  # 根据误差为当前分类器分配权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 当前的决策树桩加入弱分类器列表
        # print("样本类别估计值：", classEst)
        D *= np.exp(-1 * alpha * y * classEst)   # 更新每个样本的权重
        D /= np.sum(D)
        aggClassEst += alpha * classEst  # 当前集成分类器的分类结果
        # print("分类结果：", aggClassEst)
        errorRate = np.dot(np.sign(aggClassEst) != y, np.ones(m)) / m
        print("累加错误率：%f" % errorRate)
        if abs(errorRate - 0.0) < 1e-10:
            break
    return weakClassArr

def adaClassify(X, classifierArr):
    '''
    分类函数
    :param X: 待分类样本
    :param classifierArr: 弱分类器列表
    :return: 分类结果
    '''
    m = X.shape[0]
    aggClassEst = np.zeros(m)
    for i in range(len(classifierArr)):
        classEst = stumpClassify(X, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)


if __name__ == "__main__":
    trainX, trainY = loadDataSet('data/horseColicTraining.txt')   # 加载数据集
    testX, testY = loadDataSet('data/horseColicTest.txt')

    # 在训练数据集上训练出多个弱分类器
    classifierArray = adaBoostTrainDS(trainX, trainY, 500)

    test_predict = adaClassify(testX, classifierArray)
    error = len(np.where(test_predict != testY)[0]) / len(testY)
    print("错误率：%f" % error)

