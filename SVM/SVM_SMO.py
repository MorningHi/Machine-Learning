#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/16
@Author  : Rezero
'''
# 完整版SVM，使用完整版SMO算法进行优化，并使用RBF核


import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split


class optStruct:
    def __init__(self, data, labels, C, toler, kTup=('Linear', 0)):
        self.X = data
        self.y = labels
        self.C = C
        self.toler = toler
        self.m = data.shape[0]   # 样本数量
        self.alphas = np.zeros(self.m)
        self.b = 0
        self.cacheE = np.zeros((self.m, 2)) # 误差缓存，第一列表示是否有效的标志，第二列为对应的误差值
        self.kTup = kTup
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], self.kTup)   # 对样本进行核函数映射

def kernelTrans(X, A, kTup):
    m, n = X.shape
    K = np.zeros((m, 1))
    if kTup[0] == 'Linear':
        K = np.dot(X, A)   # 线性核
    elif kTup[0] == 'RBF':
        K = np.exp(-np.sum((X - A) * (X - A), axis=1) / kTup[1] ** 2)
    else:
        raise NameError('无法识别的核函数')
    return K

def loadData(fileName):
    dataSet = np.loadtxt(fileName)
    return dataSet[:, :-1], dataSet[:, -1]  # 最后一列为类别标签

def calcEk(oS, k):
    '''计算在第k个样本上的误差'''
    fXk = np.dot(oS.K[:, k], oS.alphas * oS.y) + oS.b
    Ek = fXk - oS.y[k]
    return Ek

def selectJrand(i, m):
    j = 0
    while j == i:
        j = np.random.randint(m)
    return j

def clipAlpha(alpha, L, H):
    '''
    对alpha进行越界处理
    :param L: 下边界
    :param H: 上边界
    '''
    if alpha < L:
        alpha = L
    elif alpha > H:
        alpha = H
    return alpha

def selectJ(i, oS, Ei):
    '''选择另一个需要优化的alpha_j'''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.cacheE[i] = [1, Ei]
    validCacheEList = np.nonzero(oS.cacheE[:, 0])[0]   # 标志位不为零的有效误差列表
    # 在有效列表里找到可以最大的 Ei-Ek 对应的k
    if len(validCacheEList) > 0:
        for k in validCacheEList:
            if k == i:   continue   # 两个需要优化的alpha应当不相同
            Ek = calcEk(oS, k)   # 计算样本Xk的误差
            deltaE = np.abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK, Ej
    else:
        # 有效列表为空则随机选择一个j并计算误差
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej

def updateEk(oS, k):
    '''更新误差缓存中的Ek'''
    Ek = calcEk(oS, k)
    oS.cacheE[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    # 优化alphas
    if (oS.y[i] * Ei < -oS.toler and oS.alphas[i] < oS.C) or (oS.y[i] * Ei > oS.toler and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)  # 选择另一个要优化的alpha
        alphaIold = oS.alphas[i]   # 保存更新前的alpha_i 和 alpha_j的值
        alphaJold = oS.alphas[j]
        # 计算alpha_j的上下边界
        if oS.y[i] != oS.y[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H:  print("L == H");  return 0

        eta = 2 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0: print("eta >= 0");   return 0
        # 更新alpha_j
        oS.alphas[j] -= oS.y[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("alpha_j not moving enough")
            return 0
        # alpha_i和alpha_j变化方向相反
        oS.alphas[i] += oS.y[i] * oS.y[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.y[i] * (oS.alphas[i] - alphaIold) * oS.K[i , i] - oS.y[j] * (oS.alphas[j] - alphaJold) * oS.K[j, i]
        b2 = oS.b - Ej - oS.y[i] * (oS.alphas[i] - alphaIold) * oS.K[i , j] - oS.y[j] * (oS.alphas[j] - alphaJold) * oS.K[j , j]
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0


def smoP(oS, maxIter):
    iter = 0
    entireSet = True
    alphaParisChanged = 0
    # 迭代停止条件：迭代次数超过最大次数或者在整个数据集上都未对任意alpha对进行更新
    while iter < maxIter and (alphaParisChanged > 0 or entireSet):
        alphaParisChanged = 0
        if entireSet:   # 遍历整个数据集
            for i in range(oS.m):
                alphaParisChanged += innerL(i ,oS)
                print("完整遍历，第%d次迭代，样本编号：%d，alpha优化对数：%d" % (iter, i, alphaParisChanged))
            iter += 1
        else:   # 非边界遍历
            nonBoundIs = np.nonzero((oS.alphas > 0) * (oS.alphas < C))[0]  # 非边界值alpha的索引
            for i in nonBoundIs:
                alphaParisChanged += innerL(i, oS)
                print("非边界值遍历， 第%d次迭代，样本编号：%d，alpha优化对数：%d" % (iter, i, alphaParisChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaParisChanged == 0:
            entireSet = True

        print("迭代次数：%d" % iter)

    return oS.alphas, oS.b

def showClassifier(X, y, w, alphas, b):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Paired)

    svsIndex = np.where(alphas > 0)  # 支持向量样本
    for i in svsIndex:
        plt.scatter(X[i, 0], X[i, 1], color='', marker='o', edgecolors='g', s=300)

    # 绘制分割平面
    # x1 = np.min(X[:, 0])
    # x2 = np.max(X[:, 0])
    # y1 = -(w[0] * x1 + b) / w[1]
    # y2 = -(w[0] * x2 + b) / w[1]
    # plt.plot([x1, x2], [y1, y2])
    plt.show()

def get_w(X, y, alphas):
    '''
    根据数据集和求得的alphas计算w
    '''
    return np.dot(y * alphas, X)

def predict(X, sVs, labelSV, alphas, b, kTup):
    '''
    根据支持向量预测样本的类别
    :param X:        输入样本
    :param sVs:      支持向量
    :param labelSV:  支持向量的类别标签
    :param alphas:   支持向量对应的alpha
    :return:         输入样本的预测结果
    '''
    m = X.shape[0]
    y = np.zeros(m)
    for i in range(m):
        kernelEval = kernelTrans(sVs, X[i, :], kTup)   # 对样本进行转换
        y[i] = np.dot(kernelEval.T, labelSV * alphas) + b
    return y


if __name__ =="__main__":
    # dataSet, labels = loadData('data/testSet.txt')
    # startTime = time.time()
    C = 100
    toler = 0.001
    maxIter = 50
    # oS = optStruct(dataSet, labels, C, toler)
    # alphas, b = smoP(oS, 40)   # 使用线性核解决线性分类
    # endTime = time.time()
    # print((endTime - startTime))
    # w = get_w(dataSet, labels, alphas)  # 计算w
    # showClassifier(dataSet, labels, w, alphas, b)


    # 加载非线性可分数据集并使用RBF核进行测试
    dataSet, labels = loadData('data/nonLinearSet.txt')
    trainX, testX, trainY, testY = train_test_split(dataSet, labels, test_size=0.3, random_state=0)  # 分割训练集和测试集

    oS = optStruct(trainX, trainY, C, toler, ('RBF', 1.5))
    alphas, b = smoP(oS, 40)  # 优化alphas和b
    w = get_w(trainX, trainY , alphas)
    showClassifier(dataSet, labels, w, alphas, b)
    svIndex = np.where(alphas > 0)[0]   # 支持向量的样本编号
    print("支持向量数：%d" % len(svIndex))
    sVs = trainX[svIndex]        # 支持向量
    labelSV = trainY[svIndex]    # 支持向量对应的编号

    train_predict = predict(trainX, sVs, labelSV, alphas[svIndex], b, oS.kTup)  # 训练集上的预测结果
    train_error = len(np.where((trainY * train_predict) < 0)[0]) / trainX.shape[0]
    print("训练集上的错误率：%.2f" % train_error)

    test_predict = predict(testX, sVs, labelSV, alphas[svIndex], b, oS.kTup)
    test_error = len(np.where((testY * test_predict) < 0)[0]) / testX.shape[0]
    print("测试集上的错误率：%.2f" % test_error)





