#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/15
@Author  : Rezero
'''

# 在线性可分的数据集testSet上实现SVM

import numpy as np
import matplotlib.pyplot as plt
import time


def loadData(fileName):
    dataSet = np.loadtxt(fileName)
    return dataSet[:, :-1], dataSet[:, -1]  # 最后一列为类别标签

def selectJ(i, m):
    '''
    随机选择另一个需要优化的alpha_j， j != i
    :param i: alpha_i的索引
    :param m: 样本总数
    :return: j
    '''
    j = i
    while j == i:
        j =np.random.randint(m)
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


def simpleSMO(X, y, C, toler, maxIter):
    '''
    利用简化的SMO算法优化SVM的目标函数
    :param X: 数据集的特征
    :param y: 数据集的类别标签
    :param C: 软间隔有关的常数
    :param toler: 容错率
    :param maxIter: 最大迭代次数
    :return: alpha和b
    '''
    m, n = dataSet.shape  # m,n 分别为样本数和特征数
    b = 0
    alphas = np.zeros(m)
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0 # 成对alpha改变量
        for i in range(m):
            # 计算误差Ei
            fXi = np.dot(np.dot(X, X[i, :]).T, alphas * y) + b
            Ei = fXi - y[i]
            # 优化alphas
            if (y[i] * Ei < -toler and alphas[i] < C) or (y[i] * Ei > toler and alphas[i] > 0):
                j = selectJ(i, m)  # 随机选择另一个要优化的alpha_j
                # 计算误差Ej
                fXj = np.dot(np.dot(X, X[j, :]).T, alphas * y) + b
                Ej = fXj - y[j]
                # 更新前的alpha_i 和 alpha_j的值
                alphaIold = alphas[i]
                alphaJold = alphas[j]
                # 根据 yi 和 yj 确定 alpha_j 的上下界
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])

                if L == H: print("L == H"); continue
                # 计算ets
                eta = 2 * np.dot(X[i, :], X[j, :].T) - np.dot(X[i, :], X[i, :].T) - np.dot(X[j, :], X[j, :].T)
                if eta >= 0: print("eta >= 0");  continue
                # 更新alpha_i 和 alpha_j
                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], L, H)
                if np.abs(alphas[j] - alphaJold) < 0.00001:
                    print("alpha_j not moving enough")
                    continue
                # alpha_i和alpha_j变化方向相反
                alphas[i] += y[i] * y[j] * (alphaJold - alphas[j])
                # 更新b
                b1 = b - Ei - y[i] * (alphas[i] - alphaIold) * np.dot(X[i, :], X[i, :].T) - y[j] * (alphas[j] - alphaJold) * np.dot(X[j, :], X[i, :].T)
                b2 = b - Ej - y[i] * (alphas[i] - alphaIold) * np.dot(X[i, :], X[j, :].T) - y[j] * (alphas[j] - alphaJold) * np.dot(X[j, :], X[j, :].T)
                if alphas[i] > 0 and alphas[i] < C:
                    b = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaPairsChanged += 1
                print("iter: %d i: %d, paris changed %d" % (iter, i, alphaPairsChanged))

        if alphaPairsChanged == 0:
             iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)

    return alphas, b

def showClassifier(X, y, w, alphas, b):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Paired)

    svsIndex = np.where(alphas > 0)  # 支持向量样本
    for i in svsIndex:
        plt.scatter(X[i, 0], X[i, 1], color='', marker='o', edgecolors='g', s=300)

    # 绘制分割平面
    x1 = np.min(X[:, 0])
    x2 = np.max(X[:, 0])
    y1 = -(w[0] * x1 + b) / w[1]
    y2 = -(w[0] * x2 + b) / w[1]
    plt.plot([x1, x2], [y1, y2])
    plt.show()

def get_w(X, y, alphas):
    '''
    根据数据集和求得的alphas计算w
    '''
    return np.dot(y * alphas, X)


if __name__ =="__main__":
    dataSet, labels = loadData('data/testSet.txt')
    startTime = time.time()
    alphas, b = simpleSMO(dataSet, labels, 0.6, 0.001, 40)
    endTime = time.time()
    print("running time: %.5fs" % (endTime - startTime))

    w = get_w(dataSet, labels, alphas)  # 计算w

    showClassifier(dataSet, labels, w, alphas, b)


