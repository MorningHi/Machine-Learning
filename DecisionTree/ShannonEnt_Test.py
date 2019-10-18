#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/9/29
@Author  : Rezero
'''


from math import log

def calShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数量
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 样本向量的最后一个元素是对应的标签
        # 对每一类样本进行计数，存储为字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0   # 计算信息熵
    for key in labelCounts.keys():
        prob = labelCounts[key] / numEntries   # 类别为key的样本的比例(概率)
        shannonEnt  -= prob * log(prob, 2)

    return shannonEnt


dataSet = [[1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']]

# entropy = calShannonEnt(dataSet)
# print(entropy)
dataSet.append([0, 0, 'maybe'])
entropy = calShannonEnt(dataSet)
print('After adding a new ‘maybe’ sample, entropy is %f' % entropy)