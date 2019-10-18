#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/9/29
@Author  : Rezero
'''

import operator
from math import log
import pickle
import treePlotter


def loadData(filepath):
    dataSet = []
    with open(filepath, encoding='utf-8') as fr:
        for line in fr.readlines():
            sample = line.strip().split('\t')
            dataSet.append(sample)
    features = dataSet[0][:-1]  # 第一行为特征标签
    del(dataSet[0])
    return dataSet, features

# 计算数据集的香农熵
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
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        shannonEnt  -= prob * log(prob, 2)

    return shannonEnt

def splitDataSet(dataSet, axis, values):
    '''
    :param dataSet: 原始数据集
    :param axis:    当前特征的索引
    :param values:  当前的特征值
    :return:
    '''
    retDataset = []
    # 遍历数据集，找出所有当前维特征值和传入参数的特征值相同的样本
    for featVec in dataSet:
        if featVec[axis] == values:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])  # 去掉当前维特征构成新样本
            retDataset.append(reduceFeatVec)    # 符合条件的样本构成子集
    return retDataset

# 得到最优划分特征的索引
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数
    baseEntropy = calShannonEnt(dataSet)   # 整个数据集的原始香农熵
    bestInfoGain = 0   # 最大的信息增益
    bestFeature = -1   # 最优特征的索引
    for i in range(numFeatures):
        featureList = [sample[i] for sample in dataSet] # 每个样本的第i个特征组成的列表
        uniqueVals = set(featureList)  # 去重复，得到第i个特征的所有可能值

        # 根据当前特征的不同值划分数据集并计算信息熵
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy   # 当前划分的信息增益
        # 得到最大信息增益对应的划分特征的索引
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain

    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:  # 用字典存储每一类的计数
            classCount[vote] = 0
        classCount[vote] += 1
    # 将每一类的计数按升序排序并返回计数最多的类别标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, features):
    '''
    递归的方式创建决策树
    :param dataSet: 数据集
    :param features:  特征标签/特征名称的列表
    :return: 决策树(用字典存储)
    '''
    classList = [example[-1] for example in dataSet]  # 类别列表
    # 递归停止条件一：所有的类别标签相同，此时所有样本被划分到同一类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归停止条件2：已经遍历完了所有特征，仍没有得到有效划分。len(dataSet[0])=1原因是遍历完所有特征后每一个样本仅剩类别标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 此时返回出现次数最多的类别

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = features[bestFeat]  # 最好分类特征对应的特征标签
    myTree = {bestFeatLabel:{}}
    del(features[bestFeat])  # 当前特征从特征列表里删除
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)   # 当前特征的所有特征值
    for value in uniqueValues:
        subFeatures = features[:]
        # 递归地构建子树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subFeatures)

    return myTree

def storeTree(inputTree, fileName):
    with open(fileName,'wb') as fp:
        pickle.dump(inputTree, fp)

def grabTree(fileName):
    with open(fileName, 'rb') as f:
        tree = pickle.loads(f.read())
    return tree

def classify(inputTree, features, testVec):
    '''
    根据树的结构对输入样本进行分类
    :param inputTree: 决策树的结构
    :param features:  特征标签列表
    :param testVec:   输入的测试样本特征向量
    :return:          分类结果
    '''
    firstStr = list(inputTree.keys())[0]  # 决策树的第一个特征属性
    secondDict = inputTree[firstStr]
    featIndex = features.index(firstStr)
    classLabel = None
    for key in secondDict.keys():  # 遍历子树
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':  # 说明当前结点还有子树
                classLabel = classify(secondDict[key], features, testVec)
            else:  # 遍历到叶子节点，则叶子结点就是分类结果返回
                classLabel = secondDict[key]
    return classLabel

if __name__ == "__main__":
    dataSet, features = loadData('data/watermelon2_0.txt')
    myTree = createTree(dataSet, features.copy())
    treePlotter.createPlot(myTree)
    print(myTree)

    storeTree(myTree, 'myTree.txt')
    loadTree = grabTree('myTree.txt')
    print(loadTree)

    vector = ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑']
    result = classify(myTree, features, vector)
    print(result)


