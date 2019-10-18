#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/4
@Author  : Rezero
'''

import os
import numpy as np

def textParse(bigString):
    import re
    listOfTokens = re.split('\W+', bigString) # 分割邮件内容，\W* 表示除字母、数字以外的其他字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def loadDataSet():
    docList = []
    classList = []
    hamPath = 'email/ham/'     # 正常邮件的路径
    spamPath = 'email/spam/'   # 垃圾邮件路径
    
    fileList = os.listdir(hamPath)
    for i in range(len(fileList)):
        with open(hamPath + fileList[i]) as fr:
            wordList = textParse(fr.read()) # 获取每个文件的词向量
        docList.append(wordList)
        classList.append(1)  # 正常邮件的类别标签为1
    fileList = os.listdir(spamPath)
    for i in range(len(fileList)):
        with open(spamPath + fileList[i]) as fr:
            wordList = textParse(fr.read())
        docList.append(wordList)
        classList.append(0)  # 垃圾邮件的类别标签为0
    return docList, classList
  
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)   # 求新的文档词向量集合和原集合的并集
    return list(vocabSet)

# 根据vocabList将句子转换为向量表示
def Word2Vec(vocabList, wordList):
    wordVec = [0] * len(vocabList)
    for word in wordList:
        if word in vocabList:  # 对出现在句子中的vocabList中的元素进行计数
            wordVec[vocabList.index(word)] += 1
    return wordVec
    
def split_train_test(dataSet, labels):
    testIndex = [np.random.randint(len(docList)) for i in range(10)]
    trainingSet = []
    trainingClasses = []
    testSet = []
    testClasses = []
    for i in range(len(dataSet)):
        if i in testIndex:
            testSet.append(dataSet[i])
            testClasses.append(labels[i])
        else:
            trainingSet.append(Word2Vec(vocabList, dataSet[i]))
            trainingClasses.append(labels[i])
    return trainingSet, trainingClasses, testSet, testClasses
    
def trainNB(trainingSet, trainingClasses):
    trainLength = trainingSet.shape[0]
    numWords = trainingSet.shape[1]
    pAbusive = np.sum(trainingClasses) / trainLength   # 类别为1的概率 p(c_i=1)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(trainLength):
        if trainingClasses[i] == 1:
            p1Num += trainingSet[i, :]
            p1Denom += np.sum(trainingSet[i, :])
        else:
            p0Num += trainingSet[i ,:]
            p0Denom += np.sum(trainingSet[i, :])
    p1Vec = np.log(p1Num / p1Denom)    # 每个类别的条件概率
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive

def classifyNB(testVec, p0Vec, p1Vec, pClass1):
    p0 = np.sum(testVec * p0Vec) + np.log(1 - pClass1)
    p1 = np.sum(testVec * p1Vec) + np.log(pClass1)
    if p0 > p1:
        return 0
    else:
        return 1

if __name__ == "__main__":  
    docList, classList = loadDataSet()   # 加载数据集和标签
    vocabList = createVocabList(docList)  # 创建文档中所有出现的不重复词列表
    
    # 从数据集中随机选出十组数据作为训练集
    trainingSet, trainingClasses, testSet, testClasses = split_train_test(docList, classList)
    
    # 根据训练集计算条件概率
    p0V, p1V, pSpam = trainNB(np.array(trainingSet), np.array(trainingClasses))
    
    # 在测试集上进行测试
    error = 0
    for i in range(len(testSet)):
        testVec = Word2Vec(vocabList, testSet[i])
        if classifyNB(np.array(testVec), p0V, p1V, pSpam) != testClasses[i]:
            error += 1
            print("classification error: " + str(testSet[i]))
    print("The error rate is: %f" % (error / len(testSet)))