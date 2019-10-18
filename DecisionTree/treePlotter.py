#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2019/10/1
@Author  : Rezero
'''

import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="sawtooth", fc="0.8") # 决策节点
leafNode = dict(boxstyle="round4", fc="0.8")   # 叶子节点
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)



# 获取树的叶子节点数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict =myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':   # 表明当前节点是非叶子节点，则递归
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1       # 叶子节点
    return numLeafs

# 计算树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getNumLeafs(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制父节点和子节点之间的文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 树的叶子数目
    depth = getTreeDepth(myTree)     # 树的深度
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + numLeafs) /2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':    # 递归地进行节点绘制
            plotTree(secondDict[key],cntrPt, str(key))
        else:                                           # 绘制叶子节点
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD  # 更新下一个节点的y坐标

def createPlot(inTree):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 0.8
    plotTree(inTree, (0.6, 0.8), '')
    plt.show()