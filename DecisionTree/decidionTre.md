## 决策树的基本概念
&emsp;&emsp;决策树是一类常用的机器学习方法，决策树实现决策的过程和我们平时做决定的过程很相似。想想如果自己马上要放假，要不要出去浪就是个大问题，首先考虑老板交代的接近deadline的项目有没有完成，如果完成了
就可一放心大胆的浪了，否则就乖乖磕研吧；任务完成了，但是转念一想，最近剁手太多没钱，算了还是宅着省钱吧；突然发现发工资了，有钱浪了，赶紧看看天气预报，如果假期天气不错果断室外放飞自我，如果不适合出行那就找一些室内活动……
如下图所示，是不是有种`if else`的感觉。
&emsp;&emsp;决策树的工作过程可以用类似于上图这样的流程图来表示。其中长方形代表判断模块(decision block)，代表决策的判断条件；椭圆形表示终止模块(terminating block)，表示决策的结论；此外从判断模块引出的箭头表示
分支(branch)，它指向另一个判断模块或是终止模块。如果将上面的流程图看成一棵树，每个判断模块就是一个节点，终止模块就是叶子节点，这样从根节点开始进行判断一直到叶子节点就形成了一条决策路径。显然，决策树是一种可解释的模型。

## 决策树的构造
&emsp;&emsp;决策树的决策过程比较容易理解，那么怎么构造一棵有效的决策树呢？正如程序中的判断条件的嵌套关系不能乱，构造决策树的时候也是遵循一定的规则的。想想一下我们平时做决策的时候，总是先考虑最重要的因素。构建决策树的思想也类似，
我们通过划分数据集得到**信息增益**最高的特征并优先考虑它。下面首先介绍一下香农熵和信息增益。
### 香农熵和信息增益
&emsp;&emsp;香农熵(Shannon Entropy)又叫做信息熵，是由香农提出的来自于信息论中的一个概念，用来衡量信息的无序程度(纯度)。简单来说，越是无序的数据，其包含的信息越多，其纯度越低，反之，越是有序的数据其纯度越高。假设当前样本集合为 $D$，
其中第 $k$ 类样本所占的比例为 $p_k, (k=1,2,...,|y|)$，|y|表示样本类别数。则数据集 $D$ 的香农熵定义为
$$Ent(D)=-\sum_{k=1}^{|y|}p_klog_2p_k \tag{1}$$
$Ent(D)$ 的值越小，则 $D$ 的纯度越高。接下来以一个短小精悍的数据集为例计算其香农熵：
```python
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

entropy = calShannonEnt(dataSet)
print(entropy)
```
上面的数据集前两列为特征，最后一列字符串为类别标签。运行上面这段程序可以得到以下输出结果：
接下来我们网数据集中加入另一个样本，再计算数据集的香农熵：
``dataSet.append([0, 0, 'maybe'])``
运行结果如下，加入新样本后由于增加了新的类别导致数据集的熵增大，意味着其纯度降低。
&emsp;&emsp;说完了香农熵，接下来说说信息增益。对于样本的某个特征属性 $a$, 假定它是离散的，并且在整个数据集上 $a$ 可能的取值为${a^1,a^2,...,a^V}$，那么如果我们根据属性 $a$的取值对数据集进行划分就可以得到
$V$ 个不同的子集，每个子集包含了所有在属性 $a$ 上取值为 $a^v$ 的样本。这样，利用特征属性 $a$ 对数据集 $D$ 划分所产生的信息增益可以由下式计算
$$Gain(D, a)=Ent(D)-\sum_{v=1}^{V}\frac{\left|D^v\right|}{|D|}Ent(D^v) \tag{2}$$
其中$\frac{\left|D^v\right|}$ 表示属性 $a$ 取值为 $a^v$ 的样本所占的比例。令上式取得最大值的属性就是划分当前数据集的最优属性，代码如下：
```python
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
```
上面的函数输入参数为带划分数据集，函数通过计算按不同特征属性划分数据集得到的信息增益，最后返回使得信息增益最大的特征的索引。那么返回信息增益最大的特征的索引有什么用呢？往下看就知道了
## 数据集划分
## 利用决策树实现西瓜数据集分类
## sklearn决策预测隐形眼镜类型