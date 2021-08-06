from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from math import log
import operator
import pandas as pd
import numpy as np
import pydotplus
import pickle  # pickle包可以将决策树保存下来，方便下次直接调用
from six import StringIO
from sklearn import tree


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    函数说明：计算给定数据集的经验熵（香农熵）
    :param dataSet:数据集
    :return:经验熵（香农熵）
    """
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每隔标签出现的次数
    labelCounts = {}
    # 对每组特征向量进行统计 得到不同分类的概率
    for featVec in dataSet:
        # 提取标签信息
        currentLabel = featVec[-1]
        # 当标签没有放入统计次数字典 就添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 否则加一计数
        labelCounts[currentLabel] += 1
    # 计算熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算该标签的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt


# myDat, labels = createDataSet()
# print(calcShannonEnt(myDat))
# myDat[0][-1] = 'what'
# print(calcShannonEnt(myDat))
# 熵越高 则说明混合的数据也越多（判断所需要的信息更多）


def splitDataSet(dataSet, axis, value):
    """
    函数说明：按照给定特征划分数据集
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value:需要返回的特征的值
    :return:划分后的数据集
    """
    # 创建返回的数据列表 函数内部对列表对象的修改，将会影响该列表对象的整个生存周期
    reDataSet = []
    # 遍历数据每一行
    for featVec in dataSet:
        if featVec[axis] == value:
            # 两次切片去掉axis特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            # 将符合条件的添加到返回的数据集 列表中嵌套列表
            reDataSet.append(reducedFeatVec)
            # extend列表结尾追加数据，如果数据是一个序列，则将这个序列的数据逐一添加到列表。
            # 而append是序列整体加到后面
    return reDataSet


# myDat, labels = createDataSet()
# print(splitDataSet(myDat, 0, 0))
# print(splitDataSet(myDat, 0, 1))
# print(splitDataSet(myDat, 0, 2))


def chooseBestFeatureToSplit(dataSet):
    """
    函数说明：选择最优特征
        Gain(D,g) = Ent(D) - SUM(|Dv|/|D|)*Ent(Dv)
    :param dataSet:数据集
    :return:bestFeature - 信息增益最大的（最优）特征的索引值
    """
    # 特征数量（总列数-标签数1）
    numFeatures = len(dataSet[0]) - 1
    # 计算数据集的熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    baseInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet第i个所有特征存在featList这个列表中
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复，重复的元素均被删掉
        # 从列表中创建集合是python语言得到列表中唯一元素值得最快方法
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 按第i格特征划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算熵
            newEntropy += prob*calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的增益
        print(f'第{i}个特征的信息增益为{infoGain:.3f}')
        # 计算信息增益
        if(infoGain > baseInfoGain):
            # 更新信息增益，找到最大的信息增益
            baseInfoGain = infoGain
            # 记录最大信息增益的特征的索引值
            bestFeature = i
    # 返回索引值
    return bestFeature


def majorityCnt(classList):
    """
    函数说明：统计classList中出现次数最多的元素（类标签）
        服务于递归第两个终止条件
    :param classList:类标签列表
    :return:sortedClassCount[0][0] - 出现次数最多的元素（类标签）
    """
    classCount = {}
    # 统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount += 1
    # 根据字典的值降序排序 - keys()  键  values() 值 items() 键值对
    # operator.itemgetter(1)获取对象的第1列的值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), revers=True)
    # 返回出现次数最多的元素
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    函数说明：创建决策树（ID3算法）
        递归有两个终止条件：1、所有的类标签完全相同，直接返回类标签
                        2、用完所有标签但是得不到唯一类别的分组，即特征不够用，挑选出现数量最多的类别作为返回
    :param dataSet:训练数据集
    :param labels:分类属性标签
    :return:决策树
    """
    # 取出类标签
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 当只有一个属性 类标签仍然不完全相同 返回出现次数最多的类标签
    elif len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 否则选择最优特征
    else:
        beastFeat = chooseBestFeatureToSplit(dataSet)
        # 最优特征的标签
        bestFeatLabel = labels[beastFeat]
        # 根据最优特征标签生成树  使用字典存储
        myTree = {bestFeatLabel: {}}
        # 删除已经使用的特征标签  引用传递这里删除外面也要删除
        del(labels[beastFeat])
        # 得到数据集中最优特征的属性值
        featValues = [example[beastFeat] for example in dataSet]
        # 去掉重复的属性值
        uniqueVals = set(featValues)
        # 遍历特征可能取值创建决策树
        for value in uniqueVals:
            # 保证每次调用函数createTree ()时不改变原始列表的内容 因为引用传递 会永久改变lables
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, beastFeat, value), subLabels)
        return myTree



def getNumLeafs(myTree):
    """
    函数说明：获取决策树叶子结点的数目
    :param myTree:决策树
    :return:numLeafs - 决策树的叶子结点的数目
    """
    # 初始化叶子数目
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不是list,需要转换
    # 可以使用list(myTree.keys())[0]获取结点属性
    # next() 返回迭代器的下一个项目 next(iterator[, default])
    firstStr = next((iter(myTree)))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该节点是否为字典，如果不是字典，代表该节点为叶子节点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs +=getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    函数说明：获取决策树的层数
    与获取叶子节点类似 使用递归
    叶子节点计算 访问到最低处才+1
    但这里每次访问子树都会+1  但是当前方向深度+1 最后取得最大值
    :param myTree:
    :return:
    """
    # 初始化决策树深度
    maxDepth = 0
    firststr = next(iter(myTree))
    secondDict = myTree[firststr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    函数说明：绘制结点
    :param nodeTxt:结点名
    :param centerPt:文本位置
    :param parentPt:标注的箭头位置
    :param nodeType:结点格式
    :return:
    """
    # 定义箭头格式
    arrow_args = dict(arrowstyle='<-')
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    # 绘制结点createPlot.ax1创建绘图区
    # annotate是关于一个数据点的文本
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    """
    函数说明：标注有向边属性值
    :param cntrPt:用于计算标注位置
    :param parentPt:用于计算标注位置
    :param txtString:标注内容
    :return:
    """
    # 计算标注位置（箭头起始位置的中点处）
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    函数说明：绘制决策树
    :param myTree:决策树（字典）
    :param parentPt:标注的内容
    :param nodeTxt:结点名
    :return:
    """
    # 设置结点格式boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yoff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        # 如果是叶结点，绘制叶结点，并标注有向边属性值
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    函数说明：创建绘图面板
    :param inTree: 决策树（字典）
    :return:
    """
    # 创建fig
    fig = plt.figure(1, facecolor="white")
    # 清空fig
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plotTree.totalW = float(getNumLeafs(inTree))
    # 获取决策树层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    # 显示绘制结果
    plt.show()


def classify(inputTree, featLabels, testVec):
    """
    函数说明：使用决策树分类
    :param inputTree:已经生成的决策树
    :param featLabels:labels标签数据集（无重复）
    :param testVec:测试数据列表，顺序对应最优特征标签
    :return:classLabel - 分类结果
    """
    # 获取决策树结点
    firstStr = next(iter(inputTree))
    # 下一个字典
    secondDict = inputTree[firstStr]
    # 获取下标位置
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    """
    函数说明：存储决策树
    :param inputTree:已经生成的决策树
    :param filename:决策树的存储文件名
    :return:
    """
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    """
    函数说明：读取决策树
    :param filename:filename - 决策树的存储文件名
    :return:pickle.load(fr) - 决策树字典
    """
    fr = open(filename, 'rb')
    return pickle.load(fr)


def main():
    dataSet, features = createDataSet()
    # features要改变 用另一个变量存储
    featLabels1 = features[:]
    myTree = createTree(dataSet, features)
    # storeTree(myTree, 'classifierStorage.txt')
    # myTree = grabTree('classifierStorage.txt')
    # print(myTree)
    # 测试数据
    testVec = [0, 1, 0, 0]
    result = classify(myTree, featLabels1, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')
    print(myTree)
    createPlot(myTree)
    # print(dataSet)
    # print(calcShannonEnt(dataSet))
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))


def main2():
    # 加载文件 这样更简便 不需要close
    with open('data/lenses.txt') as fr:
        # 处理文件，去掉每行两头的空白符，以\t分隔每个数据
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类别，保存在列表里
    lenses_targt = []
    for each in lenses:
        # 存储Label到lenses_targt中
        lenses_targt.append([each[-1]])
    # 特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesLabels2 = lensesLabels[:]
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    result = classify(lensesTree, lensesLabels2, ['presbyopic', 'myope', 'no', 'reduced'])
    print(result)
    createPlot(lensesTree)


def main3():
    # 加载文件 这样更简便 不需要close
    with open('data/lenses.txt') as fr:
        # 处理文件，去掉每行两头的空白符，以\t分隔每个数据
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类别，保存在列表里
    lenses_targt = []
    for each in lenses:
        # 存储Label到lenses_targt中
        lenses_targt.append([each[-1]])
    # 特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    # 提取信息，生成字典
    for each_label in lensesLabels:
        for each in lenses:
            # index方法用于从列表中找出某个值第一个匹配项的索引位置
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # 打印字典信息
    # print(lenses_dict)
    # 生成pandas.DataFrame用于对象的创建
    lenses_pd = pd.DataFrame(lenses_dict)
    # 打印数据
    # print(lenses_pd)
    # 创建LabelEncoder对象
    le = LabelEncoder()
    # 为每一列序列化
    for col in lenses_pd.columns:
        # fit_transform()干了两件事：fit找到数据转换规则，并将数据标准化
        # transform()直接把转换规则拿来用,需要先进行fit
        # transform函数是一定可以替换为fit_transform函数的，fit_transform函数不能替换为transform函数
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # 打印归一化的结果
    # print(lenses_pd)
    # 创建DecisionTreeClassifier()类
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
    # 使用数据构造决策树
    # fit(X,y):Build a decision tree classifier from the training set(X,y)
    # 所有的sklearn的API必须先fit
    clf = clf.fit(lenses_pd.values.tolist(), lenses_targt)
    dot_data = StringIO()
    # 绘制决策树
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                         class_names=clf.classes_, filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # 保存绘制好的决策树，以PDF的形式存储。
    graph.write_pdf("tree.pdf")
    # 预测
    print(clf.predict([[1, 1, 1, 0]]))


if __name__ == '__main__':
    # main()
    # main2()
    main3()