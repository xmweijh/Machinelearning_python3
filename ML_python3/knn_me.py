from matplotlib.font_manager import FontProperties
import numpy as np
import operator
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import time


def creatDataSet():
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


def classify0(inx, dataSet, labels, k):
    """
    :param inx:用于分类的数据（测试集）
    :param dataSet:用于训练的数据（训练集）（n*1维列向量）
    :param labels:分类标准（n*1维列向量）
    :param k:kNN算法参数，选择距离最小的k个点
    :return:sortedClassCount[0][0] - 分类结果
    """
    # 使用numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 将inX重复dataSetSize次  方便矩阵相维数相同
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，axis=1是行向量相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方求得欧氏距离
    distancess = sqDistances ** 0.5
    # 利用argsort返回distances升序排序  需要注意的是返回的是下标 方便后面labels对应 比如[100 89 108  115] 返回 [1 0 2 3 ]
    sorttedDistIndicies = distancess.argsort()
    # 记录次数
    classCount = {}
    # 选择出距离最小的k个点
    for i in range(k):
        # 取出排好序的第i个元素的类别
        # print(sorttedDistIndicies[i])
        voteIlabel = labels[sorttedDistIndicies[i]]
        # print(voteIlabel)
        # 字典的get()方法，返回指定键的值，如果值不在字典中返回0
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        # 降序排序字典 找到类别最多的
    # key = operator.itemgetter(1)根据字典的值进行排序
    # key = operator.itemgetter(0)根据字典的键进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回最多的类别， 即要分类的类别
    return sortedClassCount[0][0]


# group, labels = creatDataSet()
# predict = classify0([0, 0], group, labels, 3)
# print(predict)


def file2martic(filename):
    """
    :param filename:文件名
    :return:
    returnMat - 特征矩阵
    classLabelVector - 分类label向量
    """
    # 打开文件
    fr = open(filename)
    # 读取所有文件到一个列表
    arrayOlines = fr.readlines()
    # 得到文件的行数
    numberOflines = len(arrayOlines)
    # 初始化返回矩阵
    returnMat = np.zeros((numberOflines, 3))
    # 创建分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    # 读取每一行
    for line in arrayOlines:
        # 去掉每一行首尾的空白符 如回车 换行等等
        line = line.strip()
        # 将每一行内容根据'\t'进行切片
        listFromLine = line.split('\t')
        # 将数据的前三列进行提取保存到returnMat作为特征举证
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本内容进行分类
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    # 返回标签向量 以及 特征矩阵
    return returnMat, classLabelVector


def showdatas(datingDataMat, datingLabels):
    """
    :param datingDataMat:特征矩阵
    :param datingLabels:分类Label
    :return:
    """
    # 设置汉字字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列，不共享x轴和y轴，fig画布的大小为（13，8）
    # 当nrows=2，ncols=2时，代表fig画布被分为4个区域，axs[0][0]代表第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    # label的颜色配置矩阵
    LabelsColors = []
    for i in datingLabels:
        # didntLike
        if i == 1:
            LabelsColors.append('black')
        # smallDoses
        if i == 2:
            LabelsColors.append('orange')
        # largeDoses
        if i == 3:
            LabelsColors.append('red')
    # 画出散点图，以datingDataMat矩阵第一列为x，第二列为y，散点大小为15, 透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题，x轴label， y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    # weigh 加粗
    plt.setp(axs0_title_text, size=14, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black')
    # 画出散点图，以datingDataMat矩阵第一列为x，第三列为y，散点大小为15, 透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题，x轴label， y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=14, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black')
    # 画出散点图，以datingDataMat矩阵第二列为x，第三列为y，散点大小为15, 透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题，x轴label， y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰淇淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰淇淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=14, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


# datingDataMat, datingLabels = file2martic('data\datingTestSet.txt')
# showdatas(datingDataMat, datingLabels)


# 数据归一化 此问题飞行距离计算差值明显大于其他特征  按距离算影响会过大
def autoNorm(dataSet):
    """
    :param dataSet:特征矩阵
    :return:normDataSet - 归一化后的特征矩阵
            ranges - 数据范围
            minVals - 数据最小值
    """
    # 获取最小值 min(0)返回该矩阵中每一列的最小值  min(1)返回该矩阵中每一行的最小值 min()返回的就是a中所有元素的最小值
    minVals = dataSet.min(0)
    # 获取最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # shape(dataSet)返回举证行列数目
    normDataSet = np.zeros(np.shape(dataSet))
    # shape[0]返回行数
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 归一化
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 归一化数据结果，数据范围，最小值
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    测试
    :return:
    """
    filename = "data\datingTestSet.txt"
    datingDataMat, datingLabels = file2martic(filename)
    # 取所有数据的10% hoRatio越小，错误率越低
    hoRatio = 0.1
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获取normMat的行数
    m = normMat.shape[0]
    # 10%的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集
        # k选择label数+1（结果比较好）
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


# datingClassTest()


def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 输入特征
    percntTats = float(input('玩游戏所消耗时间百分比：'))
    ffMiles = float(input('每年飞行里程数：'))
    iceCream = float(input('每周消耗的冰淇淋公升数：'))
    filename = "data\datingTestSet.txt"
    datingDataMat, datingLabels = file2martic(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([percntTats, ffMiles, iceCream])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classify0(norminArr, normMat, datingLabels, 4)
    print(f'你可能{resultList[classifierResult-1]}')


# classifyPerson()