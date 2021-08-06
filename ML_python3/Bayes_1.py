from functools import reduce
import numpy as np


def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    # 返回实验样本切分的词条、类别标签向量
    return postingList, classVec


def createVocabList(dataSet):
    """
    函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataSet:整理的样本数据集
    :return:返回不重复的词条列表，也就是词汇表
    """
    # 创造集合 为了不重复
    vocabSet = set([])
    for document in dataSet:
        # 取并集 &交集 -差集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setodWords2Vec(vocabList, inputSet):
    """
    函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    :param vocabList:createVocabList返回的列表
    :param inputSet:切分的词条列表
    :return:returnVec - 文档向量，词集模型
    """
    # 创建一个0向量
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        if word in vocabList:
            # 如果词条 存在于词汇中  置位为1
            # 若这里改为+=则就是基于词袋的模型，遇到一个单词会增加单词向量中的对应值
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"the word{word} is not  in my Vocabulary")
    return returnVec


def trainNB0(trainMartix, trainCatgory):
    """
    函数说明：朴素贝叶斯分类器训练函数
    :param trainMartix:训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    :param trainCatgory:训练类标签向量，即loadDataSet返回的classVec
    :return:
    p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
    """
    # 计算训练文档的 数目
    numTrainDocs = len(trainMartix)
    # 计算每篇文档的词条数目
    numWords = len(trainMartix[0])
    # 文档术语侮辱类的概率 侮辱值为1 相加即为他数目
    pAbusive = sum(trainCatgory)/float(numTrainDocs)
    # 创建numpy.zeros数组，词条出现数初始化为0
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # 创建numpy.ones数组，词条出现数初始化为1,拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 分母初始化为0
    # p0Denom = 0.0
    # p1Denom = 0.0
    # 分母初始化为2，拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
        if trainCatgory[i] == 1:
            # 统计所有侮辱类文档中每个单词出现的次数
            p1Num += trainMartix[i]
            # 统计侮辱文档一共出现的单词的个数
            p1Denom += sum(trainMartix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)...
        else:
            # 统计所有非侮辱类文档的每个单词出现的次数
            p0Num += trainMartix[i]
            # 统计非侮辱文档一共出现的单词的个数
            p0Denom += sum(trainMartix[i])
    # 每个侮辱类单词分别出现的概率
    # p1Vect = p1Num / p1Denom
    # 取对数，防止下溢出
    p1Vect = np.log(p1Num / p1Denom)
    # 每个非侮辱类单词分别出现的概率
    # p0Vect = p0Num / p0Denom
    # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    # 返回属于侮辱类的条件概率数组、属于非侮辱类的条件概率数组、文档属于侮辱类的概率
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    函数说明：朴素贝叶斯分类器分类函数
    :param vec2Classify:待分类的词条数组
    :param p0Vec:侮辱类的条件概率数组
    :param p1Vec:非侮辱类的条件概率数组
    :param pClass1:文档属于侮辱类的概率
    :return:
    """
    # 对应元素相乘
    # p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1
    # p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    # 对应元素相乘，logA*B = logA + logB所以这里是累加
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    # 创建实验样本
    listOPosts, listclasses = loadDataSet()
    # 创建词汇表,将输入文本中的不重复的单词进行提取组成单词向量
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        # 将实验样本向量化若postinDoc中的单词在myVocabList出现则将returnVec该位置的索引置1
        # 将6组数据list存储在trainMat中
        trainMat.append(setodWords2Vec(myVocabList, postinDoc))
    # 训练朴素贝叶斯分类器
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listclasses))
    # 测试样本1
    testEntry = ['love', 'my', 'dalmation']
    # 测试样本向量化返回这三个单词出现位置的索引
    thisDoc = np.array(setodWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        # 执行分类并打印结果
        print(testEntry, '属于侮辱类')
    else:
        # 执行分类并打印结果
        print(testEntry, '属于非侮辱类')
    # 测试样本2
    testEntry = ['stupid', 'garbage']
    # 将实验样本向量化
    thisDoc = np.array(setodWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        # 执行分类并打印结果
        print(testEntry, '属于侮辱类')
    else:
        # 执行分类并打印结果
        print(testEntry, '属于非侮辱类')


def main():
    ListOPosts, listClasses = loadDataSet()
    myVocalbList = createVocabList(ListOPosts)
    # print(myVocalbList)
    # print(setodWords2Vec(myVocalbList, ListOPosts[0]))
    trainMat = []
    for postinDoc in ListOPosts:
        trainMat.append(setodWords2Vec(myVocalbList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(pAb)
    # print(p0V)
    # print(p1V)


if __name__ == '__main__':
    # main()
    testingNB()