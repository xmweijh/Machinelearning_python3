import numpy as np
import re
import random


def textParse(bigString):
    """
    函数说明：接受一个大字符串并将其解析为字符串列表
    :param bigString:
    :return:
    """
    # 用特殊符号作为切分标志进行字符串切分，即非字母、非数字
    # \W* (W大写)0个或多个非字母数字或下划线字符（等价于[^a-zA-Z0-9_]）
    listOfTockens = re.split(r'\W*', bigString)
    # 单词变成小写，去掉少于两个字符的字符串
    return [tok.lower() for tok in listOfTockens if len(tok) > 2]


def creatrVocabList(dataSet):
    """
    函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

    :param dataSet:整理的样本数据集
    :return: vocabSet - 返回不重复的词条列表，也就是词汇表
    """
    # 创建一个空的不重复列表
    # set是一个无序且不重复的元素集合
    vocabSet = set([])
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocablist, inputSet):
    """
    函数说明：根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    :param vocablist: createVocabList返回的列表
    :param inputSet:切分的词条列表
    :return:文档向量，词集模型
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocablist)
    # 遍历每个词条
    for word in inputSet:
        if word in vocablist:
            # 如果词条存在于词汇表中，则置1
            # index返回word出现在vocabList中的索引
            # 若这里改为+=则就是基于词袋的模型，遇到一个单词会增加单词向量中德对应值
            returnVec[vocablist.index(word)] = 1
        else:
            print(f'the word:{word} is not in my Vocabulary')
    return returnVec


def setOfWords2Vec(vocablist, inputSet):
    """
    函数说明：根据vocabList词汇表，构造词袋模型
    :param vocablist:createVocabList返回的列表
    :param inputSet:切分的词条列表
    :return: returnVec - 文档向量，词袋模型
    """
    returnVec = [0] * len(vocablist)
    # 遍历每个词条
    for word in inputSet:
        if word in vocablist:
            # 如果词条存在于词汇表中，则置1
            # index返回word出现在vocabList中的索引
            # 若这里改为+=则就是基于词袋的模型，遇到一个单词会增加单词向量中德对应值
            returnVec[vocablist.index(word)] += 1
        else:
            print(f'the word:{word} is not in my Vocabulary')
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
    pAbusive = sum(trainCatgory) / float(numTrainDocs)
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
    # 同理用vec2Classify来乘 也是因为p1vec是取过log后的  log p^a = a log p  出现n词的概率p^n 变为 n log p
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest():
    docList = []
    classList = []
    fullText = []
    # 遍历25个txt文件
    for i in range(1, 26):
        # 读取每个垃圾邮件，并以字符串转换为字符列表
        wordList = textParse(open('data/email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        # 标记垃圾邮件，1代表垃圾邮件
        classList.append(1)
        # 读取每个非垃圾邮件，并以字符串转换成字符串列表
        wordList = textParse(open('data/email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        # 标记非垃圾邮件，0表示非垃圾文件
        classList.append(0)
    # 创建词汇表 不重复
    vocabList = creatrVocabList(docList)
    # 创建储存训练集的索引值的列表 和 测试集 的索引值的列表
    trainingSeet = list(range(50))
    testSet = []
    # 从50个邮件中，随机挑选除50个作为训练集，10个作为测试集
    for i in range(10):
        # 随机选择索引值，随机生成一个实数
        randIndex = int(random.uniform(0,len(trainingSeet)))
        # 添加测试集的索引值
        testSet.append(trainingSeet[randIndex])
        # 在训练集列表中删除添加到测试集的索引值
        del(trainingSeet[randIndex])
    # 创建训练集矩阵和测试集类别标签向量
    trainMat = []
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSeet:
        # 将生成的词集模型添加到训练集矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 将类别添加到训练集类别标签向量中
        trainClasses.append(classList[docIndex])
        # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # 错误分类计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 测试集的词集模型
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 如果分类错误
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 错误计数器加1
            errorCount += 1
            print("分类错误的测试集：", docList[docIndex])
    print("错误率：%.2f%%" % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
