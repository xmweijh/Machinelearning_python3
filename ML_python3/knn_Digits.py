import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN


def img2vector(filename):
    """
    将32*32的二进制图像转换为1*1024向量
    :param filename:文件名
    :return:返回二进制图像的1*1024向量
    """
    # 创建1*1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline()
        # 每一行的前32个数据依次存储到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1*1024向量
    return returnVect


def handwritingClassTest():
    """
    手写数字分类测试
    :return:
    """
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFilesList = listdir('data/trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFilesList)
    # 初始化训练的Mat矩阵（全零阵），测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFilesList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)
    # 构造kNN分类器
    neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合模型，trainingMat为训练矩阵，hwLabels为对应标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('data/testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1*1024向量，用于训练
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)
        # 获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


def main():
    handwritingClassTest()


if __name__ == '__main__':
    main()
