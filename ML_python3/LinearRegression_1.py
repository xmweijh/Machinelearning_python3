import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(filename):
    # 计算特征个数，由于最后一列为y值所以减一
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # 求矩阵的行列式
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    # .I求逆矩阵
    ws = (xTx.I) * (xMat.T) * yMat
    return ws


def plotDataSet():
    xArr, yArr = loadDataSet('data/ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    # 排序
    xCopy.sort(0)
    yHat = xCopy * ws
    # 以下两行是通过corrcoef函数比较预测值和真实值的相关性
    # corrcoef函数得到相关系数矩阵
    # 得到的结果中对角线上的数据是1.0，因为yMat和自己的匹配是完美的
    # 而yHat1和yMat的相关系数为0.98
    yHat1 = xMat * ws
    print(np.corrcoef(yHat1.T, yMat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:, 1], yHat, c='red')
    # 绘制样本点即
    # flatten返回一个折叠成一维的数组。但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的
    # 矩阵.A(等效于矩阵.getA())变成了数组
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plotDataSet()