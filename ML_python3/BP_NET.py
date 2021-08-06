import numpy as np
import time

def loaddataset(filename):
    """
    加载数据集
    :param filename:文件路径
    :return: dataset:特征集
             labelset:标签集
    """
    # 数据集 一个二分类数据集：马疝病数据集  27特征
    # http://archive.ics.uci.edu/ml/datasets/Horse+Colic

    # 存放数据
    dataset = []

    # 存放标签
    labelset = []
    with open(filename) as fp:
        for i in fp.readlines():
            a = i.strip().split()

            # 每个数据行的最后一个是标签
            # 将属性缺失值替换为-1
            dataset.append([float(j) if j != '?' else -1 for j in a[:len(a) - 1]])
            labelset.append(int(float(a[-1])))
    return dataset, labelset


def parameter_initialization(x, y, z):
    """
    初始化各个参数
    :param x:输入层神经元个数
    :param y:隐层神经元个数
    :param z:输出层神经元个数
    :return:weight1 输入层与隐层的连接权重, weight2 隐层与输出层的连接权重, value1 隐层阈值, value2 输出层阈值
    """
    # 隐层阈值
    value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)

    # 输出层阈值
    value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)

    # 输入层与隐层的连接权重
    weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)

    # 隐层与输出层的连接权重
    weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)

    return weight1, weight2, value1, value2


def sigmoid(z):
    """
    激活函数
    :param z: 输入数据与权重相乘结果
    :return:[-1, 1]的数
    """
    # return 1 / (1 + np.exp(-z))
    # if z >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出  当z非常大时候
    #     return 1.0 / (1 + np.exp(-z))
    # else:
    #     # 将上式分子分母同乘以exp(z)得到
    #     return np.exp(z) / (1 + np.exp(z))
    z_ravel = z.ravel()  # 将numpy数组展平
    y = []
    for element in z_ravel.flat:
        if element >= 0:
            y.append(1.0 / (1 + np.exp(-element)))
        else:
            y.append(np.exp(element) / (np.exp(element) + 1))
    return np.mat(y).reshape(z.shape)
    # 对于数组
    # return .5 * (1 + np.tanh(.5 * z))


def Relu(z):
    return np.where(z < 0, 0, z)


def trainning(dataset, labelset, weight1, weight2, value1, value2):
    """
    训练网络
    :param dataset:特征集
    :param labelset:标签集
    :param weight1:输入层与隐层的连接权重
    :param weight2:隐层与输出层的连接权重
    :param value1:隐层阈值
    :param value2:输出层阈值
    :return:
    """
    # x为学习率
    x = 0.2
    for i in range(len(dataset)):
        # 输入数据
        inputset = np.mat(dataset[i]).astype(np.float64)

        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.float64)

        # 隐层输入
        input1 = np.dot(inputset, weight1).astype(np.float64)

        # 隐层输出
        output2 = sigmoid(input1 - value1).astype(np.float64)

        # 输出层输入
        input2 = np.dot(output2, weight2).astype(np.float64)

        # 输出层输出
        output3 = sigmoid(input2 - value2).astype(np.float64)

        # 更新公式由矩阵运算表示
        a = np.multiply(output3, 1 - output3)
        g = np.multiply(a, outputset - output3)
        b = np.dot(g, np.transpose(weight2))
        c = np.multiply(output2, 1 - output2)
        e = np.multiply(b, c)

        value1_change = -x * e
        value2_change = -x * g
        weight1_change = x * np.dot(np.transpose(inputset), e)
        weight2_change = x * np.dot(np.transpose(output2), g)

        # 更新参数
        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change
    return weight1, weight2, value1, value2


def testing(dataset, labelset, weight1, weight2, value1, value2):
    # 记录预测正确的个数
    rightcount = 0
    for i in range(len(dataset)):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)

        # 确定其预测标签
        if output3 > 0.5:
            flag = 1
        else:
            flag = 0
        if labelset[i] == flag:
            rightcount += 1
        # 输出预测结果
        print("预测为%d   实际为%d" % (flag, labelset[i]))
    # 返回正确率
    return rightcount / len(dataset)


def main():
    begin_time = time.time()
    dataset, labelset = loaddataset('data/horseColicTraining.txt')
    weight1, weight2, value1, value2 = parameter_initialization(len(dataset[0]), 2*len(dataset[0]), 1)
    for i in range(1500):
        weight1, weight2, value1, value2 = trainning(dataset, labelset, weight1, weight2, value1, value2)
    rate = testing(dataset, labelset, weight1, weight2, value1, value2)
    print(f'训练集正确率为{rate}')
    # dataSet2, labelSet2 = loaddataset('data/horseColicTest.txt')
    # rate2 = testing(dataSet2, labelSet2, weight1, weight2, value1, value2)
    # print(f'测试集准确率为{rate2}')
    end_time = time.time()
    run_time = end_time - begin_time
    print(f'程序运行时间{run_time}')


def test():
    dataSet, labelSet = loaddataset('data/horseColicTraining.txt')
    print(dataSet)
    print(labelSet)


if __name__ == '__main__':
    main()
    # test()
