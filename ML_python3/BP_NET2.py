import numpy as np

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
    return np.array(dataset), np.array(labelset)


def MSELoss(labelset, y):
    return (1 / 2) * (y - labelset) ** 2


class LinearLayer:
    def __init__(self, input_D, output_D):
        self._W = np.random.normal(0, 0.1, (input_D, output_D))  # 初始化不能为全0
        self._b = np.random.normal(0, 0.1, (1, output_D))
        self._grad_W = np.zeros((input_D, output_D))
        self._grad_b = np.zeros((1, output_D))

    def forward(self, X):
        return np.matmul(X, self._W) + self._b

    def backward(self, X, grad):
        self._grad_W = np.matmul(X.T, grad)
        self._grad_b = np.matmul(grad.T, np.ones(X.shape[0]))
        return np.matmul(grad, self._W.T)

    def update(self, learn_rate):
        self._W = self._W - self._grad_W * learn_rate
        self._b = self._b - self._grad_b * learn_rate


class Relu:
    def __init__(self):
        pass

    def forward(self, X):
        # np.where(condition, x, y)
        # 满足条件(condition)，输出x，不满足输出y。
        return np.where(X < 0, 0, X)

    def backward(self, X, grad):
        return np.where(X > 0, X, 0) * grad


def predict(model, X):
    tmp = X
    for layer in model:
        tmp = layer.forward(tmp)
    return np.where(tmp > 0.5, 1, 0)


def main():
    # 训练数据：经典的异或分类问题
    dataset, labelset = loaddataset('data/horseColicTraining.txt')

    # 初始化网络，总共2层，输入数据是21维，第一层21个节点，第二层1个节点作为输出层，激活函数使用Relu
    linear1 = LinearLayer(dataset.shape[1], dataset.shape[1])
    relu1 = Relu()
    linear2 = LinearLayer(dataset.shape[1], 1)

    # 训练网络
    for i in range(10000):

        # 前向传播Forward，获取网络输出
        o0 = dataset
        a1 = linear1.forward(o0)
        o1 = relu1.forward(a1)
        a2 = linear2.forward(o1)
        o2 = a2

        # 获得网络当前输出，计算损失loss
        y = o2.reshape(o2.shape[0])
        # loss = min(MSELoss(labelset, y))  # MSE损失函数

        # 反向传播，获取梯度
        grad = (y - labelset).reshape(dataset.shape[0], 1)
        grad = linear2.backward(o1, grad)
        grad = relu1.backward(a1, grad)
        grad = linear1.backward(o0, grad)

        learn_rate = 0.2  # 学习率

        # 更新网络中线性层的参数
        linear1.update(learn_rate)
        linear2.update(learn_rate)

        # 判断学习是否完成
        # if i % 200 == 0:
        #     print(loss)
        # if loss < 0.001:
        #     print("训练完成！ 第%d次迭代" % i)
        #     break
        # 将训练好的层打包成一个model
    model = [linear1, relu1, linear2]

    # 用训练好的模型去预测
    # 开始预测
    print("-----")
    result = predict(model, dataset)
    print(result)


if __name__ == '__main__':
    # test()
    main()
