import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit
import time


class LossFun:
    """
    损失函数
    """
    def __init__(self, lf_type="least_square"):
        self.name = "loss function"
        self.type = lf_type

    def cal(self, t, z):
        loss = 0
        if self.type == "least_square":
            loss = self.least_square(t, z)
        return loss

    def cal_deriv(self, t, z):
        delta = 0
        if self.type == "least_square":
            delta = self.least_square_deriv(t, z)
        return delta

    def least_square(self, t, z):
        zsize = z.shape
        sample_num = zsize[1]
        return np.sum(0.5 * (t - z) * (t - z) * t) / sample_num

    def least_square_deriv(self, t, z):
        return z - t


class ActivationFun:
    """
    激活函数
    """
    def __init__(self, atype="sigmoid"):
        self.name = "activation function library"
        self.type = atype

    def cal(self, a):
        z = 0
        if self.type == "sigmoid":
            z = self.sigmoid(a)
        elif self.type == "relu":
            z = self.relu(a)
        return z

    def cal_deriv(self, a):
        z = 0
        if self.type == "sigmoid":
            z = self.sigmoid_deriv(a)
        elif self.type == "relu":
            z = self.relu_deriv(a)
        return z

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def sigmoid_deriv(self, a):
        fa = self.sigmoid(a)
        return fa * (1 - fa)

    def relu(self, a):
        idx = a <= 0
        # a[idx] = 0.1 * a[idx]
        a[idx] = 0
        return a  # np.maximum(a, 0.0)

    def relu_deriv(self, a):
        # print a
        a[a > 0] = 1.0
        # a[a <= 0] = 0.1
        a[a <= 0] = 0
        # print a
        return a


class Layer:
    """
    每一层
    """
    def __init__(self, num_neural, af_type="sigmoid", learn_rate=0.5):
        self.af_type = af_type  # active function type
        self.learn_rate = learn_rate
        self.num_neural = num_neural
        self.dim = None
        self.W = None

        self.a = None
        self.X = None
        self.z = None
        self.delta = None
        self.theta = None
        self.act_fun = ActivationFun(self.af_type)

    def fp(self, X):
        self.X = X
        xsize = X.shape
        self.dim = xsize[0]
        self.num = xsize[1]

        if self.W is None:
            # self.W = np.random.random((self.dim, self.num_neural))-0.5
            # self.W = np.random.uniform(-1,1,size=(self.dim,self.num_neural))
            if self.af_type == "sigmoid":
                self.W = np.random.normal(0, 1, size=(self.dim, self.num_neural)) / np.sqrt(self.num)
            elif self.af_type == "relu":
                self.W = np.random.normal(0, 1, size=(self.dim, self.num_neural)) * np.sqrt(2.0 / self.num)
        if self.theta is None:
            # self.theta = np.random.random((self.num_neural, 1))-0.5
            # self.theta = np.random.uniform(-1,1,size=(self.num_neural,1))

            if self.af_type == "sigmoid":
                self.theta = np.random.normal(0, 1, size=(self.num_neural, 1)) / np.sqrt(self.num)
            elif self.af_type == "relu":
                self.theta = np.random.normal(0, 1, size=(self.num_neural, 1)) * np.sqrt(2.0 / self.num)
        # calculate the foreward a
        self.a = self.W.T.dot(self.X)
        # calculate the foreward z
        self.z = self.act_fun.cal(self.a)
        return self.z

    def bp(self, delta):
        self.delta = delta * self.act_fun.cal_deriv(self.a)
        self.theta = np.array([np.mean(self.theta - self.learn_rate * self.delta, 1)]).T  # 求所有样本的theta均值
        dW = self.X.dot(self.delta.T) / self.num
        self.W = self.W - self.learn_rate * dW
        delta_out = self.W.dot(self.delta)
        return delta_out


class BpNet:
    """
    神经网络整体
    """
    def __init__(self, net_struct, stop_crit, max_iter, batch_size=10):
        self.name = "net work"
        self.net_struct = net_struct
        if len(self.net_struct) == 0:
            print("没有指定层数!")
            return

        self.stop_crit = stop_crit
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.layers = []
        self.num_layers = 0
        # 创建网络
        self.create_net(net_struct)
        self.loss_fun = LossFun("least_square")

    def create_net(self, net_struct):
        self.num_layers = len(net_struct)
        for i in range(self.num_layers):
            self.layers.append(Layer(net_struct[i][0], net_struct[i][1], net_struct[i][2]))

    def train(self, X, t, Xtest=None, ttest=None):
        eva_acc_list = []
        eva_loss_list = []

        xshape = X.shape
        num = xshape[0]
        dim = xshape[1]

        for k in range(self.max_iter):
            # i = random.randint(0,num-1)
            # 随机选择batch_size个
            idxs = random.sample(range(num), self.batch_size)
            xi = np.array([X[idxs, :]]).T[:, :, 0]
            ti = np.array([t[idxs, :]]).T[:, :, 0]
            # 前向计算
            zi = self.fp(xi)

            # 偏差计算
            delta_i = self.loss_fun.cal_deriv(ti, zi)

            # 反馈计算
            self.bp(delta_i)

            # 评估精度
            if Xtest is not None:
                if k % self.stop_crit == 0:
                    [eva_acc, eva_loss] = self.test(Xtest, ttest)
                    eva_acc_list.append(eva_acc)
                    eva_loss_list.append(eva_loss)
                    print("%4d,%4f,%4f" % (k, eva_acc, eva_loss))
            else:
                print("%4d" % k)
        return [eva_acc_list, eva_loss_list]

    def test(self, X, t):
        xshape = X.shape
        num = xshape[0]
        z = self.fp_eval(X.T)
        t = t.T
        # 比较在相同列上的元素的最大返回下标
        est_pos = np.argmax(z, 0)
        real_pos = np.argmax(t, 0)
        corrct_count = np.sum(est_pos == real_pos)
        acc = 1.0 * corrct_count / num
        loss = self.loss_fun.cal(t, z)
        # print "%4f,loss:%4f"%(loss)
        return [acc, loss]

    def fp(self, X):
        z = X
        for i in range(self.num_layers):
            z = self.layers[i].fp(z)
        return z

    def bp(self, delta):
        z = delta
        for i in range(self.num_layers - 1, -1, -1):
            z = self.layers[i].bp(z)
        return z

    def fp_eval(self, X):
        layers = self.layers
        z = X
        for i in range(self.num_layers):
            z = layers[i].fp(z)
        return z


def z_score_normalization(x):
    # z-score归一化
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    return x


def plot_curve(data, title, lege, xlabel, ylabel):
    num = len(data)
    idx = range(num)
    plt.plot(idx, data, color="r", linewidth=1)

    plt.xlabel(xlabel, fontsize="xx-large")
    plt.ylabel(ylabel, fontsize="xx-large")
    plt.title(title, fontsize="xx-large")
    plt.legend([lege], fontsize="xx-large", loc='upper left');
    plt.show()


if __name__ == "__main__":
    begin_time = time.time()
    # dataset = []
    # # 存放标签
    # labelset = []
    # with open('data/horseColicTraining.txt') as fp:
    #     for i in fp.readlines():
    #         a = i.strip().split()
    #         # 每个数据行的最后一个是标签
    #         # 将属性缺失值替换为-1
    #         dataset.append([float(j) if j != '?' else -1 for j in a[:len(a) - 1]])
    #         labelset.append(int(float(a[-1])))
    raw_data = pd.read_csv('data/train.csv', header=0)
    data = raw_data.values
    dataset = data[:, 1:]
    labelset = data[:, 0]
    train_features, test_features, train_labels, test_labels = ttsplit(
        np.array(dataset), np.array(labelset), test_size=0.33, random_state=0)

    train_features = z_score_normalization(train_features)
    test_features = z_score_normalization(test_features)
    sample_num = train_labels.shape[0]
    # tr_labels = np.zeros([sample_num, 2])
    tr_labels = np.zeros([sample_num, 10])
    for i in range(sample_num):
        # 初始化训练标签
        tr_labels[i][train_labels[i]] = 1

    sample_num = test_labels.shape[0]
    # te_labels = np.zeros([sample_num, 2])
    te_labels = np.zeros([sample_num, 10])
    for i in range(sample_num):
        # 初始化测试标签
        te_labels[i][test_labels[i]] = 1

    print(train_features.shape)
    print(tr_labels.shape)
    print(test_features.shape)
    print(te_labels.shape)

    stop_crit = 100  # 停止
    max_iter = 10000  # 最大迭代次数
    batch_size = 100  # 每次训练的样本个数
    # 如定义一层隐藏层为100个神经元，输出层为类别数目神经元的网络结构，如下
    # net_struct = [[100, "relu", 0.1], [10, "sigmoid", 0.1]]  # 网络结构[[batch_size,active function, learning rate]]
    # net_struct = [ [10, "sigmoid", 0.1]]
    net_struct = [[100, "relu", 0.5], [100, "sigmoid", 0.5], [10, "sigmoid", 0.5]]
    bpNNCls = BpNet(net_struct, stop_crit, max_iter, batch_size)
    # train model

    [acc, loss] = bpNNCls.train(train_features, tr_labels, test_features, te_labels)
    # [acc, loss] = bpNNCls.train(train_features, tr_labels)
    print("training model finished")
    # create test data
    plot_curve(acc, "Bp Network Accuracy", "accuracy", "iter", "Accuracy")
    plot_curve(loss, "Bp Network Loss", "loss", "iter", "Loss")

    # test model
    [acc, loss] = bpNNCls.test(test_features, te_labels)
    print("test accuracy:%f" % acc)
    end_time = time.time()
    print(f'程序运行时间{end_time - begin_time}s')