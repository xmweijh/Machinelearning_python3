# 导入数据生成器
from sklearn.datasets import make_blobs
# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 生成样本数为200，分类为2的数据集
data = make_blobs(n_samples=200, centers=2, random_state=8)
X, y = data
# 将生成的数据集进行可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolors='k')
plt.show()

import numpy as np
# clf = KNeighborsClassifier()
# clf.fit(X, y)
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max()+1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max()+1
# # meshgrid 生成网格点坐标矩阵。
# # np.arange() 函数返回一个有终点和起点的固定步长的排列
# # 共同作用整个网络平面的用于后面预测画图
# xx, yy = np.meshgrid(np.arange(x_min, x_max, .2),
#                      np.arange(y_min, y_max, .2))
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel1)
# # cmap 搭配 c不同类别不同颜色
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolor='k')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("Classifier:KNN")
# plt.scatter(6.75, 4.82, marker='*', c='red', s=200)
# plt.show()
# print(clf.predict([[6.75, 4.821]]))

# 多元分类
# 生成样本数为500,分类数为5的数据集
# data2 = make_blobs(n_samples=500, centers=5, random_state=8)
# X2, y2 = data2
# # 用散点图将数据集进行可视化
# plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.spring, edgecolor='k')
# plt.show()
#
# clf = KNeighborsClassifier()
# clf.fit(X2, y2)
# # 下面的代码用于画图
# x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
# y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
#                      np.arange(y_min, y_max, .02))
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Pastel1)
# plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt. cm. spring, edgecolor='k')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("Classifier : KNN")
# plt.show()
#
# # 将模型的评分进行打印
# print('\n\n\n')
# print('代码运行结果: ')
# print('=========================')
# print('模型正确率: {: .2f}'.format(clf.score(X2, y2)))
# print('==============================')
# print('\n\n\n')


# 回归任务
# 导入make_ regression数据集生成器
from sklearn.datasets import make_regression

# 生成特征数量为1，噪音为50的数据集
X, y = make_regression(n_features=1, n_informative=1, noise=50, random_state=8)
# 用散点图将数据点进行可视化
plt.scatter(X, y, c='orange', edgecolor='k')
plt.show()

# 导入用于回归分析的KNN模型
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor()
# 用KINN模型拟合数据
reg.fit(X, y)
# 把预测结果用图像进行可视化
# 的-1被理解为unspecified value，意思是未指定为给定的。如果我只需要特定的行数，列数多少我无所谓，我只需要指定行数，那么列数直接用-1代替就行了，计算机帮我们算赢有多少列，反之亦然。
z = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.scatter(X, y, c='orange', edgecolor='k')
plt.plot(z, reg.predict(z), c='k', linewidth=3)
# 向图像添加标题
plt.title('KNN Regressor')
plt.show()

print('\n\n\n')
print('代码运行结果: ')
print('==============================')
print('模型评分: {: .2f}'.format(reg.score(X, y)))
print('\n\n\n')


from sklearn. neighbors import KNeighborsRegressor
# 减少模型的n_ neighbors参 数为2
reg2 = KNeighborsRegressor(n_neighbors=2)
reg2.fit(X, y)
# 重新进行可视化
plt.scatter(X, y, c='orange', edgecolor='k')
plt.plot(z, reg2.predict(z), c='k', linewidth=3)
plt.title('KNN Regressor: n neighbors=2')
plt.show()
print('\n\n\n')
print('代码运行结果: ')
print('==============================')
print('模型评分: {: .2f}'.format(reg2.score(X, y)))
print('\n\n\n')
