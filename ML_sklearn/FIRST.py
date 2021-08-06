from sklearn.datasets import load_wine
# 导入数据集
wine_dataset = load_wine()
# print(wine_dataset.keys())
# print(f'数据概括{wine_dataset["data"].shape}')
# print(wine_dataset['DESCR'])


# 导入数据拆分工具
from sklearn.model_selection import train_test_split
# 将数据拆分为训练集与测试集合
# random_state为0活着缺省时，每次生成的伪随机数均不同，并凭借他对数据集进行拆分
X_train, X_test, y_train, y_test = train_test_split(
    wine_dataset['data'], wine_dataset['target'], random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# 导入KNN分类模型
from sklearn.neighbors import KNeighborsClassifier
# 指定模型的n_neighbors = 1（近邻的数量）
knn = KNeighborsClassifier(n_neighbors=1)
# 用模型对数据进行拟合(fit)
knn.fit(X_train, y_train)
# 测试数据得分
print(knn.score(X_test, y_test))
