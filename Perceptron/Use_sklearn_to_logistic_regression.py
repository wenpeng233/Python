# coding=utf-8
# 加载scikit-learn自带的数据集模块，其中包含鸢尾花数据集
from sklearn import datasets
import numpy as np
# 加载scikit-learn自带的分割数据集的工具模块,新版的分割工具在model_selection中了
from sklearn.model_selection import train_test_split
# 加载scikit-learn可以进行特征缩放的模块
from sklearn.preprocessing import StandardScaler
# 加载scikit-learn自带的学习算法模块,LogisticRegression是用逻辑斯蒂回归函数作为激活函数的感知机模型
from sklearn.linear_model import LogisticRegression


# 加载鸢尾花数据集，这里只取2个特征
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 分割数据集为训练数据集和测试数据集,分割比例为7：3，随机数种子为0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 进行特征缩放，训练集和测试集缩放用的参数一样，保证二者之间有一致性
sc = StandardScaler()
# 这一步调用fit（）,会计算传入的参数X_train的每个特征的均值和标准差
sc.fit(X_train)
# 对特征进行缩放，缩放的参数一致，保证两数据集一致性
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 初始化学习算法类,新版本不再用 n_iter参数,改为了max_iter了
lr = LogisticRegression(C=1000.0, random_state=0)
# 调用fit()方法训练模型
lr.fit(X_train_std,y_train)

# 使用训练出来的模型进行预测
y_pred = lr.predict(X_test_std)
# 统计预测错误的数量
print("Misclassified samples: %d" % (y_test != y_pred).sum())


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 绘制决策线
def plot_decision_region(X, y, classifer, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

    Z = classifer.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 画出所有的样本
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # 画出测试样本，是测试样本高亮
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='test set')

# 将训练数据和测试数据组合在一起
# np.vstack是将元组里的数组堆叠起来，按V  水平的方式，
X_combined_std = np.vstack((X_train_std,X_test_std))
# np.vstack是将元组里的数组堆叠起来，按H  垂直的方式，
y_combined_std = np.hstack((y_train, y_test))

plot_decision_region(X=X_combined_std, y=y_combined_std, classifer=lr, test_idx=range(105,150))

plt.xlabel('petal length [Standardized]')
plt.ylabel('petal width [Standardized]')
plt.legend(loc='upper left')
plt.show()