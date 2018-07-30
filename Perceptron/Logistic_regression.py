# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import seed
from matplotlib.colors import ListedColormap


# 利用随机梯度下降的算法实现Adaline神经元，这样，每次迭代是用一个样本来更新权值，而不是像以前那样批量更新权值（指所有的样本），这样更新权值更频繁，更容易收敛
# 也可以用小批次学习，它是介于一面二者之间的，每次迭代用一小部分的样本来更新权值，这样不但收敛更快，比每次只用一个样本计算效率更高。
class Logistic_regression(object):
    '''
    Paramters
    ------------

    '''

    def __init__(self, eta=0.01, n_iter=10,
                 shubffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shubffle = shubffle
        if random_state:
            seed(random_state)


    def fit(self, X, y):
        # 初始化权值
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # 如果打开随机获取样本的开关
            if self._shuffle:
                # 随机获取一个样本
                X, y = self._shuffle(X, y)
            cost = []

            # 遍历样本
            for xi, target in zip(X, y):
                # 更新权值，并记录每次更新的代价值
                cost.append(self._update_weights(xi, target))
            # 记录这次迭代的平均代价值
            avg_cost = sum(cost)/len(y)
            # 记录每次迭代的代价值
            self.cost_.append(avg_cost)
        return self



    # 此方法用于类似与在线学习，处理流数据时调用
    def partial_fit(self, X, y):
        # 第一次调用该方法时初始化权值
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        #  判断y中有多余1个数，则批量更新权值
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        # 若只有一个y值，只更新一次
        else:
            self._update_weights(X, y)
        return  self


    # 该函数用于随机取一个样本
    def _shuffle(self, X, y):
        # permetation函数用于打乱一个数列，若传入值是一个整数，返回打乱的range（整数）数列
        r = np.random.permutation(len(y))
        return X[r], y[r]


    # 初始化权值
    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True


    # 更新权值，并返回代价（损失）,xi是一个随机样本，target是xi对应的真实值
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.T.dot(error)
        self.w_[0] += self.eta * error.sum()
        cost = (error ** 2).sum()/2.0
        return cost


    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


    # 激活函数，此处改为logistic函数，或者说是sigmoid函数
    def activation(self, X):
        z = self.net_input(X)
        exp_ = np.exp(-z)
        return 1.0/(1.0 + exp_)


    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


if __name__ == "__main__":
    # 绘制学习率曲线(参考书：Python机器学习第二章P23)
    df = pd.read_csv('./iris.data')
    y = df.iloc[:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)
    # y
    X = df.iloc[:100, [0, 2]].values
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # 特征值缩放，mean（）函数求平均值，std（）函数求标准差
    X_std = np.copy(X)
    X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada1 = Logistic_regression(n_iter=10, eta=0.01,random_state=1).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1),
               np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum -squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ax[1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Average Cost')
    plt.show()


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada2 = Logistic_regression(n_iter=10, eta=0.01, random_state=1).fit(X_std, y)
    ax[0].plot(range(1, len(ada2.cost_) + 1),
               np.log10(ada2.cost_), marker='o')
    ax[0].set_xlabel('Epochs [X_std]')
    ax[0].set_ylabel('log(Sum -squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Average Cost')
    plt.show()



