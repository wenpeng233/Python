# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


# 实现简单Adaline神经元
class AdalineGD(object):
    '''
    Paramters
    ------------

    '''

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # 计算代价(损失)
            cost = (errors**2).sum()/2.0
            # 记录每次迭代的代价(损失)
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def activation(self, X):
        return self.net_input(X)


    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


if __name__ == "__main__":
    # 绘制学习率曲线(参考书：Python机器学习第二章P23)
    df = pd.read_csv('./iris.data')
    y = df.iloc[:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)
    # y
    X = df.iloc[:100, [0, 2]].values


    # subplots将画布分割为1行2列大小为8x4的小画布
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # 以学习率为0.01实现学习算法，绘制迭代次数和代价的函数图
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1),
               np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum -squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    # 以学习率0.0001实现学习算法，绘制迭代次数和代价的函数图
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1),
               ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum -squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

    # 特征值缩放，mean（）函数求平均值，std（）函数求标准差
    X_std = np.copy(X)
    X[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()