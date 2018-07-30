# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


class Perceptron(object):
    '''
    Paramters
    ------------

    '''

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)







if __name__ == "__main__":
    df = pd.read_csv('./iris.data')
    y = df.iloc[:100, 4]
    y = np.where(y == 'Iris-setosa', -1, 1)
    # y
    X = df.iloc[:100, [0, 2]].values
    # X.values?
    # X
    plt.scatter(X[:50, 0], X[:50, 1],
                c='red', marker='o', label='setosa',
                alpha=0.3
                )
    plt.scatter(X[50:100, 0], X[50:100, 1],
                c='blue', marker='x', label='virginica',
                alpha=0.3
                )
    plt.xlabel('花瓣长度(petal lenght)')
    plt.ylabel('萼片长度(sepal lenght)')
    plt.legend(loc='upper left')
    plt.show()


    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    # 绘制决策边界
    import Plot_decision_region

    Plot_decision_region.plot_decision_region(X, y, classifer=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()