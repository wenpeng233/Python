import numpy
import scipy.io
import sklearn.externals
import matplotlib.pyplot

import  re
# numpy.set_printoptions(suppress=True)

def loadData(file_path):
    dataSet = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            datas = line.strip("\n").strip(" ").split(" ")
            line = [data for data in map(float, datas)]
            dataSet.append(line)
        return dataSet


def getDataSet(input_file, output_file, paraterm_file):
    numpy.set_printoptions(suppress=True)
    X = numpy.array(loadData(input_file))
    y = numpy.array(loadData(output_file))
    theta = numpy.array(loadData(paraterm_file))
    return X, y, theta


def get_hypothesis(X, theta):
    print("1",type(numpy.dot(X, theta)))
    return numpy.dot(X, theta)


def lossFunction(X, y, theta):
    h = get_hypothesis(X, theta)
    dataSize = h.shape[0]
    Jval = 1.0/2*sum((h - y) ** 2)/dataSize
    matplotlib.pyplot.plot(theta , Jval)
    return Jval



def count_gradient(X, y, theta):
    h = get_hypothesis(X, theta)
    dataSize = h.shape[0]
    gradient = 1.0 * numpy.dot(X.T, h - y)/dataSize
    return gradient



def let_gradient_down(X,y,theta,alpha,iter):
    # try:
        for i in range(iter):
            gradient = count_gradient(X, y, theta)
            theta -= alpha * gradient


        sklearn.externals.joblib.dump(theta,"newTheta.m")
        # file = open("newTheta.mat","w")
        #
        # theta.dtype = "float16"
        # for w in theta:
        #     file.write(str(w))
        #     file.write("\n")
        # file.close()
        # numpy.set_printoptions(suppress=False)
        print("OK")

    # except Exception:
    #     print(Exception.msg)
    #     return False
    # return True



def use_the_model(area):
    X = numpy.ones((1,2))
    X[0,1] = area
    # print(X)
    new_theta = numpy.array(sklearn.externals.joblib.load("newTheta.m"))
    # 测试：输出保存的newTheta文件里的数据
    for i in new_theta:
        print(i)
        print(type(i))
    price = get_hypothesis(area, new_theta)
    return price[0]



if __name__ == "__main__":
    # X, y, theta = getDataSet(r"D:\Octave\workspace\function\\x.mat",
    #                          r"D:\Octave\workspace\function\y.mat",
    #                          r"D:\Octave\workspace\function\theta.mat")
    #
    # print(let_gradient_down(X, y, theta, 0.01, 50))
    # # if(let_gradient_down(X, y, theta, 0.1, 100)):
    # area = numpy.array(float(input("请输入要预测的房子面积：")))
    #
    # print(type(area))
    # print("计算机预测的房价为：", use_the_model(area))
    # # else:
    # #     print("meet some erro！！！")
    #
    # # 绘图
    # # fig = matplotlib.pyplot.figure()
    # Jval = lossFunction(X, y, theta)
    # # matplotlib.pyplot.plot((X,y,theta), Jval)

    regx = r"[a-zA-Z0-9]+@[a-zA-Z0-9]+(\.[a-zA-Z])+"
    str = input("请输入：")
    if re.search(regx,str,re.M|re.I):
        print(str)
    else:
        print("邮箱地址不正确")






