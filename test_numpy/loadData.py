import numpy as np
import random


def loadData(fileName):
    dataSet = []
    with open(fileName) as fn:
        lines = fn.readlines()
        for line in lines:
            data = line.strip('\n').strip(' ').split(' ')
            line =[v for v in map(float, data)]
            # @Test
            # for v in line:
            #     print(v, 'and', type(v))
            # print("----")
            dataSet.append(line)
    return dataSet


if __name__ == "__main__":
    dataSet = loadData(r"D:\Octave\workspace\function\x.mat")
    print(dataSet)

