import numpy
import operator


def create_data_set():
    group = numpy.array([[1, 0, 1, 1], [1, 0, 1, 0], [0, 0], [0, 1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    # 对公式|x| = √( x[1]2 + x[2]2 + … + x[n]2 ) 欧式距离公式的计算
    dataSetSize = dataSet.shape(0)
    diffmat = numpy.tile(inX, (dataSetSize, 1) - dataSet)
    sqDiffMat = diffmat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5

    #对算出来的距离进行递增次序排序
    sortedDitIndicies = distances.argsort()   #argsort返回排序后的指针

    #选择距离最小的k个点


    classCount = {}
    for i in k:
        voteIlabel = labels[sortedDitIndicies[i]]
        # 统计k个点中出现类别的频率
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #get  字典中返回指定key的值，若key不存在，则返回0处的值
    # 选择频率最高的返回
    sortedClassCount = sorted(classCount.iteritems(),                    #classCount.iteritems()返回一个classCount的迭代器
                              key=operator.itemgetter(1), reverse=True)  #operator.itemgetter(1)指迭代器的第一个值
                                                                         #sorted 中key是指classCount的迭代器中的元素与它第一个元素进行比较
                                                                         #reverse=True按降序排序
                                                                         # PS：这样用函数只用找出最大的一个值，不必把排序进行完
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = create_data_set()
    print(group)
    print(labels)

