# -- coding: utf-8 --
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
def create_data_set():
    group = array([[1.0, 1.1],
                   [0.0, 0.09],
                   [0.5, 0.45],
                   [-1.4, -1.45],
                   [1.0, -0.9],
                   [-0.09, 0],
                   [-0.6, 0.5],
                   [1.4, -1.45]]
                  )
    labels = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile意为铺瓷砖,在这里是用inX以(dataSetSize,1)的规格铺成新矩阵,然后再减去dataSet矩阵
    sqDiffMat = diffMat ** 2  # 对矩阵中的元素求平方
    sqDistances = sqDiffMat.sum(axis=1)  # 对每个向量做元素的求和
    distances = sqDistances ** 0.5  # 再开根号,即可获得欧式距离
    sortedDisIndicies = distances.argsort()  # 获得升序的下标序列
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDisIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据第1个元素进行降序排序
    return sortedClassCount[0][0]


def file2matrix(filename, type='str'):
    fr = open(filename)
    arrayOLines = fr.readlines()  # 全部读取
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))  # 制作一个矩阵,先用全0来填充
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去掉头尾的多余空白符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 这里也可以写成returnMat[index],给returnMat[index]赋值为listFromLine[0:3]
        if type == 'str':
            classLabelVector.append(listFromLine[-1])
        elif type == 'int':
            classLabelVector.append(int(listFromLine[-1]))  # -1索引表示最后一个元素,这里最后一个元素是label
        index += 1
    return returnMat, classLabelVector


def test_plot(filename):
    datingDataMat, datingLabels = file2matrix(filename, 'int')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3d scatter graph')
    # add_subplot(mnp)添加子轴、图。subplot（m,n,p）或者subplot（mnp）此函数最常用：subplot是将多个图画到一个平面上的工具。其中，
    # m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，一共m行，如果第一个数字是2就是表示2行图。p是指你现
    # 在要把曲线画到figure中哪个图上，最后一个如果是1表示是从左到右第一个位置。

    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], datingDataMat[:, 2], marker='o',
               s=15.0 * array(datingLabels), c=15.0 * array(datingLabels))
    # 第一个15.0 * array(datingLabels)表示15的大小
    # 第二个15.0 * array(datingLabels)表示用15*datingLabels的列表做刻度，最低为紫色，最高为黄色

    plt.legend(['didn\'t like'])
    plt.show()


def auto_norm(dataSet):
    minVals = dataSet.min(0)  # 求得每列的最小值,即每个属性的最小值
    maxVals = dataSet.max(0)  # 求得每列的最大值,即每个属性的最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def dating_class_test(filename='datingTestSet.txt'):
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(filename)  # 获得训练集
    normMat, ranges, minVals = auto_norm(datingDataMat)  # 把训练数据归一化, 并且获得归一数据, max-min, min
    m = normMat.shape[0]  # 行数
    numTestVecs = int(m*hoRatio)  # 测试标签数
    errorCount = 0.0
    for i in range(numTestVecs):
        n = int(random.random() * (numTestVecs + 1))
        # 从0-0.1m中随机选一个数n, 以此为起点, 以0.9m+n为终点, 即选择了0.9m个数据为判定标准
        classifierResult = classify(normMat[i, :], normMat[n:m+n-numTestVecs, :], datingLabels[n:m+n-numTestVecs], 5)
        # 分类结果=classify(归一化数据[i], 归一化数据[n:m+n-numTestVecs], 对应标签[n:m+n-numTestVecs], 5)
        # 对归一化数据里的每个数据都进行分类
        print('the classifier came back with :%s, the real result is %s.' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) :  # 计算错误率
            errorCount += 1.0
    print('total error times is %d' % errorCount)
    print('total error rate is %f' % (errorCount / float(numTestVecs)))


def dating_class(filename, rate=0.1, k=3):
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = auto_norm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * rate)
    errorCount = 0.0
    for i in range(numTestVecs):
        n = int(random.random() * (numTestVecs + 1))
        # 从0-rate*m中随机选一个数n, 以此为起点, 以(1-rate)*m+n为终点, 即选择了(1-rate)*m个数据为判定标准
        classifierResult = classify(normMat[i, :], normMat[n:m+n-numTestVecs, :], datingLabels[n:m+n-numTestVecs], k)
        # 分类结果=classify(归一化数据[i], 归一化数据[n:m+n-numTestVecs], 对应标签[n:m+n-numTestVecs], k)
        # 对归一化数据里的每个数据都进行分类
        print('the classifier came back with :%s, the real result is %s.' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) :  # 计算错误率
            errorCount += 1.0
    print('total error times is %d' % errorCount)
    print('total error rate is %f' % (errorCount / float(numTestVecs)))


def person_class():
    percentage = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent filter miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    person = array([ffMiles, percentage, iceCream])
    dataMat, labels = file2matrix('datingTestSet.txt')
    dataMat, ranges, minVals = auto_norm(dataMat)
    person = (person - minVals) / ranges  # 把新数据用原有数据的标准进行归一化
    res = classify(person, dataMat, labels, 3)
    print('You will probably like this person : %s' % res)


def person_class_test():
    print('----------Dating Class Program----------')
    flag = True
    while(flag):
        person_class()
        s = input('\nDo you want to continue ?')
        s = s.strip().strip(' ').lower()
        if s[0] == 'n':
            flag = False
        else:
            print()


def img2vector(filename):
    '''returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec'''
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    numberOfItems = len(arrayOfLines[0].rstrip())
    vector = zeros((1, numberOfLines*numberOfItems))
    for i in range(numberOfLines):
        lineStr = arrayOfLines[i].rstrip()
        for j in range(numberOfItems):
            vector[0, 32*i+j] = int(lineStr[j])
    return vector


def hand_writing_class_test(trainPath='trainingDigits', testPath='testDigits', ratio=0.1):
    hwLabels = []
    traningFileList = os.listdir(trainPath)  # 打开训练集目录
    m = len(traningFileList)
    mTrain = int(ratio*m)
    trainingMat = zeros((mTrain, 1024))  # 训练矩阵,设定其shape为(ratio*m, 1024)
    for i in range(mTrain):  # 总共只取m*ratio个样本
        index = int(random.random() * m)  # 从0-m之间随机抽取一个文件
        fileNameStr = traningFileList[index]
        label = int(fileNameStr.split('.')[0].split('_')[0])  # 从文件名获得该文件的标签
        hwLabels.append(label)  # 把当前标签加入标签列表
        trainingMat[i, :] = img2vector('%s/%s' % (trainPath, fileNameStr))  # 把当前文件化成矩阵加入矩阵列表
    testFileList = os.listdir(testPath)  # 打开测试集目录
    errorCount = 0.0
    m = len(testFileList)
    mTest = int(ratio*m)
    for i in range(mTest):
        index = int(random.random() * m)
        fileNameStr = testFileList[index]
        label = int(fileNameStr.split('.')[0].split('_')[0])
        testVec = img2vector('%s/%s' % (testPath, fileNameStr))
        classResult = classify(testVec, trainingMat, hwLabels, 7)  # 用当前行矩阵和训练矩阵进行分类
        print('the classifier came back with : %d, the real answer is : %d' % (classResult, label))
        if label != classResult:
            errorCount += 1
    print('The total number of errors is %d' % errorCount)
    print('The error rate is %f' % (errorCount/float(mTest)))


