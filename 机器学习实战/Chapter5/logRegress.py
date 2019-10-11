import numpy as np
import math
import random
def loadDataSet():
    Mat=[];Label=[]
    fr=open("../file/Ch05/testSet.txt")
    for line in fr.readlines():
        line=line.strip().split('\t')
        Mat.append([1.0,float(line[0]),float(line[1])])
        Label.append(int(line[2]))
    return Mat,Label

def sigmoid(inX):

        return np.exp(inX)/(1+np.exp(inX))

def gradAscent(inX,inY):#都转成矩阵形式，一是为了矩阵乘法方便，二是array单提出一行变成(m,)形状
    inX=np.mat(inX)
    inY=np.mat(inY).T
    n,m=np.shape(inX)
    alpth=0.001
    maxCycle=500
    weight=np.ones((m,1))
    for i in range(maxCycle):
        error=inY-sigmoid(inX*weight)
        weight=weight+alpth*(inX.T*error)
    return weight

def GradAscent0(inX,inY):
    inX = np.mat(inX)
    inY = np.mat(inY).T
    n, m = np.shape(inX)
    alpth = 0.001
    maxCycle = 1000
    weight = np.ones((m, 1))
    for i in range(maxCycle):
        k=random.randint(0,n-1)
        error = inY[k] - sigmoid(inX[k] * weight)
        weight = weight + alpth * n *(inX[k].T * error)
    return weight

def gradAscent1(inX,inY,num=200):
    inX=np.mat(inX)
    inY=np.mat(inY).T
    n,m=np.shape(inX)
    weight=np.mat(np.ones((m,1)))
    for j in range(num):
        for i in range(n):
            index=random.randint(0,n-1-i)
            alpha=4/(1.0+j+i)+0.01
            error = inY[index] - sigmoid(inX[index] * weight)
            weight = weight + alpha * (error * inX[index]).T

    return weight

def plotBestFit(weight):
    import matplotlib.pyplot as plt
    inX,inY=loadDataSet()
    inX=np.array(inX)
    n=inX.shape[0]
    xcord1=[];ycord1=[]
    xcord2=[]; ycord2 = []
    for i in range (n):
        if(inY[i]==1):
            xcord1.append(inX[i,1]);ycord1.append(inX[i,2])
        else:
            xcord2.append(inX[i, 1]);ycord2.append(inX[i, 2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=50,c='red',marker='s')
    ax.scatter(xcord2, ycord2, s=50, c='green')
    x=np.arange(-3,3,0.1)
    y=(-weight[0]-weight[1]*x)/weight[2]
    y=y.T
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

def classifyResult(inX,weight):
    inX=np.mat(inX)
    weight=np.mat(weight)
    prob=sigmoid(inX*weight)
    if prob>0.5:
        return 1
    else:
        return 0

def colicTest():
    frTrain = open('../file/Ch05/horseColicTraining.txt');
    frTest = open('../file/Ch05/horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscent1(trainingSet, trainingLabels, 1500)
    plotBestFit(trainWeights)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyResult(lineArr, trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: {}".format( errorRate))
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after {0} iterations the average error rate is: {1}".format(numTests, errorSum / float(numTests)))


if __name__=='__main__':
     Mat,Label=loadDataSet()
     weight=gradAscent(Mat,Label)
    # print(weight)
     plotBestFit(weight)
    # weight=GradAscent0(Mat,Label)
    # print(weight)
    # plotBestFit(weight)
     weight=gradAscent1(Mat,Label)
    # print(weight)
     plotBestFit(weight)
    #multiTest()