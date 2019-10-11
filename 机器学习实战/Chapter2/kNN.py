import numpy as np
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
##############################约会网站预测#################################
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
######KNN算法########
def classify0(inX,dataSet,labels,k=1):
    dataSetSize=dataSet.shape[0]
    diffMax=np.tile(inX,(dataSetSize,1))#把inX阔成(dataSetSize,1)形状（复制）
    diff_1=(diffMax-dataSet)**2
    diff_2=diff_1.sum(1)
    diff_3=diff_2**0.5
    dist=diff_3.argsort()
    class1={}
    for i in range(k):
        voteLabel=labels[dist[i]]
        class1[voteLabel]=class1.get(voteLabel,0)+1
    sorted_class=sorted(class1.items(),key=lambda x:x[1],reverse=True)
#    print(sorted_class)
    return sorted_class[0][0]

def file2matrix(filename):
    fr=open(filename)
    lines=fr.readlines()
    numberOflines=len(lines)
    resultMat=np.zeros((numberOflines,3))
    Labels=[]
    index=0
    for line in lines:
        line=line.strip().split('\t')
        resultMat[index,:]=line[0:3]
        Labels.append(int(line[-1]))
        index+=1
    return resultMat,Labels

def autoNorm(dataSet):
    minValue=dataSet.min(0)
    maxValue=dataSet.max(0)
    ranges=maxValue-minValue
    norm=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    norm=dataSet-np.tile(minValue,(m,1))
    norm=norm/np.tile(ranges,(m,1))
    return norm,ranges,minValue

def datingTest():
    Ratio=0.1
    Mat,Label=file2matrix("../file/Ch02/datingTestSet2.txt")
    normal,ranges,minValue=autoNorm(Mat)
    m=Mat.shape[0]
    Testnum=int(m*Ratio)
    Errornum=0
    for i in range(Testnum):
        TestResult=classify0(normal[i,:],normal[Testnum:m,:],Label[Testnum:m],4)
        print("the classfier came back with {0},the real answer is {1}".format(
            TestResult,Label[i]
        ))
        if(TestResult!=Label[i]):
            Errornum+=1
    print("the total error rate is {0}".format(Errornum/float(Testnum)))
def classifyperson():
    resultList=["not at all","small","perfect"]
    per_games=float(input("percentage of time spent playing video game?"))
    per_miles = float(input("frequent flier miles earned per year?"))
    per_ice = float(input("liters of ice cream consumed per year?"))
    inArr=np.array([per_miles,per_games,per_ice])
    Mat,Label=file2matrix("../file/Ch02/datingTestSet2.txt")
    normal,ranges,minValue=autoNorm(Mat)
    inArrnormal=(inArr-minValue)/ranges
    result=classify0(inArrnormal,normal,Label,3)
    print("you will probably like this person {0}".format(resultList[result-1]))
##################################手写体###############################
def handVectorfile(filename):
    result=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        line=fr.readline()
        for  j in range(32):
            result[0,i*32+j]=int(line[j])
    return result
def handWriteTest():
    Label=[]
    trainList = listdir("../file/Ch02/trainingDigits")
    num_train=len(trainList)
    Mat=np.zeros((num_train,1024))
    for i in range(num_train):
        Label.append(int(trainList[i][0]))
        dir="../file/Ch02/trainingDigits/{0}".format(trainList[i])
        Mat[i,:]=handVectorfile(dir)
    testList=listdir("../file/Ch02/testDigits")
    num_test=len(testList)
    error_count=0
    for i in range(num_test):
        realLabel=int(testList[i][0])
        dirr = "../file/Ch02/testDigits/{0}".format(testList[i])
        testMat=handVectorfile(dirr)
        testLabel=classify0(testMat,Mat,Label,3)
        print("the real number is {0},the classifier come back with {1}".format(
            realLabel,testLabel
        ))
        if(realLabel!=testLabel):
            error_count+=1
    print("The handwrite classifier error rate is {0}".format(error_count/float(num_test)))


if __name__=='__main__':
    # group,labels=createDataSet()
    # result=classify0([0,0],group,labels,3)
    # print(result)
    # Mat1,Label1=file2matrix("../file/Ch02/datingTestSet2.txt")
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.scatter(Mat1[:,0],Mat1[:,1],c=np.array(Label1))
    # plt.xlabel("fly",fontsize=20)
    # plt.ylabel("game",fontsize=20)
    # plt.axis([0,100000,0,25])
    # plt.show()
    # normValue,ranges,minValues=autoNorm(Mat1)
    # print(normValue)
    # print(minValues)
    #datingTest()
    classifyperson()
    #handWriteTest()


