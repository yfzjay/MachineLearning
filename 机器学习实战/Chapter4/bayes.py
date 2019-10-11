import numpy as np
import re
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
def createVocalList(dataSet):
    vocalSet=set([])
    for line in dataSet:
        vocalSet|=set(line)
    return list(vocalSet)

def setwordsVec(vocalSet,inputSet):
    returnVec=[0]*len(vocalSet)
    for word in inputSet:
        if word in vocalSet:
            returnVec[vocalSet.index(word)]=1
        else:
            print("the word {} is not in the vocalSet!".format(word))
    return returnVec

def bagOfsetwordsVec(vocalSet,inputSet):
    returnVec=[0]*len(vocalSet)
    for word in inputSet:
        if word in vocalSet:
            returnVec[vocalSet.index(word)]+=1
        else:
            print("the word {} is not in the vocalSet!".format(word))
    return returnVec

def trainNB0(trainMap,trainLabel):
    numTrain=len(trainMap)
    numWords=len(trainMap[0])
    PA=sum(trainLabel)/float(numTrain)
    P0Num=np.ones(numWords);P1Num=np.ones(numWords)
    P0D=2.0;P1D=2.0;
    for i in range(numTrain):
        if trainLabel[i]==0:
            P0Num+=trainMap[i]
            P0D+=np.sum(trainMap[i])
        else:
            P1Num+=trainMap[i]
            P1D+=np.sum(trainMap[i])
    P1Vec=P1Num/P1D;P0Vec=P0Num/P0D
    return np.log(P0Vec),np.log(P1Vec),PA

def classifyNB(inputVec,P0V,P1V,PA):
    p0=sum(P0V*inputVec)+np.log(PA)
    p1 = sum(P1V * inputVec) + np.log(1-PA)
    if p0>p1:
        return 0
    else:
        return 1

def testNB():
    dataSet, Label = loadDataSet()
    vocalSet = createVocalList(dataSet)
    trainMap = []
    for line in dataSet:
        trainMap.append(setwordsVec(vocalSet, line))
    P0V, P1V, PA = trainNB0(trainMap, Label)
    test1=['love','my','dalmation']
    doc=np.array(setwordsVec(vocalSet,test1))
    print("{} classified as {}".format(test1,classifyNB(doc,P0V,P1V,PA)))
    test1=['stupid','garbage']
    doc = np.array(setwordsVec(vocalSet, test1))
    print("{} classified as {}".format(test1, classifyNB(doc, P0V, P1V, PA)))

def textParse1(bigString):

    #print("string::::::::{}".format(bigString))
    listOfsplit=re.split(r'\W',str(bigString))

    return [str.lower() for str in listOfsplit if len(str)>2]



def spamTest():
    import random
    docList=[];classList=[];
    for i in range(1,26):
        wordList=textParse1(open("../file/Ch04/email/ham/{}.txt".format(i),encoding='UTF-8').read())

        docList.append(wordList)
        classList.append(1)
        wordList = textParse1(open("../file/Ch04/email/spam/{}.txt".format(i),encoding='UTF-8').read())
        docList.append(wordList)
        classList.append(0)
    print(docList)
    vocalList=createVocalList(docList)
    trainSet=list(range(50));textSet=[]
    for i in range(10):
        randIndex=random.randint(0,len(trainSet)-1)
        #print(randIndex)
        del(trainSet[randIndex])
        textSet.append(randIndex)
    trainMat=[];trainclass=[];
    for index in trainSet:
        trainMat.append(bagOfsetwordsVec(vocalList,docList[index]))
        trainclass.append(classList[index])
    p0v,p1v,pa=trainNB0(trainMat,trainclass)
    errorCount=0
    for i in range(10):
        for index in textSet:
            testList = bagOfsetwordsVec(vocalList, docList[index])
            if classifyNB(np.array(testList), p0v, p1v, pa) != classList[index]:
                errorCount += 1
                print(docList[index])
    print("error count is {}".format(errorCount/float(10*len(textSet))))


if __name__=='__main__':
    # dataSet,Label=loadDataSet()
    # vocalSet=createVocalList(dataSet)
    # trainMap=[]
    # for line in dataSet:
    #     trainMap.append(setwordsVec(vocalSet,line))
    # P0V,P1V,PA=trainNB0(trainMap,Label)
    #testNB()
    spamTest()
