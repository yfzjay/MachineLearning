import numpy as np
from math import log
def calShannonEnt(dataSet):
    numSet=len(dataSet)
    labelCount={}
    for line in dataSet:
        currentLabel=line[-1]
        labelCount[currentLabel]=labelCount.get(currentLabel,0)+1
    result=0.0
    for key in labelCount:
        prob=float(labelCount[key])/numSet
        result -= prob*log(prob,2)
    return result

def splitSet(dataSet,axis,value):
    result=[]
    for line in dataSet:
        if line[axis]==value:
            temp=line[:axis]
            temp.extend(line[axis+1:])
            result.append(temp)
    return result

def chooseBestFeature(dataSet):
    numFeature=len(dataSet[0])-1
    bestEntro=calShannonEnt(dataSet)
    result=-1
    for i in range(numFeature):
        feaList=[line[i] for line in dataSet]
        feaList=set(feaList)
        newEno=0.0
        for value in feaList:
            subSet=splitSet(dataSet,i,value)
            prob=len(subSet)/float(len(dataSet))
            newEno+= prob * calShannonEnt(subSet)
        if(newEno<bestEntro):
            bestEntro=newEno
            result=i
    return result

def majorityCnt(classList):
    classCount={}
    for line in classList:
        classCount[line]=classCount.get(line,0)+1
    sortClass=sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortClass[0][0]

def createTree(dataSet,label):
    classList=[line[-1] for line in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFea=chooseBestFeature(dataSet)
    bestLabel=label[bestFea]
    myTree={bestLabel:{}}
    del(label[bestFea])
    feaValues=[line[bestFea] for line in dataSet]
    feaValues=set(feaValues)
    for value in feaValues:
        templabel=label[:]
        myTree[bestLabel][value]=createTree(splitSet(dataSet,bestFea,value),templabel)
    return myTree
def classify(inputTree,Label,Vec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    print(firstStr)
    print(Label)
    Index=Label.index(firstStr)
    result=""
    for key in secondDict:
        if key==Vec[Index]:
            if type(secondDict[key]).__name__=='dict':
                result=classify(secondDict[key],Label,Vec)
            else:
                result=secondDict[key]
    return result

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename,'rb+')
    return pickle.load(fr)

if __name__=='__main__':
    dataSet=[
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    label=['no surfacing','flippers']
    tree=createTree(dataSet,label)
    print(classify(tree,['no surfacing','flippers'],[1,1]))
    storeTree(tree,"classifierStorage.txt")
    print(grabTree('classifierStorage.txt'))