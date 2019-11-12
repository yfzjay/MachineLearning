import numpy as np
import random
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        line=line.strip().split('\t')
        dataMat.append([float(line[0]),float(line[1])])
        labelMat.append(float(line[2]))
    return dataMat,labelMat
class optStruct:
    def __init__(self,dataMat,LabelMat,C,tolor):
        self.X=dataMat
        self.Y=LabelMat
        self.C=C
        self.tol=tolor
        self.m=np.shape(dataMat)[0]
        self.alpha=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))
def calcEk(OS,k):
    fxk=float(np.multiply(OS.alpha,OS.Y).T*(OS.X*OS.X[k,:].T))+OS.b
    Ek=fxk-float(OS.Y[k])
    return Ek
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j
def selectJ(i,OS,Ei):
    maxj=-1
    maxDE=0
    Ej=0
    OS.eCache[i]=[1,Ei]#只要选择了一个i需要更新，就给他赋上Ei
    validE=np.nonzero(OS.eCache[:,0].A)[0]
    if len(validE)>1:
        for k in validE:
            if k==i:
                continue
            Ek=calcEk(OS,k)
            tmpDE=abs(Ei-Ek)
            if tmpDE>maxDE:
                maxj=k;maxDE=tmpDE;Ej=Ek
        return maxj,Ej
    else:
        maxj=selectJrand(i,OS.m)
        Ej=calcEk(OS,maxj)
    return maxj,Ej
def updateEk(OS,k):
    Ek=calcEk(OS,k)
    OS.eCache[k]=[1,Ek]
def clipAloha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj
def innerL(i,OS):
    Ei=calcEk(OS,i)
    if ((OS.Y[i]*Ei<-OS.tol)and(OS.alpha[i]<OS.C)) or ((OS.Y[i]*Ei>OS.tol)and(OS.alpha[i]>0)):
        j,Ej=selectJ(i,OS,Ei)
        aiold=OS.alpha[i].copy()
        ajold=OS.alpha[j].copy()
        if OS.Y[i]!=OS.Y[j]:
            L=max(0,OS.alpha[j]-OS.alpha[i])
            H=min(OS.C,OS.C+OS.alpha[j]-OS.alpha[i])
        else:
            L = max(0, OS.alpha[j] + OS.alpha[i]-OS.C)
            H = min(OS.C,  OS.alpha[j] + OS.alpha[i])
        if L==H:
            print("L==H")
            return 0
        eta = -(OS.X[i] * OS.X[i].T) - (OS.X[j] * OS.X[j].T)+(2.0 * OS.X[i] * OS.X[j].T)
        if eta >= 0:
            print("eta<=0")
            return 0
        OS.alpha[j] -= OS.Y[j] * (Ei - Ej) / eta
        OS.alpha[j] = clipAloha(OS.alpha[j], H, L)
        updateEk(OS, j)
        if abs(OS.alpha[j] - ajold < 0.00001):
            print("j not moving enough")
            return 0
        OS.alpha[i] += OS.Y[j] * OS.Y[i] * (ajold - OS.alpha[j])
        updateEk(OS,i)
        b1 = OS.b - Ei +float(- OS.Y[i] * (OS.alpha[i] - aiold) * OS.X[i] * OS.X[i].T - OS.Y[j] *(OS.alpha[j] - ajold) * OS.X[i] * OS.X[j].T)
        b2 = OS.b - Ej +float(- OS.Y[i] * (OS.alpha[i] - aiold) * OS.X[i] * OS.X[j].T - OS.Y[j] *(OS.alpha[j] - ajold) * OS.X[j] * OS.X[j].T)
        if (OS.alpha[i] > 0) and (OS.alpha[i] < OS.C):
            OS.b = b1
            #bf=b1f
        elif (OS.alpha[j] > 0) and (OS.alpha[j] < OS.C):
            OS.b = b2
            #bf=b2f
        else:
            OS.b = (b1 + b2) / 2.0
            #bf=(b1f + b2f) / 2.0

        return 1

    else:
        return 0
def smoP(dataMat,labelMat,C,tolor,maxIter):
    OS=optStruct(np.mat(dataMat),np.mat(labelMat).transpose(),C,tolor)
    iter=0
    entireSet=True
    alphaChanged=0
    while (iter<maxIter)and((alphaChanged>0)or(entireSet)):
        alphaChanged=0
        if entireSet:
            for i in range(OS.m):
                alphaChanged+=innerL(i,OS)
                print("fullset,iter:{},i:{},pairs changed{}".format(iter,i,alphaChanged))
            iter+=1
        else:
            nonBound=np.nonzero((OS.alpha.A>0)*(OS.alpha.A<C))[0]
            for i in nonBound:
                alphaChanged+=innerL(i,OS)
                print("non-bound-set,iter:{},i:{},pairs changed{}".format(iter, i, alphaChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif alphaChanged==0:
            entireSet=True
        print ("iteration number:{}".format(iter))
    #w = (np.multiply(OS.alpha, OS.Y).T * OS.X).T
    return OS.b,OS.alpha

def plotBestFit(weight,b,index):
    import matplotlib.pyplot as plt
    inX,inY=loadDataSet("../file/Ch06/testSet.txt")
    inX=np.array(inX)
    n=inX.shape[0]
    xcord1=[];ycord1=[]
    xcord2=[]; ycord2 = []
    xcord3 = [];ycord3 = []
    for i in range (n):
        if i not in index:
            if inY[i] == 1:
                xcord1.append(inX[i, 0]);
                ycord1.append(inX[i, 1])
            else:
                xcord2.append(inX[i, 0]);
                ycord2.append(inX[i, 1])
        else:
            xcord3.append(inX[i, 0]);
            ycord3.append(inX[i, 1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=10,c='red',marker='s')
    ax.scatter(xcord2, ycord2, s=10, c='green')
    ax.scatter(xcord3, ycord3, s=20, c='blue',marker='*')
    x=np.arange(-1,10,0.1)
    y=(-b-weight[0]*x)/weight[1]
    y=y.T
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
        # 虽然该过程遍历所有的样本点，但是通过前面的原理详解可知，大部分的alpha值为0，只有支持向量的值不为零
        # 所以这个计算还是很简单的
    return w

if __name__=='__main__':
    dataMat,LabelMat=loadDataSet("../file/Ch06/testSet.txt")
    b,alpha=smoP(dataMat,LabelMat,0.6,0.001,40)
    print("b:{}".format(b))
    print("alpha:{}".format(alpha))
    ws=calcWs(alpha,dataMat,LabelMat)
    print("ws:",ws)
    plotBestFit(ws,b,np.where(alpha.A>0)[0])
    err=0
    datamat=np.mat(dataMat)
    for i in range(len(LabelMat)):
        y = datamat[i] * ws + b
        if y > 0:
            y = 1
        else:
            y = -1
        if y != LabelMat[i]:
            err += 1
    print('err:',err)
    print('error rate:',err/len(LabelMat))

