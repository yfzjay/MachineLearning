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
def kernelTrans(X,A,kTup):
    m,n=X.shape
    K=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j]-A
            K[j]=deltaRow*deltaRow.T
        K=np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem--That Kernel is not recognized')
    return K
class optStruct:
    def __init__(self,dataMat,LabelMat,C,tolor,kTup):
        self.X=dataMat
        self.Y=LabelMat
        self.C=C
        self.tol=tolor
        self.m=np.shape(dataMat)[0]
        self.alpha=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)
def calcEk(OS,k):
    fxk=float(np.multiply(OS.alpha,OS.Y).T*OS.K[:,k])+OS.b
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
        eta = -OS.K[i,i] - OS.K[j,j]+(2.0 * OS.K[i,j])
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
        b1 = OS.b - Ei +float(- OS.Y[i] * (OS.alpha[i] - aiold) * OS.K[i,i] - OS.Y[j] *(OS.alpha[j] - ajold) * OS.K[i,j])
        b2 = OS.b - Ej +float(- OS.Y[i] * (OS.alpha[i] - aiold) * OS.K[i,j] - OS.Y[j] *(OS.alpha[j] - ajold) * OS.K[j,j])
        if (OS.alpha[i] > 0) and (OS.alpha[i] < OS.C):
            OS.b = b1
        elif (OS.alpha[j] > 0) and (OS.alpha[j] < OS.C):
            OS.b = b2
        else:
            OS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
def smoP(dataMat,labelMat,C,tolor,maxIter,kTup=('lin',0)):
    OS=optStruct(np.mat(dataMat),np.mat(labelMat).transpose(),C,tolor,kTup)
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
    return OS.b,OS.alpha

def plotBestFit(index,filename):
    import matplotlib.pyplot as plt
    inX,inY=loadDataSet(filename)
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
    plt.show()
def testRBF(k1=1.3):
    dataMat, labelMat = loadDataSet("../file/Ch06/testSetRBF.txt")
    b,alpha=smoP(dataMat,labelMat,200,0.0001,10000,('rbf',k1))
    dataMat=np.mat(dataMat);labelMat=np.mat(labelMat).T
    svInd=np.nonzero(alpha.A>0)[0]#训练集中支持向量的索引
    X=dataMat[svInd]#训练集得到的支持向量
    Y=labelMat[svInd]#训练集得到的支持向量对应的标签值
    alphas=alpha[svInd]#支持向量对应的a值，a.all()>0
    print("there are {} support vectors".format(len(svInd)))
    m,n=dataMat.shape
    errorCount=0
    for i in range(m):
        kxi=kernelTrans(X,dataMat[i,:],('rbf',k1))
        fxi=float(np.multiply(Y,alphas).T*kxi)+b
        if np.sign(fxi)!=np.sign(labelMat[i]):
            errorCount+=1
    print("the training error rate is: ",float(errorCount)/m)
    plotBestFit(svInd,"../file/Ch06/testSetRBF.txt")
    dataMat, labelMat = loadDataSet("../file/Ch06/testSetRBF2.txt")
    dataMat=np.mat(dataMat);labelMat=np.mat(labelMat).T
    m,n=dataMat.shape
    errorCount=0
    for i in range(m):
        kxi=kernelTrans(X,dataMat[i],('rbf',k1))
        fxi = float(np.multiply(Y, alphas).T * kxi) + b
        if np.sign(fxi) != np.sign(labelMat[i]):
            errorCount += 1
    print("the test error rate is: ", float(errorCount) / m)
if __name__=='__main__':
    testRBF()
