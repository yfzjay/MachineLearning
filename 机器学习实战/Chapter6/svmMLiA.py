import random
import numpy as np
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        line=line.strip().split('\t')
        dataMat.append([float(line[0]),float(line[1])])
        labelMat.append(float(line[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAloha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

def svmSimple(dataMat,classLable,C,toler,maxIter):
    dataMat=np.mat(dataMat)
    lableMat=np.mat(classLable).T
    b=0;m,n=np.shape(dataMat)
    alpha=np.mat(np.zeros((m,1)))
    iter=0
    while iter<maxIter:
        alphachanged=0
        for i in range(m):
            fxi=float((np.multiply(alpha,lableMat).T)*(dataMat*dataMat[i].T))+b
            Ei=fxi-float(lableMat[i])
            print("E{} : {}".format(i,Ei))
            if  ( (float(lableMat[i])*Ei<-toler)and(float(alpha[i])<C ) )or( (float(lableMat[i])*Ei>toler)and(alpha[i]>0) ):
                j=selectJrand(i,m)
                fxj=float((np.multiply(alpha,lableMat).T)*(dataMat*dataMat[j].T))+b
                Ej=fxj-float(lableMat[j])
                aiold=alpha[i].copy()
                ajold=alpha[j].copy()
                if lableMat[i]!=lableMat[j]:
                    L=max(0,alpha[j]-alpha[i])
                    H=min(C,C+alpha[j]-alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i]-C)
                    H = min(C, alpha[j] + alpha[i])
                if L==H:
                    print("L==H");continue
                eta=-(2.0*dataMat[i]*dataMat[j].T)+(dataMat[i]*dataMat[i].T)+(dataMat[j]*dataMat[j].T)
                if eta<=0:
                    print("eta<=0");continue
                alpha[j]+=lableMat[j]*(Ei-Ej)/eta
                alpha[j]=clipAloha(alpha[j],H,L)
                if abs(alpha[j]-ajold<0.00001):
                    print("j not moving enough");continue
                alpha[i]+=lableMat[j]*lableMat[i]*(ajold-alpha[j])
                b1=b-Ei+float(-lableMat[i]*(alpha[i]-aiold)*dataMat[i]*dataMat[i].T-lableMat[j]*(alpha[j]-ajold)*dataMat[i]*dataMat[j].T)
                b2 = b - Ej + float(-lableMat[i] * (alpha[i] - aiold) * dataMat[i] * dataMat[j].T - lableMat[j] * (alpha[j] - ajold) *dataMat[j] * dataMat[j].T)

                if (alpha[i]>0)and(alpha[i]<C):
                    b=b1
                elif (alpha[j]>0)and(alpha[j]<C):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphachanged+=1
                print("iter: {} i: {} , pairs changed {}".format(iter,i,alphachanged))
        if alphachanged==0:
            iter+=1
        else:
            iter=0
        print("iter number : {}".format(iter))
    w=np.multiply(alpha,lableMat).T*dataMat
    return b,alpha,w

def plotBestFit(weight,b,index):
    import matplotlib.pyplot as plt
    weight=weight.T
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

if __name__=='__main__':
    dataMat,LabelMat=loadDataSet("../file/Ch06/testSet.txt")
    b,alpha,w=svmSimple(dataMat,LabelMat,0.8,0.001,40)
    print("b:{}".format(b))
    print(w)
    print("alpha:{}".format(alpha[alpha>0]))
    plotBestFit(w,b,np.where(alpha>0)[0])
