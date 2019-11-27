import numpy as np
def loadDataSet(filename):
    numFeat=len(open(filename).readline().strip().split('\t'))-1
    dataMat=[];lableMat=[]
    fr=open(filename)
    for line in fr.readlines():
        line=line.strip().split('\t')
        newline=[]
        for i in range(numFeat):
            newline.append(float(line[i]))
        dataMat.append(newline)
        lableMat.append(float(line[-1]))
    return dataMat,lableMat

def standRegres1(inX,inY):
    X=np.mat(inX);Y=np.mat(inY).T
    XTX=X.T*X
    if np.linalg.det(XTX)==0.0:
        print("false")
        return
    result=XTX.I*(X.T*Y)
    return result
def standRegres2(inX,inY,times=200):
    X=np.mat(inX);Y=np.mat(inY).T
    w=np.ones((X.shape[1],1))
    a=0.001
    for i in range(times):
        error=X*w-Y
        w=w-a*X.T*error
    return w
def plotGraph(inX,inY,ws):
    import matplotlib.pyplot as plt
    inX=np.array(inX);inY=np.array(inY)
    ax=plt.subplot(111)
    ax.scatter(inX[:,1],inY[:],c='red')
    Xcopy=inX.copy()
    Xcopy=Xcopy[np.argsort(Xcopy[:,1])]
    Xmat=np.mat(Xcopy)
    Yhat=np.array(Xmat*ws)
    ax.plot(Xcopy[:,1],Yhat,color='green',LineWidth='4')
    plt.show()
if __name__=='__main__':
    inX,inY=loadDataSet("../file/Ch08/ex0.txt")
    ws1=standRegres1(inX,inY)
    ws2 = standRegres2(inX, inY,200)
    print(ws1)
    print(ws2)
    plotGraph(inX,inY,ws1)
    plotGraph(inX, inY, ws2)